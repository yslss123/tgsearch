import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from pathlib import Path

from encoding import get_encoder

# Audio feature extractor
class AudioAttNet(nn.Module):
    def __init__(self, dim_aud=64, seq_len=8):
        super(AudioAttNet, self).__init__()
        self.seq_len = seq_len
        self.dim_aud = dim_aud
        self.attentionConvNet = nn.Sequential(  # b x subspace_dim x seq_len
            nn.Conv1d(self.dim_aud, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(8, 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(4, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(2, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True)
        )
        self.attentionNet = nn.Sequential(
            nn.Linear(in_features=self.seq_len, out_features=self.seq_len, bias=True),
            nn.Softmax(dim=1)
        )

    def forward(self, x, logit_bias=None):
        # x: [1, seq_len, dim_aud]. Default behavior is exactly E1/TG.
        y = self.attentionConvNet(x.permute(0, 2, 1))
        logits = self.attentionNet[0](y.view(1, self.seq_len))
        if logit_bias is not None:
            logits = logits + logit_bias.reshape(1, self.seq_len)
        weights = torch.softmax(logits, dim=1).view(1, self.seq_len, 1)
        return torch.sum(weights * x, dim=1)  # [1, dim_aud]


# Audio feature extractor
class AudioNet(nn.Module):
    def __init__(self, dim_in=29, dim_aud=64, win_size=16):
        super(AudioNet, self).__init__()
        self.win_size = win_size
        self.dim_aud = dim_aud
        self.encoder_conv = nn.Sequential(  # n x 29 x 16
            nn.Conv1d(dim_in, 32, kernel_size=3, stride=2, padding=1, bias=True),  # n x 32 x 8
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1, bias=True),  # n x 32 x 4
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1, bias=True),  # n x 64 x 2
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),  # n x 64 x 1
            nn.LeakyReLU(0.02, True),
        )
        self.encoder_fc1 = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(0.02, True),
            nn.Linear(64, dim_aud),
        )

    def forward(self, x):
        half_w = int(self.win_size/2)
        x = x[:, :, 8-half_w:8+half_w]
        x = self.encoder_conv(x).squeeze(-1)
        x = self.encoder_fc1(x)
        return x


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden, self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=False))

        self.net = nn.ModuleList(net)
    
    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
                # x = F.dropout(x, p=0.1, training=self.training)
                
        return x




class PhaseAwareTemporal(nn.Module):
    """E2: generic onset/offset-sensitive temporal aggregation.

    The features are phoneme-relative progress, start/end proximity, duration.
    Unlike E1's class embedding, this module changes attention logits and
    provides a center-token preservation path. All new contributions start
    at zero so E2 starts with exact E1 features for fixed E1 weights.
    """

    def __init__(self, dim_aud=32, phase_dim=4):
        super().__init__()
        self.phase_net = nn.Sequential(
            nn.Linear(phase_dim, 16),
            nn.ReLU(inplace=False),
            nn.Linear(16, dim_aud)
        )
        nn.init.zeros_(self.phase_net[-1].weight)
        nn.init.zeros_(self.phase_net[-1].bias)

        self.logit_net = nn.Linear(phase_dim, 1, bias=False)
        nn.init.zeros_(self.logit_net.weight)
        self.center_scale = nn.Parameter(torch.zeros(()))

    def forward(self, tokens, phase, phone_valid):
        if phase is None or phone_valid is None:
            raise ValueError("E2 requires phone_phase and phone_valid")
        if tokens.ndim != 2 or phase.shape != (tokens.shape[0], 4):
            raise ValueError("E2 expects tokens [T,D] and phase [T,4]")
        phase = phase.to(device=tokens.device, dtype=tokens.dtype)
        valid = phone_valid.reshape(-1).to(device=tokens.device, dtype=tokens.dtype)
        # All reserved/silence/unknown phones have exactly zero phase features.
        active = ((phase.abs().sum(-1) > 0).to(tokens.dtype) * valid)
        tokens = tokens + self.phase_net(phase) * active[:, None]
        attn_bias = self.logit_net(phase).squeeze(-1) * active
        # The central token is the target frame in TG's 8-frame context.
        center = tokens.shape[0] // 2
        center_gain = torch.tanh(self.center_scale) * active[center]
        return tokens, attn_bias, center_gain


class PhonemeFusion(nn.Module):
    """E1: add aligned phoneme embeddings to per-frame audio tokens.

    The projection is zero-initialized, so the module starts as the original
    acoustic-only model and can learn a phoneme residual during training.
    This is a simple fusion baseline, NOT phase-aware temporal modeling.
    """
    def __init__(self, num_phones, audio_dim, phone_dim=16):
        super().__init__()
        self.phone_embedding = nn.Embedding(num_phones, phone_dim, padding_idx=0)
        self.phone_proj = nn.Linear(phone_dim, audio_dim, bias=False)
        nn.init.zeros_(self.phone_proj.weight)

    def forward(self, audio_tokens, phone_ids, phone_valid):
        if phone_ids is None or phone_valid is None:
            raise ValueError("E1 requires both phone_ids and phone_valid")

        ids = phone_ids.reshape(-1).to(device=audio_tokens.device, dtype=torch.long)
        valid = phone_valid.reshape(-1).to(
            device=audio_tokens.device, dtype=audio_tokens.dtype
        )
        if audio_tokens.ndim != 2 or ids.numel() != audio_tokens.shape[0] or valid.numel() != audio_tokens.shape[0]:
            raise ValueError(
                "E1 requires audio_tokens [T,D], phone_ids [T], phone_valid [T]"
            )

        # Masked/unknown labels must have zero impact on acoustic features.
        phone_delta = self.phone_proj(self.phone_embedding(ids))
        return audio_tokens + phone_delta * valid.unsqueeze(-1)


def build_phoneme_fusion(args, audio_dim):
    if not getattr(args, "use_phoneme", False):
        return None

    vocab_file = getattr(args, "phone_vocab", "")
    if not vocab_file:
        csv_file = getattr(args, "phone_csv", "")
        if not csv_file:
            raise ValueError("E1 requires --phone_csv and a fixed phone vocabulary")
        vocab_file = str(Path(csv_file).parent / "phone_vocab.json")

    with open(Path(vocab_file).expanduser(), "r", encoding="utf-8") as f:
        vocab = json.load(f)
    ids = [int(v) for v in vocab.values()]
    if not ids or len(ids) != len(set(ids)) or min(ids) < 0:
        raise ValueError("Invalid phone vocabulary IDs in " + str(vocab_file))
    num_phones = max(ids) + 1
    print("[E1] Phoneme fusion enabled: {} symbols from {}".format(
        num_phones, vocab_file
    ))
    return PhonemeFusion(num_phones=num_phones, audio_dim=audio_dim)


class MotionNetwork(nn.Module):
    def __init__(self,
                 audio_dim = 32,
                 ind_dim = 0,
                 args = None,
                 ):
        super(MotionNetwork, self).__init__()

        if 'esperanto' in args.audio_extractor:
            self.audio_in_dim = 44
        elif 'deepspeech' in args.audio_extractor:
            self.audio_in_dim = 29
        elif 'hubert' in args.audio_extractor:
            self.audio_in_dim = 1024
        else:
            raise NotImplementedError
    
        self.bound = 0.15
        self.exp_eye = True

        
        self.individual_dim = ind_dim
        if self.individual_dim > 0:
            self.individual_codes = nn.Parameter(torch.randn(10000, self.individual_dim) * 0.1) 

        # audio network
        self.audio_dim = audio_dim
        self.audio_net = AudioNet(self.audio_in_dim, self.audio_dim)

        self.audio_att_net = AudioAttNet(self.audio_dim)
        self.phoneme_fusion = build_phoneme_fusion(args, self.audio_dim)
        self.use_phoneme = self.phoneme_fusion is not None
        self.use_phone_phase = bool(getattr(args, "use_phone_phase", False))
        if self.use_phone_phase and not self.use_phoneme:
            raise ValueError("E2 --use_phone_phase requires --use_phoneme")
        self.phase_temporal = PhaseAwareTemporal(self.audio_dim) if self.use_phone_phase else None

        # DYNAMIC PART
        self.num_levels = 12
        self.level_dim = 1
        self.encoder_xy, self.in_dim_xy = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=16, log2_hashmap_size=17, desired_resolution=256 * self.bound)
        self.encoder_yz, self.in_dim_yz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=16, log2_hashmap_size=17, desired_resolution=256 * self.bound)
        self.encoder_xz, self.in_dim_xz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=16, log2_hashmap_size=17, desired_resolution=256 * self.bound)

        self.in_dim = self.in_dim_xy + self.in_dim_yz + self.in_dim_xz


        self.num_layers = 3       
        self.hidden_dim = 64

        self.exp_in_dim = 6 - 1
        self.eye_dim = 6 if self.exp_eye else 0
        self.exp_encode_net = MLP(self.exp_in_dim, self.eye_dim - 1, 16, 2)

        self.eye_att_net = MLP(self.in_dim, self.eye_dim, 16, 2)

        # rot: 4   xyz: 3   opac: 1  scale: 3
        self.out_dim = 11
        self.sigma_net = MLP(self.in_dim + self.audio_dim + self.eye_dim + self.individual_dim, self.out_dim, self.hidden_dim, self.num_layers)
        
        self.aud_ch_att_net = MLP(self.in_dim, self.audio_dim, 32, 2)


    @staticmethod
    @torch.jit.script
    def split_xyz(x):
        xy, yz, xz = x[:, :-1], x[:, 1:], torch.cat([x[:,:1], x[:,-1:]], dim=-1)
        return xy, yz, xz


    def encode_x(self, xyz, bound):
        # x: [N, 3], in [-bound, bound]
        N, M = xyz.shape
        xy, yz, xz = self.split_xyz(xyz)
        feat_xy = self.encoder_xy(xy, bound=bound)
        feat_yz = self.encoder_yz(yz, bound=bound)
        feat_xz = self.encoder_xz(xz, bound=bound)
        
        return torch.cat([feat_xy, feat_yz, feat_xz], dim=-1)
    

    def encode_audio(self, a, phone_ids=None, phone_valid=None, phone_phase=None):
        # a: [1, 29, 16] or [8, 29, 16], audio features from deepspeech
        # if emb, a should be: [1, 16] or [8, 16]

        # fix audio traininig
        if a is None: return None

        enc_a = self.audio_net(a) # [1/8, audio_dim]
        if self.phoneme_fusion is not None:
            enc_a = self.phoneme_fusion(enc_a, phone_ids, phone_valid)
        if self.phase_temporal is not None:
            enc_a, attn_bias, center_gain = self.phase_temporal(enc_a, phone_phase, phone_valid)
            center_token = enc_a[enc_a.shape[0] // 2].unsqueeze(0)
            summary = self.audio_att_net(enc_a.unsqueeze(0), logit_bias=attn_bias)
            enc_a = summary + center_gain * (center_token - summary)
        else:
            enc_a = self.audio_att_net(enc_a.unsqueeze(0))
        return enc_a


    def forward(self, x, a, e=None, c=None, phone_ids=None, phone_valid=None, phone_phase=None):
        # x: [N, 3], in [-bound, bound]
        enc_x = self.encode_x(x, bound=self.bound)

        enc_a = self.encode_audio(a, phone_ids, phone_valid, phone_phase)
        enc_a = enc_a.repeat(enc_x.shape[0], 1)
        aud_ch_att = self.aud_ch_att_net(enc_x)
        enc_w = enc_a * aud_ch_att
        
        eye_att = torch.relu(self.eye_att_net(enc_x))
        enc_e = self.exp_encode_net(e[:-1])
        enc_e = torch.cat([enc_e, e[-1:]], dim=-1)
        enc_e = enc_e * eye_att
        if c is not None:
            c = c.repeat(enc_x.shape[0], 1)
            h = torch.cat([enc_x, enc_w, enc_e, c], dim=-1)
        else:
            h = torch.cat([enc_x, enc_w, enc_e], dim=-1)

        h = self.sigma_net(h)

        d_xyz = h[..., :3] * 1e-2
        d_rot = h[..., 3:7]
        d_opa = h[..., 7:8]
        d_scale = h[..., 8:11]
        return {
            'd_xyz': d_xyz,
            'd_rot': d_rot,
            'd_opa': d_opa,
            'd_scale': d_scale,
            'ambient_aud' : aud_ch_att.norm(dim=-1, keepdim=True),
            'ambient_eye' : eye_att.norm(dim=-1, keepdim=True),
        }


    # optimizer utils
    def get_params(self, lr, lr_net, wd=0):

        params = [
            {'params': self.audio_net.parameters(), 'lr': lr_net, 'weight_decay': wd}, 
            {'params': self.encoder_xy.parameters(), 'lr': lr},
            {'params': self.encoder_yz.parameters(), 'lr': lr},
            {'params': self.encoder_xz.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
        ]
        params.append({'params': self.audio_att_net.parameters(), 'lr': lr_net * 5, 'weight_decay': 0.0001})
        if self.phoneme_fusion is not None:
            params.append({'params': self.phoneme_fusion.parameters(), 'lr': lr_net, 'weight_decay': wd})
        if self.phase_temporal is not None:
            params.append({'params': self.phase_temporal.parameters(), 'lr': lr_net, 'weight_decay': wd})
        if self.individual_dim > 0:
            params.append({'params': self.individual_codes, 'lr': lr_net, 'weight_decay': wd})
        
        params.append({'params': self.aud_ch_att_net.parameters(), 'lr': lr_net, 'weight_decay': wd})
        params.append({'params': self.eye_att_net.parameters(), 'lr': lr_net, 'weight_decay': wd})
        params.append({'params': self.exp_encode_net.parameters(), 'lr': lr_net, 'weight_decay': wd})

        return params




class MouthMotionNetwork(nn.Module):
    def __init__(self,
                 audio_dim = 32,
                 ind_dim = 0,
                 args = None,
                 ):
        super(MouthMotionNetwork, self).__init__()

        if 'esperanto' in args.audio_extractor:
            self.audio_in_dim = 44
        elif 'deepspeech' in args.audio_extractor:
            self.audio_in_dim = 29
        elif 'hubert' in args.audio_extractor:
            self.audio_in_dim = 1024
        else:
            raise NotImplementedError
        
        
        self.bound = 0.15

        
        self.individual_dim = ind_dim
        if self.individual_dim > 0:
            self.individual_codes = nn.Parameter(torch.randn(10000, self.individual_dim) * 0.1) 

        # audio network
        self.audio_dim = audio_dim
        self.audio_net = AudioNet(self.audio_in_dim, self.audio_dim)

        self.audio_att_net = AudioAttNet(self.audio_dim)
        self.phoneme_fusion = build_phoneme_fusion(args, self.audio_dim)
        self.use_phoneme = self.phoneme_fusion is not None
        self.use_phone_phase = bool(getattr(args, "use_phone_phase", False))
        if self.use_phone_phase and not self.use_phoneme:
            raise ValueError("E2 --use_phone_phase requires --use_phoneme")
        self.phase_temporal = PhaseAwareTemporal(self.audio_dim) if self.use_phone_phase else None

        # DYNAMIC PART
        self.num_levels = 12
        self.level_dim = 1
        self.encoder_xy, self.in_dim_xy = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=64, log2_hashmap_size=17, desired_resolution=384 * self.bound)
        self.encoder_yz, self.in_dim_yz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=64, log2_hashmap_size=17, desired_resolution=384 * self.bound)
        self.encoder_xz, self.in_dim_xz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=64, log2_hashmap_size=17, desired_resolution=384 * self.bound)

        self.in_dim = self.in_dim_xy + self.in_dim_yz + self.in_dim_xz

        ## sigma network
        self.num_layers = 3
        self.hidden_dim = 32

        self.out_dim = 3
        self.sigma_net = MLP(self.in_dim + self.audio_dim + self.individual_dim, self.out_dim, self.hidden_dim, self.num_layers)
        
        self.aud_ch_att_net = MLP(self.in_dim, self.audio_dim, 32, 2)
    

    def encode_audio(self, a, phone_ids=None, phone_valid=None, phone_phase=None):
        # a: [1, 29, 16] or [8, 29, 16], audio features from deepspeech
        # if emb, a should be: [1, 16] or [8, 16]

        # fix audio traininig
        if a is None: return None

        enc_a = self.audio_net(a) # [1/8, audio_dim]
        if self.phoneme_fusion is not None:
            enc_a = self.phoneme_fusion(enc_a, phone_ids, phone_valid)
        if self.phase_temporal is not None:
            enc_a, attn_bias, center_gain = self.phase_temporal(enc_a, phone_phase, phone_valid)
            center_token = enc_a[enc_a.shape[0] // 2].unsqueeze(0)
            summary = self.audio_att_net(enc_a.unsqueeze(0), logit_bias=attn_bias)
            enc_a = summary + center_gain * (center_token - summary)
        else:
            enc_a = self.audio_att_net(enc_a.unsqueeze(0))
        return enc_a
    

    @staticmethod
    @torch.jit.script
    def split_xyz(x):
        xy, yz, xz = x[:, :-1], x[:, 1:], torch.cat([x[:,:1], x[:,-1:]], dim=-1)
        return xy, yz, xz


    def encode_x(self, xyz, bound):
        # x: [N, 3], in [-bound, bound]
        N, M = xyz.shape
        xy, yz, xz = self.split_xyz(xyz)
        feat_xy = self.encoder_xy(xy, bound=bound)
        feat_yz = self.encoder_yz(yz, bound=bound)
        feat_xz = self.encoder_xz(xz, bound=bound)
        
        return torch.cat([feat_xy, feat_yz, feat_xz], dim=-1)


    def forward(self, x, a, phone_ids=None, phone_valid=None, phone_phase=None):
        # x: [N, 3], in [-bound, bound]
        enc_x = self.encode_x(x, bound=self.bound)

        enc_a = self.encode_audio(a, phone_ids, phone_valid, phone_phase)
        enc_w = enc_a.repeat(enc_x.shape[0], 1)
        # aud_ch_att = self.aud_ch_att_net(enc_x)
        # enc_w = enc_a * aud_ch_att

        h = torch.cat([enc_x, enc_w], dim=-1)

        h = self.sigma_net(h)

        d_xyz = h * 1e-2
        d_xyz[..., 0] = d_xyz[..., 0] / 5
        d_xyz[..., 2] = d_xyz[..., 2] / 5
        return {
            'd_xyz': d_xyz,
            # 'ambient_aud' : aud_ch_att.norm(dim=-1, keepdim=True),
        }


    # optimizer utils
    def get_params(self, lr, lr_net, wd=0):

        params = [
            {'params': self.audio_net.parameters(), 'lr': lr_net, 'weight_decay': wd}, 
            {'params': self.encoder_xy.parameters(), 'lr': lr},
            {'params': self.encoder_yz.parameters(), 'lr': lr},
            {'params': self.encoder_xz.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
        ]
        params.append({'params': self.audio_att_net.parameters(), 'lr': lr_net * 5, 'weight_decay': 0.0001})
        if self.phoneme_fusion is not None:
            params.append({'params': self.phoneme_fusion.parameters(), 'lr': lr_net, 'weight_decay': wd})
        if self.phase_temporal is not None:
            params.append({'params': self.phase_temporal.parameters(), 'lr': lr_net, 'weight_decay': wd})
        if self.individual_dim > 0:
            params.append({'params': self.individual_codes, 'lr': lr_net, 'weight_decay': wd})
        
        params.append({'params': self.aud_ch_att_net.parameters(), 'lr': lr_net, 'weight_decay': wd})

        return params
