#!/usr/bin/env bash
# Continue a TalkingGaussian inference from DeepSpeech features already extracted by
# run_tg_new_video_audio.sh. Limit to available validation cameras; no feature re-extraction.
# Usage: bash run_tg_resume_val.sh data/new_audio.wav data/zh02 output/zh02
set -Eeuo pipefail
if [[ $# -ne 3 ]]; then
  echo 'Usage: bash run_tg_resume_val.sh INPUT_WAV DATASET_DIR MODEL_DIR' >&2
  exit 2
fi
INPUT_WAV="$1"
DATASET="$2"
MODEL="$3"
STEM="$(basename "${INPUT_WAV%.*}")"
WORK="$MODEL/custom_audio/$STEM"
FULL_FEATURES="$WORK/input_ds.npy"
SHORT_FEATURES="$WORK/input_ds_val.npy"
SHORT_WAV="$WORK/input_val.wav"
RENDER="$MODEL/train/ours_None/renders/out.mp4"
FINAL="$WORK/final_with_audio_val.mp4"
for f in "$INPUT_WAV" "$DATASET/transforms_val.json" "$MODEL/chkpnt_fuse_latest.pth" "$FULL_FEATURES"; do
  [[ -s "$f" ]] || { echo "Missing or empty file: $f" >&2; exit 2; }
done
[[ -f synthesize_fuse.py ]] || { echo 'Run from TalkingGaussian project root' >&2; exit 2; }
for c in python ffmpeg ffprobe; do command -v "$c" >/dev/null || { echo "Missing command: $c" >&2; exit 2; }; done

printf '\n[1/3] Clip existing DeepSpeech features to the validation camera count...\n'
DURATION="$(python - "$FULL_FEATURES" "$DATASET/transforms_val.json" "$SHORT_FEATURES" <<'PY'
import json, numpy as np, sys
src, cam_json, dst = sys.argv[1:]
with open(cam_json, encoding='utf-8') as fh:
    n_cams = len(json.load(fh)['frames'])
a = np.load(src)
if a.ndim != 3 or a.shape[1:] != (16, 29):
    raise SystemExit('Wrong DeepSpeech feature shape {}, expected [N,16,29]'.format(a.shape))
n = min(n_cams, a.shape[0])
if n < 1:
    raise SystemExit('No usable audio frames or validation cameras')
np.save(dst, a[:n].copy())
print('{:.6f}'.format(n / 25.0))
print('Reference camera frames: {}, source audio frames: {}, render frames: {}'.format(n_cams, a.shape[0], n), file=sys.stderr)
PY
)"

echo "Rendering duration: $DURATION seconds (FPS=25)"
ffmpeg -hide_banner -loglevel warning -nostdin -xerror -y \
  -i "$INPUT_WAV" -t "$DURATION" -map 0:a:0 -ac 1 -ar 16000 -c:a pcm_s16le "$SHORT_WAV"

printf '\n[2/3] Render using the validation cameras and the clipped features...\n'
python synthesize_fuse.py -s "$DATASET" -m "$MODEL" \
  --eval --use_train --audio "$SHORT_FEATURES" --audio_extractor deepspeech --fast
[[ -s "$RENDER" ]] || { echo "Missing render: $RENDER" >&2; exit 3; }

printf '\n[3/3] Mux the matching speech...\n'
ffmpeg -hide_banner -loglevel warning -nostdin -y \
  -i "$RENDER" -i "$SHORT_WAV" -map 0:v:0 -map 1:a:0 \
  -c:v copy -c:a aac -b:a 192k -shortest "$FINAL"
[[ -s "$FINAL" ]] || { echo "Failed to create: $FINAL" >&2; exit 3; }
echo "DONE: $FINAL"
ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1 "$FINAL"
