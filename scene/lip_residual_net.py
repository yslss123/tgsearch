# -*- coding: utf-8 -*-
"""Compatibility re-export for callers already using the scene package.

For CPU-only tests and utilities, import from the top-level lip_residual_net
module so scene/__init__.py does not eagerly initialize TG's CUDA hash-grid.
"""
from lip_residual_net import LipResidualMotionNetwork, PhonemeContextEncoder

__all__ = ["LipResidualMotionNetwork", "PhonemeContextEncoder"]
