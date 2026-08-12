#!/usr/bin/env bash
# Single source of truth for the EXP-076 SSL-v2 recipe environment.
#
# model_config() folds the dropout/augmentation constants into the checkpoint
# guard, so an infer/eval call that forgets them fails with "model configuration
# changed" even though the weights are fine.  Every script that touches a
# v2_ssl_* checkpoint sources this file instead of restating the block.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# cache v2 geometry (32 frames, 96x128)
export CUHKX_CACHE_SEGMENTS=32
export CUHKX_CACHE_HEIGHT=96
export CUHKX_CACHE_WIDTH=128
# EXP-062/065 regularized constants (replicated +2.5 micro / +3.0 object)
export CUHKX_VMIL_CROP_MIN=0.60
export CUHKX_VMIL_DROPOUT=0.35
export CUHKX_VMIL_MDROP_IR=0.25
export CUHKX_VMIL_MDROP_DEPTH=0.25
export CUHKX_VMIL_MDROP_THERMAL=0.35
export CUHKX_VMIL_LABEL_SMOOTHING=0.10
export CUHKX_GATE_OVERRIDE_LEADERBOARD=1
