#!/usr/bin/env bash
set -euo pipefail

python training/train.py \
  --config configs/mask2former/mask2former_swin-l_lars_512x1024.py \
  --work-dir work_dirs/mask2former_swin-l_lars_512x1024
