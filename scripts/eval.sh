#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: bash scripts/eval.sh <checkpoint_path>"
  exit 1
fi

python training/eval.py \
  --config configs/mask2former/mask2former_swin-l_lars_512x1024.py \
  --checkpoint "$1" \
  --work-dir work_dirs/mask2former_swin-l_lars_512x1024/eval
