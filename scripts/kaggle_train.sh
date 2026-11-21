#!/usr/bin/env bash
# Kaggle GPU session. Mounts the prepare notebook's output (read-only) as data/,
# restores runs/ from the seq2hic-runs dataset if one is attached, trains both
# arms with --resume under a time budget, and evaluates once both are DONE.
# Everything worth keeping lands in /kaggle/working/runs (this notebook's output).
set -euo pipefail
PREP=$(ls -d /kaggle/input/*/data 2>/dev/null | head -1)
[ -n "$PREP" ] || { echo "prepare output not attached"; exit 1; }
ln -sfn "$PREP" data
mkdir -p /kaggle/working/runs
PREV=$(ls -d /kaggle/input/*/runs 2>/dev/null | head -1 || true)
[ -n "$PREV" ] && cp -rn "$PREV"/. /kaggle/working/runs/
ln -sfn /kaggle/working/runs runs
pip install -q pyfaidx cooler
export PYTHONPATH=src
nvidia-smi --query-gpu=name,memory.total --format=csv | tee runs/gpu.txt
BUDGET_H=${BUDGET_H:-11}
start=$(date +%s)
for arm in free_seq_only free_seq_atac; do
  [ -f runs/$arm/DONE ] && continue
  left=$(python -c "import time; print(round($BUDGET_H - (time.time() - $start) / 3600, 2))")
  if python -c "import sys; sys.exit(0 if $left < 1.0 else 1)"; then
    echo "session budget used up; run again next session"; exit 0
  fi
  python -m seq2hic.train --config configs/$arm.yaml --resume --max-hours "$left" 2>&1 | tee -a runs/$arm.train.log
  [ -f runs/$arm/DONE ] || { echo "$arm not finished; run again next session"; exit 0; }
done
for arm in free_seq_only free_seq_atac; do
  python -m seq2hic.evaluate --config configs/$arm.yaml --split test 2>&1 | tee runs/$arm.eval.log
done
python -m seq2hic.ablation --runs runs/free_seq_only runs/free_seq_atac 2>&1 | tee runs/ablation.log
