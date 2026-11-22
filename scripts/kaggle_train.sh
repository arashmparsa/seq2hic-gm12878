#!/usr/bin/env bash
# Kaggle GPU session. Mounts the prepare notebook's output (read-only) as data/,
# restores runs/ from the seq2hic-runs dataset if one is attached, trains both
# arms with --resume under a time budget, and evaluates once both are DONE.
# Everything worth keeping lands in /kaggle/working/runs (this notebook's output).
set -euo pipefail
echo "attached inputs:"; find /kaggle/input -maxdepth 5 -type d 2>/dev/null | head -40 || true
# kernel_sources mount depth differs between Kaggle images; find the data/ dir that holds processed/
PREP=$(find /kaggle/input -maxdepth 6 -type d -path "*/data/processed" 2>/dev/null | head -1 || true)
PREP=${PREP%/processed}
[ -n "$PREP" ] || { echo "prepare output not attached"; exit 1; }
echo "prepare output: $PREP"
# processed/ stays on the read-only mount; raw/ (fasta + .fai) is copied to local disk
# because pyfaidx rebuilds an index it thinks is stale and the mount is not writable.
mkdir -p data/raw
ln -sfn "$PREP/processed" data/processed
cp -r "$PREP/raw/." data/raw/
touch data/raw/*.fai
mkdir -p /kaggle/working/runs
PREV=$(find /kaggle/input -maxdepth 6 -type d -name runs 2>/dev/null | head -1 || true)
[ -n "$PREV" ] && cp -rn "$PREV"/. /kaggle/working/runs/
ln -sfn /kaggle/working/runs runs
pip install -q pyfaidx cooler
export PYTHONPATH=src
nvidia-smi --query-gpu=name,memory.total --format=csv | tee runs/gpu.txt
BUDGET_H=${BUDGET_H:-11}
NGPU=$(nvidia-smi -L | wc -l)
train_arm() {  # arm gpu
  [ -f runs/$1/DONE ] && return 0
  CUDA_VISIBLE_DEVICES=$2 python -m seq2hic.train --config configs/$1.yaml --resume     --max-hours "$BUDGET_H" > runs/$1.train.log 2>&1 || true
}
if [ "$NGPU" -ge 2 ]; then   # T4 x2: one arm per GPU, same schedule, run side by side
  train_arm free_seq_only 0 & train_arm free_seq_atac 1 & wait
else
  start=$(date +%s)
  for arm in free_seq_only free_seq_atac; do
    left=$(python -c "import time; print(round($BUDGET_H - (time.time() - $start) / 3600, 2))")
    python -c "import sys; sys.exit(0 if $left < 1.0 else 1)" && break
    BUDGET_H=$left train_arm $arm 0
  done
fi
tail -n 3 runs/*.train.log
for arm in free_seq_only free_seq_atac; do
  [ -f runs/$arm/DONE ] || { echo "$arm not finished; run again next session"; exit 0; }
done
for arm in free_seq_only free_seq_atac; do
  python -m seq2hic.evaluate --config configs/$arm.yaml --split test 2>&1 | tee runs/$arm.eval.log
done
python -m seq2hic.ablation --runs runs/free_seq_only runs/free_seq_atac 2>&1 | tee runs/ablation.log
