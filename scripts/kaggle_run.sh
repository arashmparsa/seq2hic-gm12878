#!/usr/bin/env bash
# Train both arms on a free Kaggle GPU (T4/P100, 16 GB, 12 h sessions, ~30 GPU-h/week).
# Usage in a Kaggle notebook cell (Settings: Accelerator = GPU, Internet = on):
#   !git clone <this repo> seq2hic && cd seq2hic && bash scripts/kaggle_run.sh
# Every session: rerun the same cell. Finished steps are skipped and training
# resumes from runs/<arm>/last.pt. Download /kaggle/working/seq2hic/runs at the
# end of each session (Kaggle wipes the disk unless you "Save Version").
set -euo pipefail
pip install -q -r requirements.txt pyBigWig
export PYTHONPATH=src
BUDGET_H=${BUDGET_H:-10.5}   # leave headroom under Kaggle's 12 h limit

if [ ! -f data/processed/.prepared ]; then
  python -m seq2hic.encode fetch ENCSR830NVY --output-type "mapping quality thresholded contact matrix"
  python -m seq2hic.encode fetch ENCSR637XSC --output-type "fold change over control" --file-format bigWig
  python -m seq2hic.encode fetch ENCSR637XSC --output-type "IDR thresholded peaks" --file-format bed
  python -m seq2hic.encode genome
  python -m seq2hic.prepare --config configs/free_seq_atac.yaml   # stride lives in the window index
  touch data/processed/.prepared
fi

start=$(date +%s)
for arm in free_seq_only free_seq_atac; do
  [ -f runs/$arm/DONE ] && continue
  left=$(python -c "import time; print(round($BUDGET_H - (time.time() - $start) / 3600, 2))")
  if python -c "import sys; sys.exit(0 if $left < 1.5 else 1)"; then
    echo "session budget used up; rerun this cell next session"; exit 0
  fi
  python -m seq2hic.train --config configs/$arm.yaml --resume --max-hours "$left"
  [ -f runs/$arm/DONE ] || { echo "$arm not finished; rerun next session"; exit 0; }
done
for arm in free_seq_only free_seq_atac; do
  python -m seq2hic.evaluate --config configs/$arm.yaml --split test
done
python -m seq2hic.ablation --runs runs/free_seq_only runs/free_seq_atac
