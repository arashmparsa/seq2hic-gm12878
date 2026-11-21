#!/usr/bin/env bash
# Kaggle CPU session (no GPU quota used): build the shared dataset once.
# Outputs to /kaggle/working/data (kept as this notebook's output, which the
# training notebook mounts read-only):
#   data/raw/{GRCh38 fasta + .fai, gencode.tss.bed, atac.idr_peaks.bed}
#   data/processed/{index.json, targets.npy, masks.npy, expected.npy, atac/*.npy}
# Pinned inputs (ENCODE, GRCh38):
#   Hi-C  ENCFF916GWS  read remotely (see configs/free_*.yaml)
#   ATAC  ENCFF603BJO  fold change over control, pooled reps 1-3 (4.2 GB)
#   peaks ENCFF871TST  IDR thresholded peaks, pooled
set -euo pipefail
OUT=/kaggle/working/data
mkdir -p "$OUT/raw" "$OUT/processed" data/raw
ln -sfn "$OUT/processed" data/processed
pip install -q hic-straw pyBigWig pyfaidx cooler
export PYTHONPATH=src

get() { # url dest md5
  [ -f "$2" ] && echo "$3  $2" | md5sum -c --quiet && return 0
  curl -sSfL --retry 5 -o "$2" "$1"; echo "$3  $2" | md5sum -c --quiet
}
S3=https://encode-public.s3.amazonaws.com/2021/03/16
get $S3/2286d032-1237-4f3e-a960-9540079fc80e/ENCFF603BJO.bigWig data/raw/atac.fold_change.bigWig 7729f7175175ba9a0fd08222b1800b7b
get $S3/f5294dc2-90d7-42f4-a5c2-83dc658078fb/ENCFF871TST.bed.gz data/raw/atac.idr_peaks.bed.gz 2982f59e3b7b5e0fdf30e1c623c726d4
gunzip -kf data/raw/atac.idr_peaks.bed.gz
python -m seq2hic.encode genome --out data/raw

python -m seq2hic.prepare --config configs/free_seq_atac.yaml

python - <<'PY'   # builds the .fai so the read-only copy never needs to write one
from seq2hic.genome import Genome; Genome("data/raw/GRCh38.primary_assembly.genome.fa")
PY
cp data/raw/GRCh38.primary_assembly.genome.fa* data/raw/gencode.tss.bed data/raw/atac.idr_peaks.bed "$OUT/raw/"
rm -f "$OUT/raw/"*.gz
du -sh "$OUT"/*
