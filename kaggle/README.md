# Running on free Kaggle hardware

Two private Kaggle notebooks, pushed with the Kaggle CLI (`kaggle kernels push -p kaggle/<name>`):

1. **prepare** (CPU only, no GPU quota): downloads the genome and pinned ATAC files,
   reads the 95 GB Hi-C file remotely with HTTP range requests (hicstraw), and saves
   `data/` (~10 GB) as the notebook's output.
2. **train** (GPU): mounts that output read-only, trains both arms with `--resume`
   under an 11 h budget, and evaluates once both are done. Results are in its
   `runs/` output. To continue in a new session, download `runs/`, upload it as a
   private dataset, and add it to `dataset_sources`.

`run.py` pins the repo commit it runs, so each session's code is recorded.
