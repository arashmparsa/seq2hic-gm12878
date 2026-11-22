# Kaggle CPU pre-launch check: clone the pinned commit and run scripts/kaggle_train.sh
# with PREFLIGHT=1 (every train/val window through the real loader + one step).
import subprocess, sys
REPO = "https://github.com/arashmparsa/seq2hic-gm12878.git"
COMMIT = "__COMMIT__"
sh = f"""set -e
rm -rf /tmp/seq2hic && git clone -q {REPO} /tmp/seq2hic && cd /tmp/seq2hic
git checkout -q {COMMIT} && git log -1 --format='commit %H'
PREFLIGHT=1 bash scripts/kaggle_train.sh"""
sys.exit(subprocess.call(["bash", "-c", sh]))
