#!/usr/bin/env bash
# setup_facexformer.sh
# Run this ONCE in the a2 env on your GPU node to install FaceXFormer.
# Usage:
#   conda activate a2
#   bash setup_facexformer.sh /path/to/workdir

set -e

WORKDIR=${1:-"${WORK:?pass the workdir or set WORK}"}
FACEXFORMER_DIR="$WORKDIR/facexformer"

echo "=== FaceXFormer setup ==="
echo "WORKDIR: $WORKDIR"
echo "Repo will be at: $FACEXFORMER_DIR"

# 1. Clone repo (skip if already exists)
if [ ! -d "$FACEXFORMER_DIR" ]; then
    echo "[1] Cloning facexformer..."
    git clone https://github.com/Kartik-3004/facexformer.git "$FACEXFORMER_DIR"
else
    echo "[1] Repo already exists at $FACEXFORMER_DIR — skipping clone"
fi

# 2. Install requirements into existing a2 env
echo "[2] Installing facexformer requirements..."
cd "$FACEXFORMER_DIR"
# The repo's own requirements specify old torch — skip torch lines, 
# we already have a working torch in a2
grep -v "^torch" requirements.txt > /tmp/facex_reqs_filtered.txt || true
pip install -r /tmp/facex_reqs_filtered.txt

# facenet_pytorch is needed by inference.py (for MTCNN — we don't use it,
# but it's imported at module level)
pip install facenet-pytorch

# huggingface_hub for checkpoint download
pip install huggingface_hub

# 3. Download checkpoint
echo "[3] Downloading FaceXFormer checkpoint from HuggingFace..."
mkdir -p "$FACEXFORMER_DIR/ckpts"
python - <<'EOF'
import sys
sys.path.insert(0, ".")
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="kartiknarayan/facexformer",
    filename="ckpts/model.pt",
    local_dir="./"
)
print("Checkpoint saved to ./ckpts/model.pt")
EOF

echo ""
echo "=== Done ==="
echo ""
echo "FaceXFormer repo:  $FACEXFORMER_DIR"
echo "Checkpoint:        $FACEXFORMER_DIR/ckpts/model.pt"
echo ""
echo "Pass these to anonymize.py:"
echo "  --facexformer-repo $FACEXFORMER_DIR"
echo "  --facexformer-ckpt $FACEXFORMER_DIR/ckpts/model.pt"
