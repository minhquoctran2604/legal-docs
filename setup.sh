#!/bin/bash
# Setup script: install deps and pull LFS model
# Usage: bash setup.sh [env_name]
set -e

ENV_NAME="${1:-legal_env}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Legal NER Setup ==="

# Pull LFS objects (model weights)
echo "[1/3] Pulling model weights via Git LFS..."
git lfs pull
echo "Done."

# Create virtualenv if not exists
if [ ! -d "$SCRIPT_DIR/$ENV_NAME" ]; then
    echo "[2/3] Creating Python environment '$ENV_NAME'..."
    python3 -m venv "$SCRIPT_DIR/$ENV_NAME"
else
    echo "[2/3] Environment '$ENV_NAME' already exists."
fi

# Install requirements
echo "[3/3] Installing dependencies..."
source "$SCRIPT_DIR/$ENV_NAME/bin/activate"
pip install -q --upgrade pip
pip install -q -r "$SCRIPT_DIR/legal_ner/requirements.txt"

echo ""
echo "Setup complete! Run inference:"
echo "  source $ENV_NAME/bin/activate"
echo "  cd legal_ner"
echo "  python -m training.infer --model data/models/legal-ner-combined/final --pdf /path/to/judgment.pdf"
