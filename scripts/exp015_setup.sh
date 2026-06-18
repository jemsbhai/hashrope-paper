#!/bin/bash
# EXP-015 / claim S3 -- one-time environment setup on the cluster node.
# Creates a venv and installs the serving + measurement stack. The node has
# internet, so models/tokenizers download on first run.
#
# Usage:
#   bash scripts/exp015_setup.sh
#   # then (optional, pre-fetch the smoke + serving models to avoid first-run stalls):
#   #   huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct
#   #   huggingface-cli download Qwen/Qwen2.5-7B-Instruct

set -euo pipefail

VENV="${EXP015_VENV:-$HOME/exp015-venv}"

echo "[setup] creating venv at $VENV"
python3 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel

# Serving engine (pulls a compatible torch). If your CUDA needs a specific torch
# build, install torch first per https://pytorch.org, then run this.
echo "[setup] installing sglang[all]"
pip install "sglang[all]"

echo "[setup] installing measurement + data deps"
pip install \
    nvidia-ml-py \
    "hashrope==0.2.2" \
    transformers \
    datasets \
    numpy

echo "[setup] versions:"
python - <<'PY'
import importlib
for m in ["sglang", "torch", "pynvml", "hashrope", "transformers", "numpy"]:
    try:
        mod = importlib.import_module(m)
        print(f"  {m}: {getattr(mod, '__version__', 'unknown')}")
    except Exception as e:
        print(f"  {m}: IMPORT FAILED ({e})")
try:
    import torch
    print(f"  cuda_available: {torch.cuda.is_available()} | n_gpus: {torch.cuda.device_count()}")
except Exception as e:
    print(f"  cuda check failed: {e}")
PY

echo "[setup] done. Smoke-test next:"
echo "  source $VENV/bin/activate && cd <repo>"
echo "  python scripts/exp015_bench.py --regime R1 --smoke --gpus 0 --model Qwen/Qwen2.5-0.5B-Instruct"
echo "  python scripts/exp015_bench.py --regime R2 --smoke --gpus 0,1,2,3 --model Qwen/Qwen2.5-0.5B-Instruct"
