#!/bin/bash
# EXP-015 / claim S3 -- one-time environment setup on the cluster node (vLLM).
#
# Engine: vLLM offline LLM (in-process; no HTTP server). Switched from SGLang on
# 2026-06-21 after sgl-kernel wheels proved ABI-incompatible with the node's
# torch 2.6.0+cu124. vLLM ships self-contained CUDA kernels (no separate kernel
# wheel), so this avoids the sgl-kernel ABI wall.
#
# Strategy: install the KNOWN-GOOD torch (2.6.0+cu124) FIRST, then vLLM 0.8.5.post1
# (which requires torch==2.6.0 -- already satisfied, so pip will NOT replace it
# with a CUDA-13 build). transformers is PINNED to 4.51.3 (the version that fixes
# the Qwen2 tokenizer all_special_tokens_extended error). Do not run a bare
# `pip install vllm` (it would pull a newer vLLM that drags torch to a cu130 build
# the node cannot use).
#
# NOTE: only needed for a FRESH venv. An existing working venv (torch 2.6.0+cu124,
# vllm 0.8.5.post1, transformers 4.51.3) does NOT need a rebuild.
#
# Usage:
#   bash scripts/exp015_setup.sh
#   # optional pre-fetch to avoid first-run stalls:
#   #   huggingface-cli download Qwen/Qwen2.5-7B-Instruct-1M

set -euo pipefail

VENV="${EXP015_VENV:-$HOME/exp015-venv}"

echo "[setup] creating venv at $VENV"
python3 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel

# 1) known-good torch FIRST (CUDA-12 build that works on the node)
echo "[setup] installing torch 2.6.0 + torchvision (cu124)"
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

echo "[setup] torch check (must print: 2.6.0+cu124 True <n_gpus>)"
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"

# 2) vLLM pinned to the release built for torch 2.6 (requires torch==2.6.0, which
#    is already installed, so this does NOT change torch).
echo "[setup] installing vllm==0.8.5.post1 (torch 2.6 line)"
pip install vllm==0.8.5.post1

# 3) measurement + data deps; transformers pinned to the tokenizer-compatible
#    version. Installed AFTER vllm so the pin wins (none of these move torch).
echo "[setup] installing measurement + data deps (transformers pinned 4.51.3)"
pip install nvidia-ml-py "hashrope==0.2.2" "transformers==4.51.3" datasets numpy

echo "[setup] re-checking torch was NOT moved (must still be 2.6.0+cu124 True):"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

echo "[setup] versions:"
python - <<'PYEOF'
import importlib
for m in ["torch", "vllm", "pynvml", "hashrope", "transformers", "datasets", "numpy"]:
    try:
        mod = importlib.import_module(m)
        print(f"  {m}: {getattr(mod, '__version__', 'unknown')}")
    except Exception as e:
        print(f"  {m}: IMPORT FAILED ({e})")
try:
    import torch, vllm  # noqa
    print(f"  cuda_available: {torch.cuda.is_available()} | n_gpus: {torch.cuda.device_count()}")
except Exception as e:
    print(f"  combined import/cuda check failed: {e}")
PYEOF

echo ""
echo "[setup] done. If torch shows 2.6.0+cu124, vllm imports, and transformers is"
echo "[setup] 4.51.3, run the smoke gate next (real 1M model, one command):"
echo "  sbatch scripts/exp015_smoke.sbatch    # R1-tiny + R2 @ L=1e6 binding constraint"
echo "[setup] the smoke MUST show real Track A (not 'skipped') and pass the preflight."
