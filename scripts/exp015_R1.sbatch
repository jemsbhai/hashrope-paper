#!/bin/bash
#SBATCH --job-name=exp015_R1
#SBATCH --output=exp015_R1_%j.out
#SBATCH --error=exp015_R1_%j.err
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00

# EXP-015 / claim S3 -- Regime R1 (realistic ShareGPT/LMSYS streams, ~2k-token
# prefixes). Single-GPU cells run 4-way in parallel across the 4 GPUs. Full
# relaunch per cell. Expected wall-clock ~3 h; 12 h cap leaves headroom.
# Model: Qwen/Qwen2.5-7B-Instruct-1M. The bench caps R1 max_model_len at 32768
# (mirrors the prior Instruct run; avoids reserving 1M-context KV at ~2k prefixes).
# Pull-and-run: CUDA driver-link fix auto-applied below; no manual env needed.
#
# Just: sbatch scripts/exp015_R1.sbatch

set -euo pipefail

REPO="${EXP015_REPO:-${SLURM_SUBMIT_DIR:-$HOME/hashrope-paper}}"
VENV="${EXP015_VENV:-$HOME/exp015-venv}"
MODEL="${EXP015_MODEL:-Qwen/Qwen2.5-7B-Instruct-1M}"
SEEDS="${EXP015_SEEDS:-42,43,44}"
INVOCATIONS="${EXP015_INVOCATIONS:-3}"

echo "[sbatch] node=$(hostname) job=${SLURM_JOB_ID:-na} $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv || true

source "$VENV/bin/activate"
cd "$REPO"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1

# --- CUDA driver link fix (Triton JIT links -lcuda; the node ships libcuda.so.1
# only, no bare libcuda.so). Auto-detect the REAL driver library (never a CUDA
# toolkit stub) and expose a bare-name symlink on the link + runtime paths.
# Idempotent; handles a dangling prior symlink. Pull-and-run: no manual step. ---
CUDALINK="$HOME/cudalink"
mkdir -p "$CUDALINK"
if [ ! -e "$CUDALINK/libcuda.so" ]; then
    LIBCUDA="$(find /usr/lib /usr/lib64 /lib /lib64 /usr/local /opt -name 'libcuda.so.1' 2>/dev/null | head -n1 || true)"
    if [ -z "$LIBCUDA" ]; then
        LIBCUDA="$(ldconfig -p 2>/dev/null | awk '/libcuda\.so\.1/{print $NF; exit}' || true)"
    fi
    if [ -n "$LIBCUDA" ] && [ -e "$LIBCUDA" ]; then
        ln -sf "$LIBCUDA" "$CUDALINK/libcuda.so"
        echo "[sbatch] linked $CUDALINK/libcuda.so -> $LIBCUDA"
    else
        echo "[sbatch] WARNING: libcuda.so.1 not found; Triton -lcuda link may fail"
    fi
fi
export LIBRARY_PATH="$CUDALINK${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CUDALINK${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

python scripts/exp015_bench.py \
    --regime R1 \
    --gpus 0,1,2,3 \
    --model "$MODEL" \
    --seeds "$SEEDS" \
    --invocations "$INVOCATIONS"

echo "[sbatch] R1 done $(date -u +%FT%TZ)"
echo "[sbatch] results: $REPO/experiments/exp_015_energy/results/exp015_R1_raw_latest.json"
