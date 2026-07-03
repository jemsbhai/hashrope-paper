#!/bin/bash
#SBATCH --job-name=exp015_smoke
#SBATCH --output=exp015_smoke_%j.out
#SBATCH --error=exp015_smoke_%j.err
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=04:00:00

# EXP-015 / claim S3 -- SMOKE / red-green gate on the REAL serving model
# (Qwen/Qwen2.5-7B-Instruct-1M) BEFORE the confirmatory grid. Validates the new
# long-context path end to end:
#   - R1 smoke (single GPU): pipeline on the 1M model at ~2k prefixes.
#   - R2 smoke (tp=4): the BINDING constraint -- L=1,000,000 (near the model's
#     1,010,000 ceiling), chunked-prefill + enforce_eager + max_num_seqs=1,
#     prefix cache OFF and ON, NVML energy window.
# Both cells MUST print real Track A (energy_path, total_joules, cached_tokens;
# never "skipped") and pass the fail-fast preflight. Green here gates the grid.
# Smoke outputs are namespaced (*_smoke_raw_latest.json / _tmp_*_smoke) so they
# CANNOT contaminate the confirmatory grid via the resume cache.
#
# Just: sbatch scripts/exp015_smoke.sbatch   (then send the two smoke JSONs + logs)

set -euo pipefail

REPO="${EXP015_REPO:-${SLURM_SUBMIT_DIR:-$HOME/hashrope-paper}}"
VENV="${EXP015_VENV:-$HOME/exp015-venv}"
MODEL="${EXP015_MODEL:-Qwen/Qwen2.5-7B-Instruct-1M}"

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

echo "[sbatch] === R1 smoke (1 GPU, 1M model) ==="
python scripts/exp015_bench.py --regime R1 --smoke --gpus 0 --model "$MODEL"

echo "[sbatch] === R2 smoke (tp=4, L=1,000,000 -- binding constraint) ==="
python scripts/exp015_bench.py --regime R2 --smoke --gpus 0,1,2,3 --model "$MODEL"

echo "[sbatch] smoke done $(date -u +%FT%TZ)"
echo "[sbatch] send these two + the .out/.err logs:"
echo "  $REPO/experiments/exp_015_energy/results/exp015_R1_smoke_raw_latest.json"
echo "  $REPO/experiments/exp_015_energy/results/exp015_R2_smoke_raw_latest.json"
