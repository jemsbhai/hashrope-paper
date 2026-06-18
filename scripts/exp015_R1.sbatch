#!/bin/bash
#SBATCH --job-name=exp015_R1
#SBATCH --output=exp015_R1_%j.out
#SBATCH --error=exp015_R1_%j.err
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00

# EXP-015 / claim S3 -- Regime R1 (realistic ShareGPT/LMSYS streams, ~2k-token
# prefixes). Single-GPU cells run 4-way in parallel across the 4 GPUs.
# Full relaunch per cell. Expected wall-clock ~3 h; 12 h cap leaves headroom.
#
# Edit REPO and VENV below for the cluster, then: sbatch scripts/exp015_R1.sbatch

set -euo pipefail

REPO="${EXP015_REPO:-$HOME/hashropeictaipaper}"
VENV="${EXP015_VENV:-$HOME/exp015-venv}"
MODEL="${EXP015_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
SEEDS="${EXP015_SEEDS:-42,43,44}"
INVOCATIONS="${EXP015_INVOCATIONS:-3}"

echo "[sbatch] node=$(hostname) job=${SLURM_JOB_ID:-na} $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv || true

source "$VENV/bin/activate"
cd "$REPO"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1

python scripts/exp015_bench.py \
    --regime R1 \
    --gpus 0,1,2,3 \
    --model "$MODEL" \
    --seeds "$SEEDS" \
    --invocations "$INVOCATIONS"

echo "[sbatch] R1 done $(date -u +%FT%TZ)"
echo "[sbatch] results: $REPO/experiments/exp_015_energy/results/exp015_R1_raw_latest.json"
