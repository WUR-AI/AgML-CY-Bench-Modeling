#!/bin/bash
#
# Refit one walk-forward origin and compare predictions to saved test_preds.csv.
# Use this on a GPU node for torch models (transformer_lf, etc.).
#
# Quick check (maize NL Transformer, origin 2020, eos v3):
#   sbatch --partition=gpu --gpus=1 cybench/runs/slurm/verify_reproduction.sh
#
# Interactive login node (RF only):
#   MODEL=random_forest FORCE_CPU=1 bash cybench/runs/slurm/verify_reproduction.sh
#
#SBATCH --job-name=cybench_repro
#SBATCH --output=output/verify_reproduction/out_%j.txt
#SBATCH --error=output/verify_reproduction/err_%j.txt
#SBATCH --mem-per-cpu=4G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
##SBATCH --partition=gpu
##SBATCH --gpus=1

set -euo pipefail

if [[ -f "${SLURM_SUBMIT_DIR:-}/cybench/runs/slurm/slurm_common.sh" ]]; then
  export SLURM_DIR="${SLURM_SUBMIT_DIR}/cybench/runs/slurm"
else
  export SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
source "${SLURM_DIR}/slurm_common.sh"
slurm_setup
mkdir -p output/verify_reproduction

CROP="${CROP:-maize}"
COUNTRY="${COUNTRY:-NL}"
MODEL="${MODEL:-transformer_lf}"
HORIZON="${PREDICTION_HORIZON:-eos}"
ORIGINS="${ORIGINS:-2020}"
SEED="${SEED:-42}"
FORCE_CPU="${FORCE_CPU:-0}"
CYBENCH_EXPERIMENT_NAME="${CYBENCH_EXPERIMENT_NAME:-baselines_NL_eos_v4}"

LUSTRE_ROOT="${CYBENCH_OUTPUT_ROOT:-/lustre/backup/SHARED/AIN/agml/output}"
BASELINES_DIR="${CYBENCH_BASELINES_DIR:-${BASELINES_DIR}}"
if [[ ! -d "${BASELINES_DIR}" && -d "${LUSTRE_ROOT}/${CYBENCH_EXPERIMENT_NAME}" ]]; then
  BASELINES_DIR="${LUSTRE_ROOT}/${CYBENCH_EXPERIMENT_NAME}"
fi

extra=()
if [[ "${FORCE_CPU}" == "1" ]]; then
  extra+=(--force-cpu)
fi

echo "Verify reproduction | ${CROP}/${COUNTRY} | model=${MODEL} | horizon=${HORIZON}"
echo "  baselines=${BASELINES_DIR}"
echo "  origins=${ORIGINS} | seed=${SEED} | device=$(device_mode_label)"

poetry run python cybench/runs/analysis/verify_walk_forward_reproduction.py \
  --crop "${CROP}" \
  --country "${COUNTRY}" \
  --model "${MODEL}" \
  --horizon "${HORIZON}" \
  --seed "${SEED}" \
  --baselines-dir "${BASELINES_DIR}" \
  --origins "${ORIGINS}" \
  "${extra[@]}" \
  -v
