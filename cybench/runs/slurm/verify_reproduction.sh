#!/bin/bash
#
# Read-only reproduction check: retrains walk-forward origins in memory and
# compares to saved test_preds.csv. Does NOT write into baselines or walk-forward
# run directories on lustre.
#
# Quick check (maize NL Transformer, origin 2020, from scratch + 2 in-process runs):
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

export CYBENCH_EXPERIMENT_NAME="${CYBENCH_EXPERIMENT_NAME:-baselines_NL_eos_v4}"

if [[ -f "${SLURM_SUBMIT_DIR:-}/cybench/runs/slurm/slurm_common.sh" ]]; then
  export SLURM_DIR="${SLURM_SUBMIT_DIR}/cybench/runs/slurm"
else
  export SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
source "${SLURM_DIR}/slurm_common.sh"
slurm_setup
export CYBENCH_TORCH_THREADS="${CYBENCH_TORCH_THREADS:-1}"

RESULTS_DIR="${REPO_ROOT}/output/verify_reproduction/results"
mkdir -p "${REPO_ROOT}/output/verify_reproduction" "${RESULTS_DIR}"

CROP="${CROP:-maize}"
COUNTRY="${COUNTRY:-NL}"
MODEL="${MODEL:-transformer_lf}"
HORIZON="${PREDICTION_HORIZON:-eos}"
ORIGINS="${ORIGINS:-2020}"
SEED="${SEED:-42}"
FROM_SCRATCH="${FROM_SCRATCH:-1}"
WITHIN_RUN_REPEATS="${WITHIN_RUN_REPEATS:-2}"
FORCE_CPU="${FORCE_CPU:-1}"

LUSTRE_ROOT="${CYBENCH_OUTPUT_ROOT:-/lustre/backup/SHARED/AIN/agml/output}"
if [[ -n "${CYBENCH_BASELINES_DIR:-}" ]]; then
  BASELINES_DIR="${CYBENCH_BASELINES_DIR}"
elif [[ -d "${LUSTRE_ROOT}/${CYBENCH_EXPERIMENT_NAME}" ]]; then
  BASELINES_DIR="${LUSTRE_ROOT}/${CYBENCH_EXPERIMENT_NAME}"
else
  echo "[FATAL] Baselines dir not found under ${LUSTRE_ROOT}/${CYBENCH_EXPERIMENT_NAME}" >&2
  exit 1
fi
export BASELINES_DIR

extra=()
if [[ "${FORCE_CPU}" == "1" ]]; then
  extra+=(--force-cpu)
fi
if [[ "${FROM_SCRATCH}" == "1" ]]; then
  extra+=(--from-scratch)
fi

stamp=$(date +%Y%m%d_%H%M%S)
job_tag="${SLURM_JOB_ID:-local}_${stamp}"
report_path="${RESULTS_DIR}/${CROP}_${COUNTRY}_${MODEL}_origin${ORIGINS}_seed${SEED}_${job_tag}.json"

echo "Verify reproduction | ${CROP}/${COUNTRY} | model=${MODEL} | horizon=${HORIZON}"
echo "  baselines=${BASELINES_DIR} (read-only)"
echo "  origins=${ORIGINS} | seed=${SEED} | from_scratch=${FROM_SCRATCH}"
echo "  within_run_repeats=${WITHIN_RUN_REPEATS} | device=$(device_mode_label)"
echo "  report=${report_path}"

poetry run python cybench/runs/analysis/verify_walk_forward_reproduction.py \
  --crop "${CROP}" \
  --country "${COUNTRY}" \
  --model "${MODEL}" \
  --horizon "${HORIZON}" \
  --seed "${SEED}" \
  --baselines-dir "${BASELINES_DIR}" \
  --origins "${ORIGINS}" \
  --within-run-repeats "${WITHIN_RUN_REPEATS}" \
  --report "${report_path}" \
  "${extra[@]}" \
  -v
