#!/bin/bash
#
# Parallel SHAP: one array task = one walk-forward origin (test year).
# After the array completes, run collect_shap_importance.sh (or the Python collector).
#
# Example (maize NL, TabICL, 5 origins on GPU):
#   MODELS=tabicl PERMUTATION_REPEATS=3 VERBOSE=1 \
#     sbatch --partition=gpu --gpus=1 --array=0-4 \
#     cybench/runs/slurm/shap_importance_array.sh
#
# Then collect:
#   CROP=maize COUNTRY=NL bash cybench/runs/slurm/collect_shap_importance.sh
#
#SBATCH --job-name=cybench_shap_o
#SBATCH --output=output/shap_importance/out_%A_%a.txt
#SBATCH --error=output/shap_importance/err_%A_%a.txt
#SBATCH --mem-per-cpu=4G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
##SBATCH --partition=gpu
##SBATCH --gpus=1
##SBATCH --array=0-4

set -euo pipefail

export CYBENCH_EXPERIMENT_NAME="${CYBENCH_EXPERIMENT_NAME:-baselines_NL_eos_v4}"

if [[ -f "${SLURM_SUBMIT_DIR:-}/cybench/runs/slurm/slurm_common.sh" ]]; then
  export SLURM_DIR="${SLURM_SUBMIT_DIR}/cybench/runs/slurm"
else
  export SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
source "${SLURM_DIR}/slurm_common.sh"
slurm_setup
mkdir -p output/shap_importance

CROP="${CROP:-maize}"
COUNTRY="${COUNTRY:-NL}"
HORIZON="${PREDICTION_HORIZON:-eos}"
MODELS="${MODELS:-random_forest}"
ORIGINS_LIST="${ORIGINS_LIST:-2016 2017 2018 2019 2020}"
FORCE_CPU="${FORCE_CPU:-0}"
VERBOSE="${VERBOSE:-0}"
MAX_BACKGROUND="${MAX_BACKGROUND:-}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"
SHAPIQ_BUDGET="${SHAPIQ_BUDGET:-}"
PERMUTATION_REPEATS="${PERMUTATION_REPEATS:-}"

read -r -a ORIGINS_ARR <<< "${ORIGINS_LIST}"
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "This script is intended for sbatch --array=... (set ORIGINS= manually for a local run)." >&2
  ORIGIN="${ORIGINS:-${ORIGINS_ARR[0]}}"
else
  ORIGIN="${ORIGINS_ARR[$SLURM_ARRAY_TASK_ID]}"
fi

if [[ -z "${ORIGIN:-}" ]]; then
  echo "No origin for array task ${SLURM_ARRAY_TASK_ID:-?} in ORIGINS_LIST=${ORIGINS_LIST}" >&2
  exit 1
fi

LUSTRE_ROOT="${CYBENCH_OUTPUT_ROOT:-/lustre/backup/SHARED/AIN/agml/output}"
if [[ -n "${CYBENCH_BASELINES_DIR:-}" ]]; then
  BASELINES_DIR="${CYBENCH_BASELINES_DIR}"
elif [[ -d "${LUSTRE_ROOT}/${CYBENCH_EXPERIMENT_NAME}" ]]; then
  BASELINES_DIR="${LUSTRE_ROOT}/${CYBENCH_EXPERIMENT_NAME}"
fi
export BASELINES_DIR

OUT_DIR="${SHAP_OUTPUT_DIR:-${LUSTRE_ROOT}/shap_importance/${CROP}_${COUNTRY}_${HORIZON}}"

extra=(--origins "${ORIGIN}" --skip-summary)
if [[ "${FORCE_CPU}" == "1" ]]; then
  extra+=(--force-cpu)
fi
if [[ -n "${MAX_BACKGROUND}" ]]; then
  extra+=(--max-background "${MAX_BACKGROUND}")
fi
if [[ -n "${MAX_EVAL_SAMPLES}" ]]; then
  extra+=(--max-eval-samples "${MAX_EVAL_SAMPLES}")
fi
if [[ -n "${SHAPIQ_BUDGET}" ]]; then
  extra+=(--shapiq-budget "${SHAPIQ_BUDGET}")
fi
if [[ -n "${PERMUTATION_REPEATS}" ]]; then
  extra+=(--permutation-repeats "${PERMUTATION_REPEATS}")
fi
verbose=()
if [[ "${VERBOSE}" == "1" ]]; then
  verbose=(-v)
fi

echo "SHAP importance (array) | ${CROP}/${COUNTRY} | models=${MODELS} | origin=${ORIGIN}"
echo "  baselines=${BASELINES_DIR}"
echo "  output=${OUT_DIR}"

poetry run python cybench/runs/analysis/compute_shap_importance.py \
  --crop "${CROP}" \
  --country "${COUNTRY}" \
  --models "${MODELS}" \
  --horizon "${HORIZON}" \
  --baselines-dir "${BASELINES_DIR}" \
  --output-dir "${OUT_DIR}" \
  "${extra[@]}" \
  "${verbose[@]}"
