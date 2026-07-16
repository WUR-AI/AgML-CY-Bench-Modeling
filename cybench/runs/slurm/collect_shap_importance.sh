#!/bin/bash
#
# Collect parallel per-origin SHAP outputs into shap_summary.yaml / aggregate CSVs.
#
# Example:
#   CROP=maize COUNTRY=NL \
#     bash cybench/runs/slurm/collect_shap_importance.sh
#
set -euo pipefail

export CYBENCH_EXPERIMENT_NAME="${CYBENCH_EXPERIMENT_NAME:-baselines_NL_eos_v4}"

if [[ -f "${SLURM_SUBMIT_DIR:-}/cybench/runs/slurm/slurm_common.sh" ]]; then
  export SLURM_DIR="${SLURM_SUBMIT_DIR}/cybench/runs/slurm"
else
  export SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
source "${SLURM_DIR}/slurm_common.sh"
slurm_setup

CROP="${CROP:-maize}"
COUNTRY="${COUNTRY:-NL}"
HORIZON="${PREDICTION_HORIZON:-eos}"
MODELS="${MODELS:-}"
VERBOSE="${VERBOSE:-0}"

LUSTRE_ROOT="${CYBENCH_OUTPUT_ROOT:-/lustre/backup/SHARED/AIN/agml/output}"
if [[ -n "${CYBENCH_BASELINES_DIR:-}" ]]; then
  BASELINES_DIR="${CYBENCH_BASELINES_DIR}"
elif [[ -d "${LUSTRE_ROOT}/${CYBENCH_EXPERIMENT_NAME}" ]]; then
  BASELINES_DIR="${LUSTRE_ROOT}/${CYBENCH_EXPERIMENT_NAME}"
fi

OUT_DIR="${SHAP_OUTPUT_DIR:-${LUSTRE_ROOT}/shap_importance/${CROP}_${COUNTRY}_${HORIZON}}"

extra=()
if [[ -n "${MODELS}" ]]; then
  extra+=(--models "${MODELS}")
fi
if [[ -n "${BASELINES_DIR:-}" ]]; then
  extra+=(--baselines-dir "${BASELINES_DIR}")
fi
verbose=()
if [[ "${VERBOSE}" == "1" ]]; then
  verbose=(-v)
fi

echo "Collect SHAP | ${CROP}/${COUNTRY} | output=${OUT_DIR}"

poetry run python cybench/runs/analysis/collect_shap_importance.py \
  --output-dir "${OUT_DIR}" \
  --crop "${CROP}" \
  --country "${COUNTRY}" \
  "${extra[@]}" \
  "${verbose[@]}"
