#!/bin/bash
#
# Collect parallel per-origin SHAP outputs into shap_summary.yaml / aggregate CSVs.
#
# Auto-discover all crop×country cases with origin artifacts (default):
#   bash cybench/runs/slurm/collect_shap_importance.sh
#
# Single case:
#   CROP=maize COUNTRY=NL bash cybench/runs/slurm/collect_shap_importance.sh
#
# Preview without writing:
#   bash cybench/runs/slurm/collect_shap_importance.sh --dry-run
#
set -euo pipefail

if [[ -f "${SLURM_SUBMIT_DIR:-}/cybench/runs/slurm/slurm_common.sh" ]]; then
  export SLURM_DIR="${SLURM_SUBMIT_DIR}/cybench/runs/slurm"
else
  export SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
source "${SLURM_DIR}/slurm_common.sh"
slurm_setup

LUSTRE_ROOT="${CYBENCH_OUTPUT_ROOT:-/lustre/backup/SHARED/AIN/agml/output}"
SHAP_ROOT="${SHAP_ROOT:-${LUSTRE_ROOT}/shap_importance}"
HORIZON="${PREDICTION_HORIZON:-eos}"
BASELINES_VERSION="${BASELINES_VERSION:-4}"
MODELS="${MODELS:-}"
VERBOSE="${VERBOSE:-0}"
DRY_RUN=false
CROP="${CROP:-}"
COUNTRY="${COUNTRY:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run|-n)
      DRY_RUN=true
      shift
      ;;
    --horizon)
      HORIZON=$2
      shift 2
      ;;
    --countries)
      shift
      COUNTRIES=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        COUNTRIES+=("$1")
        shift
      done
      ;;
    --crops)
      shift
      CROPS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        CROPS+=("$1")
        shift
      done
      ;;
    -h|--help)
      sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

extra=(--horizon "${HORIZON}" --output-root "${LUSTRE_ROOT}" --baselines-version "${BASELINES_VERSION}")
if [[ -n "${MODELS}" ]]; then
  extra+=(--models "${MODELS}")
fi
if [[ "${VERBOSE}" == "1" ]]; then
  extra+=(-v)
fi
if [[ "${DRY_RUN}" == true ]]; then
  extra+=(--dry-run)
fi

if [[ -n "${CROP}" && -n "${COUNTRY}" ]]; then
  export CYBENCH_EXPERIMENT_NAME="${CYBENCH_EXPERIMENT_NAME:-baselines_${COUNTRY}_eos_v${BASELINES_VERSION}}"
  if [[ -n "${CYBENCH_BASELINES_DIR:-}" ]]; then
    BASELINES_DIR="${CYBENCH_BASELINES_DIR}"
  elif [[ -d "${LUSTRE_ROOT}/${CYBENCH_EXPERIMENT_NAME}" ]]; then
    BASELINES_DIR="${LUSTRE_ROOT}/${CYBENCH_EXPERIMENT_NAME}"
  fi
  OUT_DIR="${SHAP_OUTPUT_DIR:-${SHAP_ROOT}/${CROP}_${COUNTRY}_${HORIZON}}"
  if [[ -n "${BASELINES_DIR:-}" ]]; then
    extra+=(--baselines-dir "${BASELINES_DIR}")
  fi
  echo "Collect SHAP | ${CROP}/${COUNTRY} | output=${OUT_DIR}"
  poetry run python cybench/runs/analysis/collect_shap_importance.py \
    --output-dir "${OUT_DIR}" \
    --crop "${CROP}" \
    --country "${COUNTRY}" \
    "${extra[@]}"
else
  discover_extra=()
  [[ -n "${COUNTRIES:-}" ]] && discover_extra+=(--countries "${COUNTRIES[@]}")
  [[ -n "${CROPS:-}" ]] && discover_extra+=(--crops "${CROPS[@]}")
  echo "Collect SHAP | auto-discover | shap_root=${SHAP_ROOT} | horizon=${HORIZON}"
  poetry run python cybench/runs/analysis/collect_shap_importance.py \
    --shap-root "${SHAP_ROOT}" \
    "${discover_extra[@]}" \
    "${extra[@]}"
fi
