#!/bin/bash
#
# Compute SHAP feature importance by retraining walk-forward models from frozen
# screening artifacts. Default: maize NL family reps (RF, Transformer, TabPFN).
#
# Example (GPU node recommended for tabpfn + transformer_lf):
#   sbatch cybench/runs/slurm/shap_importance.sh
#
# Login-node pilot (RF only, CPU):
#   MODELS=random_forest ORIGINS=2020 FORCE_CPU=1 \
#     bash cybench/runs/slurm/shap_importance.sh
#
#SBATCH --job-name=cybench_shap
#SBATCH --output=output/shap_importance/out_%j.txt
#SBATCH --error=output/shap_importance/err_%j.txt
#SBATCH --mem-per-cpu=4G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
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
mkdir -p output/shap_importance

CROP="${CROP:-maize}"
COUNTRY="${COUNTRY:-NL}"
HORIZON="${PREDICTION_HORIZON:-eos}"
MODELS="${MODELS:-random_forest,transformer_lf,tabpfn}"
ORIGINS="${ORIGINS:-}"
LAST_ORIGIN_ONLY="${LAST_ORIGIN_ONLY:-no}"
FORCE_CPU="${FORCE_CPU:-0}"
CYBENCH_EXPERIMENT_NAME="${CYBENCH_EXPERIMENT_NAME:-baselines_NL_eos_v2}"
BASELINES_DIR="${CYBENCH_BASELINES_DIR:-${BASELINES_DIR}}"

LUSTRE_ROOT="${CYBENCH_OUTPUT_ROOT:-/lustre/backup/SHARED/AIN/agml/output}"
if [[ ! -d "${BASELINES_DIR}" && -d "${LUSTRE_ROOT}/${CYBENCH_EXPERIMENT_NAME}" ]]; then
  BASELINES_DIR="${LUSTRE_ROOT}/${CYBENCH_EXPERIMENT_NAME}"
fi

OUT_DIR="${SHAP_OUTPUT_DIR:-${LUSTRE_ROOT}/shap_importance/${CROP}_${COUNTRY}_${HORIZON}}"

extra=()
if [[ -n "${ORIGINS}" ]]; then
  extra+=(--origins "${ORIGINS}")
fi
if [[ "${LAST_ORIGIN_ONLY}" == "yes" ]]; then
  extra+=(--last-origin-only)
fi
if [[ "${FORCE_CPU}" == "1" ]]; then
  extra+=(--force-cpu)
fi

echo "SHAP importance | ${CROP}/${COUNTRY} | models=${MODELS} | horizon=${HORIZON}"
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
  -v
