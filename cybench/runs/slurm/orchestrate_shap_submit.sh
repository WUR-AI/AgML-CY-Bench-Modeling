#!/usr/bin/env bash
#
# Submit SHAP array jobs (one task = one walk-forward origin) for many crop×country pairs.
#
# Usage (from repo root on anunna):
#   cybench/runs/slurm/orchestrate_shap_submit.sh --list
#   cybench/runs/slurm/orchestrate_shap_submit.sh --dry-run
#   cybench/runs/slurm/orchestrate_shap_submit.sh --models random_forest transformer_lf
#   cybench/runs/slurm/orchestrate_shap_submit.sh --countries NL DE US --max 5
#
# After arrays finish, collect per crop×country:
#   CROP=maize COUNTRY=NL bash cybench/runs/slurm/collect_shap_importance.sh
#
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: orchestrate_shap_submit.sh [options]

Plan and sbatch shap_importance_array.sh for benchmark crop×country pairs.
Defaults: random_forest + transformer_lf, eos horizon, baselines_*_eos_v4.

Options:
  --list              Show plan and exit
  --dry-run           Print sbatch commands without submitting
  --models M ...      Model slugs (default: random_forest transformer_lf)
  --crops C ...       Limit crops (default: maize wheat)
  --countries CC ...  Limit countries
  --horizon H         eos | mid | ... (default: eos)
  --version N         Baselines batch version (default: 4)
  --output-root DIR   Lustre output root (default: lustre .../output)
  --data-dir DIR      cybench/data override
  --all               Include already-complete jobs in listing
  --force             Resubmit all origins (ignore existing YAML)
  --max N             Submit at most N array jobs this run
  --collect           After submit, print collect commands for submitted pairs

Environment passed to each array job:
  CROP, COUNTRY, MODELS, ORIGINS_LIST, CYBENCH_BASELINES_DIR, SHAP_OUTPUT_DIR,
  PREDICTION_HORIZON, FORCE_CPU=1 (default on cpu partition)
EOF
}

if [[ -f "${SLURM_SUBMIT_DIR:-}/cybench/runs/slurm/slurm_common.sh" ]]; then
  SLURM_DIR="${SLURM_SUBMIT_DIR}/cybench/runs/slurm"
elif [[ -n "${REPO_ROOT:-}" && -f "${REPO_ROOT}/cybench/runs/slurm/slurm_common.sh" ]]; then
  SLURM_DIR="${REPO_ROOT}/cybench/runs/slurm"
else
  SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

SHAP_ARRAY="${SLURM_DIR}/shap_importance_array.sh"
PLAN_PY="${SLURM_DIR}/shap_submit_lib.py"

shap_model_tag() {
  case "$1" in
    random_forest) echo "rf" ;;
    transformer_lf) echo "trf" ;;
    tabpfn) echo "tabpfn" ;;
    tabicl) echo "tabicl" ;;
    tabdpt) echo "tabdpt" ;;
    *) echo "${1//_/-}" ;;
  esac
}

shap_job_name() {
  local crop=$1 country=$2 model=$3
  local tag
  tag="$(shap_model_tag "${model}")"
  # Slurm job names are limited; crop×country×model slug is enough to grep sacct/squeue.
  echo "shap_${crop}_${country}_${tag}"
}

LIST_ONLY=false
DRY_RUN=false
FORCE=false
ALL=false
COLLECT=false
VERSION=4
HORIZON="eos"
MAX_JOBS=0
OUTPUT_ROOT=""
DATA_DIR=""
REQUESTED_MODELS=()
CROPS=()
COUNTRIES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --list)
      LIST_ONLY=true
      shift
      ;;
    --dry-run|-n)
      DRY_RUN=true
      shift
      ;;
    --models)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        REQUESTED_MODELS+=("$1")
        shift
      done
      ;;
    --crops)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        CROPS+=("$1")
        shift
      done
      ;;
    --countries)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        COUNTRIES+=("$1")
        shift
      done
      ;;
    --horizon)
      HORIZON=$2
      shift 2
      ;;
    --version)
      VERSION=$2
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT=$2
      shift 2
      ;;
    --data-dir)
      DATA_DIR=$2
      shift 2
      ;;
    --all)
      ALL=true
      shift
      ;;
    --force)
      FORCE=true
      shift
      ;;
    --max)
      MAX_JOBS=$2
      shift 2
      ;;
    --collect)
      COLLECT=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

plan_args=(poetry run python "${PLAN_PY}" --horizon "${HORIZON}" --version "${VERSION}")
[[ -n "${OUTPUT_ROOT}" ]] && plan_args+=(--output-root "${OUTPUT_ROOT}")
[[ -n "${DATA_DIR}" ]] && plan_args+=(--data-dir "${DATA_DIR}")
[[ ${#REQUESTED_MODELS[@]} -gt 0 ]] && plan_args+=(--models "${REQUESTED_MODELS[@]}")
[[ ${#CROPS[@]} -gt 0 ]] && plan_args+=(--crops "${CROPS[@]}")
[[ ${#COUNTRIES[@]} -gt 0 ]] && plan_args+=(--countries "${COUNTRIES[@]}")
[[ "${ALL}" == true ]] && plan_args+=(--all)
[[ "${FORCE}" == true ]] && plan_args+=(--force)

if [[ "${LIST_ONLY}" == true ]]; then
  "${plan_args[@]}"
  exit 0
fi

mapfile -t PLAN_LINES < <("${plan_args[@]}" --plan-tsv)

if [[ ${#PLAN_LINES[@]} -eq 0 ]]; then
  echo "[DONE] No SHAP array jobs to submit (use --list to inspect)"
  exit 0
fi

echo "[INFO] ${#PLAN_LINES[@]} SHAP array job(s) to submit"
submitted=0
declare -a COLLECT_PAIRS=()

for line in "${PLAN_LINES[@]}"; do
  IFS=$'\t' read -r CROP CC MODEL HZN BATCH ORIGINS_LIST ARRAY_SPEC N_REGIONS N_PENDING BASELINES_DIR OUT_DIR SLURM_MEM <<< "${line}"
  if [[ "${MAX_JOBS}" -gt 0 && "${submitted}" -ge "${MAX_JOBS}" ]]; then
    echo "[STOP] --max ${MAX_JOBS} reached"
    break
  fi

  job_name="$(shap_job_name "${CROP}" "${CC}" "${MODEL}")"
  mem="${SLURM_MEM:-32G}"
  sbatch_args=(
    --job-name="${job_name}"
    --array="${ARRAY_SPEC}"
    --mem="${mem}"
  )

  export CROP="${CROP}"
  export COUNTRY="${CC}"
  # MODELS must be a scalar for sbatch children (not a bash array); see REQUESTED_MODELS above.
  unset MODELS
  export MODELS="${MODEL}"
  export ORIGINS_LIST="${ORIGINS_LIST}"
  export CYBENCH_BASELINES_DIR="${BASELINES_DIR}"
  export SHAP_OUTPUT_DIR="${OUT_DIR}"
  export CYBENCH_EXPERIMENT_NAME="${BATCH}"
  export PREDICTION_HORIZON="${HZN}"
  export FORCE_CPU="${FORCE_CPU:-1}"

  echo ""
  echo "=== ${CROP}/${CC} | ${MODEL} | origins=${ORIGINS_LIST} | array=${ARRAY_SPEC} | regions=${N_REGIONS} ==="
  echo "    job=${job_name} mem=${mem} MODELS=${MODELS}"
  if [[ "${DRY_RUN}" == true ]]; then
    echo "[DRY-RUN] sbatch ${sbatch_args[*]} ${SHAP_ARRAY}"
  else
    sbatch "${sbatch_args[@]}" "${SHAP_ARRAY}"
  fi
  submitted=$((submitted + 1))
  COLLECT_PAIRS+=("${CROP}:${CC}:${OUT_DIR}:${BASELINES_DIR}")
done

echo ""
if [[ "${DRY_RUN}" == true ]]; then
  echo "[DONE] Would submit ${submitted} SHAP array job(s)"
else
  echo "[DONE] Submitted ${submitted} SHAP array job(s)"
fi

if [[ "${COLLECT}" == true && ${#COLLECT_PAIRS[@]} -gt 0 ]]; then
  echo ""
  echo "[COLLECT] Run after arrays complete (per unique crop×country):"
  declare -A SEEN=()
  for entry in "${COLLECT_PAIRS[@]}"; do
    IFS=: read -r CROP CC OUT_DIR BASELINES_DIR <<< "${entry}"
    key="${CROP}:${CC}"
    if [[ -n "${SEEN[$key]:-}" ]]; then
      continue
    fi
    SEEN["$key"]=1
    echo "  CROP=${CROP} COUNTRY=${CC} SHAP_OUTPUT_DIR=${OUT_DIR} CYBENCH_BASELINES_DIR=${BASELINES_DIR} bash ${SLURM_DIR}/collect_shap_importance.sh"
  done
fi
