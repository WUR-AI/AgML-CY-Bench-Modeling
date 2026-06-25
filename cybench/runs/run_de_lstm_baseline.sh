#!/usr/bin/env bash
#
# Germany: CropBench-style LSTM baseline on CY-Bench data loading.
# Weekly AgERA5 (5 vars) + FPAR, mid-season, no static context.
# Non-positive yields are always dropped in DataFactory; quality outlier flags off.
#
# Usage (from repo root):
#   cybench/runs/run_de_lstm_baseline.sh all
#   cybench/runs/run_de_lstm_baseline.sh screening --local
#   cybench/runs/run_de_lstm_baseline.sh all --dry-run
#
# Fixed hyperparams (CropBench-style): no Optuna / HPO in screening.
#
set -euo pipefail

MODEL=lstm_baseline

usage() {
  cat <<EOF
Usage: run_de_lstm_baseline.sh <screening|walk_forward|all> [options]

Model ${MODEL} — end-to-end LSTM (hidden=256, 2 layers, last pool), CropBench training defaults.
Screening uses fixed hyperparams (no Optuna); same protocol as CropBench single-train setup.

Data (dataset/temporal=cropbench_lstm):
  - Weekly (7-day) AgERA5: tmin, tmax, tavg, prec, rad + FPAR
  - Middle-of-season horizon; static context dropped
  - Non-positive yields always excluded; no quality outlier filter (filter_samples=null)

Options:
  --batch NAME         experiment.name (default: baselines_de_lstm_baseline_v1)
  --crop CROP          wheat | maize (default: maize)
  --country CC         ISO country code (default: DE)
  --horizon H          end_of_sequence (default: middle-of-season)
  --local              Run poetry on this machine (default: SLURM)
  --cpu                SLURM on CPU partition
  --no-dependency      SLURM "all": walk-forward without afterok on screening
  --frozen-dir PATH    Walk-forward only (--local): screening artifact folder
  --collect            After walk-forward (--local): collect results
  --dry-run            Print commands without executing
  -h, --help           Show this help
EOF
}

PHASE=""
BATCH="baselines_de_lstm_baseline_v1"
CROP="maize"
COUNTRY="DE"
HORIZON="middle-of-season"
DEVICE="cuda"
FROZEN_DIR=""
COLLECT=false
DRY_RUN=false
USE_SLURM=true
SLURM_CPU=false
SLURM_NO_DEPENDENCY=false

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

PHASE=$1
shift

case "${PHASE}" in
  screening|walk_forward|all) ;;
  -h|--help) usage; exit 0 ;;
  *)
    echo "Unknown phase: ${PHASE}" >&2
    usage
    exit 1
    ;;
esac

while [[ $# -gt 0 ]]; do
  case "$1" in
    --batch) BATCH=$2; shift 2 ;;
    --crop) CROP=$2; shift 2 ;;
    --country) COUNTRY=$2; shift 2 ;;
    --horizon) HORIZON=$2; shift 2 ;;
    --device) DEVICE=$2; shift 2 ;;
    --frozen-dir) FROZEN_DIR=$2; shift 2 ;;
    --local) USE_SLURM=false; shift ;;
    --cpu) SLURM_CPU=true; DEVICE=cpu; shift ;;
    --no-dependency) SLURM_NO_DEPENDENCY=true; shift ;;
    --collect) COLLECT=true; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SLURM_DIR="${REPO_ROOT}/cybench/runs/slurm"
SUBMIT_ARRAY="${SLURM_DIR}/submit_array.sh"
cd "${REPO_ROOT}"

OUTPUT_ROOT="${REPO_ROOT}/../output"
BATCH_DIR="${OUTPUT_ROOT}/${BATCH}"
MANIFEST_DIR="${SLURM_DIR}/manifests/${BATCH}"
MANIFEST="${MANIFEST_DIR}/benchmark_jobs_gpu.txt"
OVERRIDES_FILE="${MANIFEST_DIR}/extra_overrides.txt"

if [[ "${DRY_RUN}" != true && "${USE_SLURM}" == false ]]; then
  mkdir -p "${BATCH_DIR}"
  OUTPUT_ROOT="$(cd "${OUTPUT_ROOT}" && pwd)"
  BATCH_DIR="${OUTPUT_ROOT}/${BATCH}"
fi

run_cmd() {
  if [[ "${DRY_RUN}" == true ]]; then
    printf '  '
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

write_extra_overrides_file() {
  mkdir -p "${MANIFEST_DIR}"
  cat > "${OVERRIDES_FILE}" <<EOF
dataset/temporal=cropbench_lstm
dataset.temporal.season.end_of_sequence=${HORIZON}
dataset.target.filter_samples=null
+process={name:select_context,drop:[year,sos_sin,sos_cos,eos_sin,eos_cos,loc_x,loc_y,loc_z,awc,bulk_density,drainage_class_1,drainage_class_2,drainage_class_3,drainage_class_4,drainage_class_5,drainage_class_6],keep:null}
EOF
}

write_job_manifest() {
  mkdir -p "${MANIFEST_DIR}"
  cat > "${MANIFEST}" <<EOF
# ${CROP} ${COUNTRY} CropBench-style ${MODEL} (weekly AgERA5+FPAR, fixed hparams, no HPO)
${CROP} ${COUNTRY} ${MODEL} torch no no yes
EOF
}

shared_local_overrides() {
  cat <<EOF
dataset/crop=${CROP}
dataset.country=${COUNTRY}
dataset.framework=torch
dataset.use_cache=true
dataset/temporal=cropbench_lstm
dataset.temporal.season.end_of_sequence=${HORIZON}
dataset.target.filter_samples=null
model=${MODEL}
+process={name:select_context,drop:[year,sos_sin,sos_cos,eos_sin,eos_cos,loc_x,loc_y,loc_z,awc,bulk_density,drainage_class_1,drainage_class_2,drainage_class_3,drainage_class_4,drainage_class_5,drainage_class_6],keep:null}
experiment.name=${BATCH}
experiment.n_repetitions=1
experiment.device=${DEVICE}
EOF
}

read_local_overrides() {
  mapfile -t OVERRIDES < <(shared_local_overrides)
}

horizon_tag() {
  poetry run python -c \
    "from cybench.util.prediction_horizon import prediction_horizon_tag; print(prediction_horizon_tag('${HORIZON}'))"
}

find_frozen_screening_dir() {
  local htag run_dir frozen
  htag=$(horizon_tag)
  run_dir=$(ls -td "${BATCH_DIR}/${CROP}_${COUNTRY}_${MODEL}_screening_${htag}_"* 2>/dev/null | head -1 || true)
  if [[ -z "${run_dir}" ]]; then
    echo "No screening run under ${BATCH_DIR} for ${CROP}/${COUNTRY}/${MODEL} horizon=${HORIZON}" >&2
    return 1
  fi
  frozen=$(find "${run_dir}" -name optimal_model.yaml -printf '%h\n' 2>/dev/null | head -1)
  if [[ -n "${frozen}" ]]; then
    echo "${frozen}"
    return 0
  fi
  echo "No optimal_model.yaml under ${run_dir}" >&2
  return 1
}

run_local_screening() {
  read_local_overrides
  local -a cmd=(poetry run python cybench/runs/run_experiments.py)
  cmd+=("${OVERRIDES[@]}")
  cmd+=(validation=screening)
  echo "== Local screening | ${CROP}/${COUNTRY} | model=${MODEL} | batch=${BATCH} | no HPO"
  run_cmd "${cmd[@]}"
}

run_local_walk_forward() {
  local frozen=$1
  if [[ ! -f "${frozen}/optimal_model.yaml" ]]; then
    echo "Missing optimal_model.yaml in ${frozen}" >&2
    exit 1
  fi
  read_local_overrides
  local -a cmd=(poetry run python cybench/runs/run_experiments.py)
  cmd+=("${OVERRIDES[@]}")
  cmd+=(validation=walk_forward "validation.frozen_screening_dir=${frozen}")
  echo "== Local walk-forward | ${CROP}/${COUNTRY} | frozen=${frozen}"
  run_cmd "${cmd[@]}"
}

run_local_collect() {
  local -a cmd=(
    poetry run python cybench/runs/analysis/collect_walk_forward_results.py
    --batch "${BATCH}"
    --output-dir "${OUTPUT_ROOT}/paper_${BATCH}"
  )
  echo "== Collect → ${OUTPUT_ROOT}/paper_${BATCH}"
  run_cmd "${cmd[@]}"
}

SLURM_LAST_JOB_ID=""

submit_slurm_phase() {
  local phase=$1
  local dependency=${2:-}
  local -a cmd=(
    "${SUBMIT_ARRAY}" "${phase}" "${MANIFEST}"
    --batch "${BATCH}" --array 0 --group gpu
  )
  if [[ "${SLURM_CPU}" == true ]]; then cmd+=(--cpu); else cmd+=(--gpu); fi
  if [[ -n "${dependency}" ]]; then cmd+=(--dependency "${dependency}"); fi
  echo "== SLURM ${phase} | ${CROP}/${COUNTRY} | model=${MODEL} | horizon=${HORIZON}"
  if [[ "${DRY_RUN}" == true ]]; then
    echo "  export PREDICTION_HORIZON=${HORIZON}"
    echo "  export CYBENCH_EXTRA_OVERRIDES_FILE=${OVERRIDES_FILE}"
    run_cmd "${cmd[@]}"
    SLURM_LAST_JOB_ID=""
    return 0
  fi
  if ! command -v sbatch >/dev/null 2>&1; then
    echo "sbatch not found. Use --local or run on a cluster login node." >&2
    exit 1
  fi
  local out
  out=$("${cmd[@]}" 2>&1)
  echo "${out}"
  SLURM_LAST_JOB_ID=$(echo "${out}" | awk -F= '/^job_id=/{print $2}')
  if [[ -z "${SLURM_LAST_JOB_ID}" ]]; then
    echo "Failed to parse SLURM job id" >&2
    exit 1
  fi
}

prepare_slurm_assets() {
  write_job_manifest
  write_extra_overrides_file
  export PREDICTION_HORIZON="${HORIZON}"
  export CYBENCH_EXTRA_OVERRIDES_FILE="${OVERRIDES_FILE}"
  export CYBENCH_EXPERIMENT_NAME="${BATCH}"
  echo "Manifest:  ${MANIFEST}"
  echo "Overrides: ${OVERRIDES_FILE}"
  echo "Model:     ${MODEL}"
}

run_slurm_all() {
  local wf_dep=""
  prepare_slurm_assets
  submit_slurm_phase screening
  if [[ "${SLURM_NO_DEPENDENCY}" != true && -n "${SLURM_LAST_JOB_ID}" ]]; then
    wf_dep="afterok:${SLURM_LAST_JOB_ID}"
  fi
  submit_slurm_phase walk_forward "${wf_dep}"
}

if [[ "${USE_SLURM}" == true ]]; then
  case "${PHASE}" in
    screening) prepare_slurm_assets; submit_slurm_phase screening ;;
    walk_forward) prepare_slurm_assets; submit_slurm_phase walk_forward ;;
    all) run_slurm_all ;;
  esac
else
  case "${PHASE}" in
    screening) run_local_screening ;;
    walk_forward)
      [[ -z "${FROZEN_DIR}" ]] && FROZEN_DIR=$(find_frozen_screening_dir)
      run_local_walk_forward "${FROZEN_DIR}"
      [[ "${COLLECT}" == true ]] && run_local_collect
      ;;
    all)
      run_local_screening
      if [[ "${DRY_RUN}" != true ]]; then
        FROZEN_DIR=$(find_frozen_screening_dir)
        run_local_walk_forward "${FROZEN_DIR}"
        [[ "${COLLECT}" == true ]] && run_local_collect
      fi
      ;;
  esac
fi

echo "Done."
