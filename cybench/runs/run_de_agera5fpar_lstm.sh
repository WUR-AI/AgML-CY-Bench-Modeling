#!/usr/bin/env bash
#
# Germany colleague-style LSTM: AgERA5 + FPAR, avg_pool tokenizer, no static context.
# Default: submit screening → walk-forward via SLURM (same pipeline as other torch models).
#
# Usage (from repo root):
#   cybench/runs/run_de_agera5fpar_lstm.sh all
#   cybench/runs/run_de_agera5fpar_lstm.sh screening --batch my_de_lstm_v1
#   cybench/runs/run_de_agera5fpar_lstm.sh all --local          # run poetry on this machine
#   cybench/runs/run_de_agera5fpar_lstm.sh all --cpu --dry-run  # SLURM on CPU partition
#
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_de_agera5fpar_lstm.sh <screening|walk_forward|all> [options]

Phases:
  screening      HPO on train/val, final fit on train+val, evaluate on held-out test block
  walk_forward   Rolling forecasts using frozen hyperparameters from screening
  all            screening, then walk-forward (SLURM: wf chained with afterok on screening)

Options:
  --batch NAME         experiment.name / output folder (default: baselines_de_agera5fpar_lstm_v1)
  --crop CROP          wheat | maize (default: maize)
  --country CC         ISO country code (default: DE)
  --horizon H          end_of_sequence (default: middle-of-season → run dirs *_mid_season_*)
  --hp-trials N        Optuna trials in screening (default: 20)
  --local              Run poetry/python on this machine (default: submit SLURM jobs)
  --cpu                SLURM: main partition, no GPU (CYBENCH_FORCE_CPU=1)
  --no-dependency      SLURM "all": submit walk-forward without afterok on screening
  --frozen-dir PATH    Walk-forward only, local mode: explicit screening artifact folder
  --collect            After walk-forward, run collect_walk_forward_results.py (local only)
  --dry-run            Print planned commands without executing
  -h, --help           Show this help

SLURM (default):
  Logs: output/screening/ and output/walk_forward/ under repo root
  Results: ../output/<batch>/maize_DE_lstm_lf_{screening,walk_forward}_mid_season_<timestamp>/

Local (--local):
  Same Hydra overrides; writes directly under ../output/<batch>/
EOF
}

PHASE=""
BATCH="baselines_de_agera5fpar_lstm_v1"
CROP="maize"
COUNTRY="DE"
HORIZON="middle-of-season"
DEVICE="cuda"
HP_TRIALS=20
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
  -h|--help)
    usage
    exit 0
    ;;
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
    --hp-trials) HP_TRIALS=$2; shift 2 ;;
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

# Dataset/model overrides shared by SLURM and local runs (crop/country/model set elsewhere on SLURM).
write_extra_overrides_file() {
  mkdir -p "${MANIFEST_DIR}"
  cat > "${OVERRIDES_FILE}" <<EOF
dataset/temporal=no_aggregate
model/torch_model/temporal_encoder/tokenizer=avg_pool
model.torch_model.embed_dim=8
~dataset.temporal.sources.ndvi
~dataset.temporal.sources.soil_moisture
dataset.temporal.sources.meteo.select=[tmin,tmax,tavg,prec,rad,et0,vpd]
+process={name:select_context,drop:[year,sos_sin,sos_cos,eos_sin,eos_cos,loc_x,loc_y,loc_z,awc,bulk_density,drainage_class_1,drainage_class_2,drainage_class_3,drainage_class_4,drainage_class_5,drainage_class_6],keep:null}
EOF
}

write_job_manifest() {
  mkdir -p "${MANIFEST_DIR}"
  cat > "${MANIFEST}" <<EOF
# ${CROP} ${COUNTRY} colleague-style lstm_lf (AgERA5 + FPAR, avg_pool, no static)
${CROP} ${COUNTRY} lstm_lf torch yes no yes
EOF
}

shared_local_overrides() {
  cat <<EOF
dataset/crop=${CROP}
dataset.country=${COUNTRY}
dataset.framework=torch
dataset.use_cache=true
dataset/temporal=no_aggregate
dataset.temporal.season.end_of_sequence=${HORIZON}
model=lstm_lf
model/torch_model/temporal_encoder/tokenizer=avg_pool
model.torch_model.embed_dim=8
~dataset.temporal.sources.ndvi
~dataset.temporal.sources.soil_moisture
dataset.temporal.sources.meteo.select=[tmin,tmax,tavg,prec,rad,et0,vpd]
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
  local htag model_name run_dir frozen
  htag=$(horizon_tag)
  model_name="lstm_lf"
  run_dir=$(ls -td "${BATCH_DIR}/${CROP}_${COUNTRY}_${model_name}_screening_${htag}_"* 2>/dev/null | head -1 || true)
  if [[ -z "${run_dir}" ]]; then
    echo "No screening run under ${BATCH_DIR} for ${CROP}/${COUNTRY}/lstm_lf horizon=${HORIZON}" >&2
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
  cmd+=(
    validation=screening
    +hp_search=bayesian
    "hp_search.n_trials=${HP_TRIALS}"
  )
  echo "== Local screening | ${CROP}/${COUNTRY} | batch=${BATCH} | device=${DEVICE}"
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
  cmd+=(
    validation=walk_forward
    "validation.frozen_screening_dir=${frozen}"
  )
  echo "== Local walk-forward | ${CROP}/${COUNTRY} | frozen=${frozen}"
  run_cmd "${cmd[@]}"
}

run_local_collect() {
  local -a cmd=(
    poetry run python cybench/runs/analysis/collect_walk_forward_results.py
    --batch "${BATCH}"
    --output-dir "${OUTPUT_ROOT}/paper_${BATCH}"
  )
  echo "== Collect walk-forward results → ${OUTPUT_ROOT}/paper_${BATCH}"
  run_cmd "${cmd[@]}"
}

submit_slurm_phase() {
  local phase=$1
  local dependency=${2:-}
  local -a cmd=(
    "${SUBMIT_ARRAY}" "${phase}" "${MANIFEST}"
    --batch "${BATCH}"
    --array 0
    --group gpu
  )
  if [[ "${SLURM_CPU}" == true ]]; then
    cmd+=(--cpu)
  else
    cmd+=(--gpu)
  fi
  if [[ -n "${dependency}" ]]; then
    cmd+=(--dependency "${dependency}")
  fi
  echo "== SLURM ${phase} | ${CROP}/${COUNTRY} | batch=${BATCH} | horizon=${HORIZON}"
  if [[ "${DRY_RUN}" == true ]]; then
    echo "  export PREDICTION_HORIZON=${HORIZON}"
    echo "  export HP_TRIALS=${HP_TRIALS}"
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
    echo "Failed to parse SLURM job id from submit_array output" >&2
    exit 1
  fi
}

SLURM_LAST_JOB_ID=""

prepare_slurm_assets() {
  write_job_manifest
  write_extra_overrides_file
  export PREDICTION_HORIZON="${HORIZON}"
  export HP_TRIALS="${HP_TRIALS}"
  export CYBENCH_EXTRA_OVERRIDES_FILE="${OVERRIDES_FILE}"
  export CYBENCH_EXPERIMENT_NAME="${BATCH}"
  echo "Manifest:     ${MANIFEST}"
  echo "Overrides:    ${OVERRIDES_FILE}"
  echo "Output batch: ../output/${BATCH}/"
}

run_slurm_screening() {
  prepare_slurm_assets
  submit_slurm_phase screening
}

run_slurm_walk_forward() {
  local dependency=${1:-}
  prepare_slurm_assets
  submit_slurm_phase walk_forward "${dependency}"
}

run_slurm_all() {
  local wf_dep=""
  prepare_slurm_assets
  submit_slurm_phase screening
  if [[ "${SLURM_NO_DEPENDENCY}" != true && -n "${SLURM_LAST_JOB_ID}" ]]; then
    wf_dep="afterok:${SLURM_LAST_JOB_ID}"
  fi
  submit_slurm_phase walk_forward "${wf_dep}"
  if [[ "${COLLECT}" == true ]]; then
    echo ""
    echo "Note: --collect is not submitted as a SLURM job. After walk-forward completes, run:"
    echo "  poetry run python cybench/runs/analysis/collect_walk_forward_results.py \\"
    echo "    --batch ${BATCH} --output-dir ${OUTPUT_ROOT}/paper_${BATCH}"
  fi
}

if [[ "${USE_SLURM}" == true ]]; then
  case "${PHASE}" in
    screening) run_slurm_screening ;;
    walk_forward) run_slurm_walk_forward ;;
    all) run_slurm_all ;;
  esac
else
  case "${PHASE}" in
    screening)
      run_local_screening
      ;;
    walk_forward)
      if [[ -z "${FROZEN_DIR}" ]]; then
        FROZEN_DIR=$(find_frozen_screening_dir)
      fi
      run_local_walk_forward "${FROZEN_DIR}"
      if [[ "${COLLECT}" == true ]]; then
        run_local_collect
      fi
      ;;
    all)
      run_local_screening
      if [[ "${DRY_RUN}" == true ]]; then
        echo "(dry-run: would auto-discover frozen screening dir for walk-forward)"
      else
        FROZEN_DIR=$(find_frozen_screening_dir)
        run_local_walk_forward "${FROZEN_DIR}"
        if [[ "${COLLECT}" == true ]]; then
          run_local_collect
        fi
      fi
      ;;
  esac
fi

echo "Done."
