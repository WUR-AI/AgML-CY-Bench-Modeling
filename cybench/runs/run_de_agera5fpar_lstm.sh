#!/usr/bin/env bash
#
# Germany-only colleague-style LSTM: AgERA5 + FPAR, avg_pool tokenizer, no static context.
# Runs the same screening → walk-forward pipeline as the SLURM benchmark (other torch models).
#
# Usage (from repo root):
#   cybench/runs/run_de_agera5fpar_lstm.sh all
#   cybench/runs/run_de_agera5fpar_lstm.sh screening --batch my_de_lstm_v1
#   cybench/runs/run_de_agera5fpar_lstm.sh walk_forward --frozen-dir /path/to/.../2017_2018_2019_2020_2021
#   cybench/runs/run_de_agera5fpar_lstm.sh all --crop maize --device cpu --hp-trials 5 --dry-run
#
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_de_agera5fpar_lstm.sh <screening|walk_forward|all> [options]

Phases:
  screening      HPO on train/val, final fit on train+val, evaluate on held-out test block
  walk_forward   Rolling forecasts using frozen hyperparameters from screening
  all            screening, then walk_forward (auto-discovers latest screening artifacts)

Options:
  --batch NAME         experiment.name / output folder (default: baselines_de_agera5fpar_lstm_v1)
  --crop CROP          wheat | maize (default: maize)
  --country CC         ISO country code (default: DE)
  --horizon H          end_of_sequence (default: middle-of-season → run dirs *_mid_season_*)
  --device DEV         cuda | cpu (default: cuda)
  --hp-trials N        Optuna trials in screening (default: 20)
  --frozen-dir PATH    Screening split folder with optimal_model.yaml (walk_forward only)
  --collect            After walk_forward, run collect_walk_forward_results.py
  --dry-run            Print commands without executing
  -h, --help           Show this help

Output (relative to repo root):
  ../output/<batch>/maize_DE_lstm_lf_screening_mid_season_<timestamp>/
  ../output/<batch>/maize_DE_lstm_lf_walk_forward_mid_season_<timestamp>/
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
cd "${REPO_ROOT}"

OUTPUT_ROOT="${REPO_ROOT}/../output"
BATCH_DIR="${OUTPUT_ROOT}/${BATCH}"
if [[ "${DRY_RUN}" != true ]]; then
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

# Shared Hydra overrides (must match across screening and walk-forward).
shared_overrides() {
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

read_overrides() {
  mapfile -t OVERRIDES < <(shared_overrides)
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

run_screening() {
  read_overrides
  local -a cmd=(poetry run python cybench/runs/run_experiments.py)
  cmd+=("${OVERRIDES[@]}")
  cmd+=(
    validation=screening
    +hp_search=bayesian
    "hp_search.n_trials=${HP_TRIALS}"
  )
  echo "== Screening | ${CROP}/${COUNTRY} | batch=${BATCH} | device=${DEVICE} | out=${BATCH_DIR}"
  run_cmd "${cmd[@]}"
}

run_walk_forward() {
  local frozen=$1
  if [[ ! -f "${frozen}/optimal_model.yaml" ]]; then
    echo "Missing optimal_model.yaml in ${frozen}" >&2
    exit 1
  fi
  read_overrides
  local -a cmd=(poetry run python cybench/runs/run_experiments.py)
  cmd+=("${OVERRIDES[@]}")
  cmd+=(
    validation=walk_forward
    "validation.frozen_screening_dir=${frozen}"
  )
  echo "== Walk-forward | ${CROP}/${COUNTRY} | batch=${BATCH} | frozen=${frozen}"
  run_cmd "${cmd[@]}"
}

run_collect() {
  local -a cmd=(
    poetry run python cybench/runs/analysis/collect_walk_forward_results.py
    --batch "${BATCH}"
    --output-dir "${OUTPUT_ROOT}/paper_${BATCH}"
  )
  echo "== Collect walk-forward results → ${OUTPUT_ROOT}/paper_${BATCH}"
  run_cmd "${cmd[@]}"
}

case "${PHASE}" in
  screening)
    run_screening
    ;;
  walk_forward)
    if [[ -z "${FROZEN_DIR}" ]]; then
      FROZEN_DIR=$(find_frozen_screening_dir)
    fi
    run_walk_forward "${FROZEN_DIR}"
    if [[ "${COLLECT}" == true ]]; then
      run_collect
    fi
    ;;
  all)
    run_screening
    if [[ "${DRY_RUN}" == true ]]; then
      echo "(dry-run: would auto-discover frozen screening dir for walk-forward)"
      run_cmd echo walk_forward skipped in dry-run after screening
    else
      FROZEN_DIR=$(find_frozen_screening_dir)
      run_walk_forward "${FROZEN_DIR}"
      if [[ "${COLLECT}" == true ]]; then
        run_collect
      fi
    fi
    ;;
esac

echo "Done."
