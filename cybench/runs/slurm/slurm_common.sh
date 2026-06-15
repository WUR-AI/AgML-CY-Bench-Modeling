# Shared helpers for screening / walk-forward SLURM jobs.
# Source from screening.sh or walk_forward.sh — do not execute directly.

slurm_setup() {
  module load 2024
  module load Python/3.12.3-GCCcore-13.3.0

  # Repo root = three levels above cybench/runs/slurm.
  # SLURM_DIR is set by screening.sh / walk_forward.sh before sourcing this file.
  local _slurm_dir="${SLURM_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
  REPO_ROOT="${REPO_ROOT:-$(cd "${_slurm_dir}/../../.." && pwd)}"
  # Hydra writes to ../output/<experiment.name> relative to repo root (see conf/config.yaml).
  BASELINES_DIR="${CYBENCH_BASELINES_DIR:-$(cd "${REPO_ROOT}/../output/baselines" && pwd)}"
  JOB_MANIFEST="${JOB_MANIFEST:-${REPO_ROOT}/cybench/runs/slurm/benchmark_jobs.txt}"
  PREDICTION_HORIZON="${PREDICTION_HORIZON:-eos}"
  export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
  cd "${REPO_ROOT}"
}

horizon_tag() {
  poetry run python -c "from cybench.util.prediction_horizon import prediction_horizon_tag; print(prediction_horizon_tag('${PREDICTION_HORIZON}'))"
}

# Read manifest line for SLURM_ARRAY_TASK_ID (0-based; skips comments).
read_benchmark_job() {
  local line
  line=$(awk '!/^#/ && NF >= 7 {print}' "${JOB_MANIFEST}" | awk "NR == ${SLURM_ARRAY_TASK_ID}+1")
  if [[ -z "${line}" ]]; then
    echo "No job for array task ${SLURM_ARRAY_TASK_ID} in ${JOB_MANIFEST}" >&2
    exit 1
  fi
  read -r CROP COUNTRY MODEL FRAMEWORK HP_SEARCH FEATURE_DESIGN NEEDS_GPU <<< "${line}"
}

# CPU tabular: one Optuna trial at a time, sklearn uses all SLURM CPUs.
# GPU torch / TabPFN: one trial at a time on a single GPU.
configure_parallelism() {
  local -n _common=$1
  if [[ "${FRAMEWORK}" == "pandas" ]]; then
    _common+=(dataset.framework=pandas)
    if [[ "${FEATURE_DESIGN}" == "yes" ]]; then
      _common+=(dataset/temporal=feature_design)
    fi
    _common+=("dataset.temporal.season.end_of_sequence=${PREDICTION_HORIZON}")
    _common+=(experiment.n_jobs=1)
    if [[ "${NEEDS_GPU}" == "yes" ]]; then
      # TabPFN: PandasDataset but inference on CUDA (see model/tabpfn.yaml).
      _common+=(model.device=cuda model.allow_cpu_fallback=false)
    else
      _common+=(experiment.device=cpu)
    fi
  else
    _common+=(dataset.framework=torch experiment.device=cuda experiment.n_jobs=1)
    _common+=("dataset.temporal.season.end_of_sequence=${PREDICTION_HORIZON}")
  fi
}

configure_hpo_extras() {
  local -n _extra=$1
  if [[ "${HP_SEARCH}" == "yes" ]]; then
    _extra+=(+hp_search=bayesian hp_search.n_trials="${HP_TRIALS:-20}")
    _extra+=(
      "hp_search.storage.url=sqlite:///${TMPDIR:-/tmp}/optuna_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.db"
    )
  fi
  if [[ "${FEATURE_DESIGN}" == "yes" && "${FRAMEWORK}" == "pandas" ]]; then
    _extra+=(+feature_selection=mrmr)
  fi
}

# Find .../<test_years>/optimal_model.yaml under the latest screening run for crop/country/model.
# Matches horizon-tagged runs first (e.g. ..._screening_eos_<timestamp>), then legacy names.
find_frozen_screening_dir() {
  local crop=$1 country=$2 model=$3
  local htag run_dir frozen pattern
  htag=$(horizon_tag)
  for pattern in \
    "${BASELINES_DIR}/${crop}_${country}_${model}_screening_${htag}_*" \
    "${BASELINES_DIR}/${crop}_${country}_${model}_screening_*"
  do
    run_dir=$(ls -td ${pattern} 2>/dev/null | head -1 || true)
    if [[ -z "${run_dir}" ]]; then
      continue
    fi
    frozen=$(find "${run_dir}" -name optimal_model.yaml -printf '%h\n' 2>/dev/null | head -1)
    if [[ -n "${frozen}" ]]; then
      echo "${frozen}"
      return 0
    fi
  done
  echo "No screening run with optimal_model.yaml for ${crop}/${country}/${model} (horizon=${PREDICTION_HORIZON})" >&2
  return 1
}
