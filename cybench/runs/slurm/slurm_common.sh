# Shared helpers for screening / walk-forward SLURM jobs.
# Source from screening.sh or walk_forward.sh — do not execute directly.

# GPU jobs (submit_array.sh): WUR lustre uses partition gpu + --gpus=1.
# Override on other clusters, e.g.:
#   export SLURM_GPU_PARTITION=gpu_a100
#   export SLURM_GPU_REQUEST="--gres=gpu:1"
SLURM_GPU_PARTITION="${SLURM_GPU_PARTITION:-gpu}"
SLURM_GPU_REQUEST="${SLURM_GPU_REQUEST:---gpus=1}"
# GPU partition on WUR lustre: max walltime is often 2 days (screening.sh defaults to 4d).
SLURM_GPU_TIME_LIMIT="${SLURM_GPU_TIME_LIMIT:-2-00:00:00}"

append_gpu_sbatch_args() {
  local -n _extra=$1
  if [[ -n "${SLURM_GPU_PARTITION}" ]]; then
    _extra+=(--partition="${SLURM_GPU_PARTITION}")
  fi
  # shellcheck disable=SC2206
  local gpu_args=(${SLURM_GPU_REQUEST})
  _extra+=("${gpu_args[@]}")
  if [[ -n "${SLURM_GPU_TIME_LIMIT}" ]]; then
    _extra+=(--time="${SLURM_GPU_TIME_LIMIT}")
  fi
}

gpu_sbatch_summary() {
  local parts=()
  if [[ -n "${SLURM_GPU_PARTITION}" ]]; then
    parts+=("-p ${SLURM_GPU_PARTITION}")
  fi
  parts+=("${SLURM_GPU_REQUEST}")
  if [[ -n "${SLURM_GPU_TIME_LIMIT}" ]]; then
    parts+=("--time=${SLURM_GPU_TIME_LIMIT}")
  fi
  echo "${parts[*]}"
}

validate_experiment_name() {
  local name=$1
  if [[ ! "${name}" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]]; then
    echo "Invalid batch/experiment name '${name}': use letters, digits, . _ - (no slashes or spaces)" >&2
    exit 1
  fi
}

# Walk-forward seed repetitions (experiment.n_repetitions). Seeds are experiment.seed + i.
validate_wf_repetitions() {
  local n=$1
  if [[ ! "${n}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid walk-forward repetitions '${n}': must be a positive integer" >&2
    exit 1
  fi
}

# Per-batch working manifests (regenerate / split). Not passed to SLURM directly.
manifest_batch_dir() {
  local slurm_dir=$1 batch=$2
  local parent="${slurm_dir}/manifests"
  if [[ -d "${parent}/${batch}" ]]; then
    echo "${parent}/${batch}"
    return
  fi
  local key entry base
  key=$(printf '%s' "${batch}" | tr '[:upper:]' '[:lower:]')
  for entry in "${parent}"/baselines_*; do
    [[ -d "${entry}" ]] || continue
    base=$(basename "${entry}")
    if [[ "$(printf '%s' "${base}" | tr '[:upper:]' '[:lower:]')" == "${key}" ]]; then
      echo "${entry}"
      return
    fi
  done
  echo "${parent}/${batch}"
}

# Immutable copy at sbatch time — in-flight jobs keep this path even if working manifests change.
snapshot_job_manifest() {
  local src=$1 phase=$2 batch=$3 slurm_dir=$4
  local src_base dest_dir dest stamp
  src_base=$(basename "${src}" .txt)
  dest_dir="$(manifest_batch_dir "${slurm_dir}" "${batch}")"
  mkdir -p "${dest_dir}"
  stamp=$(date -u +%Y%m%dT%H%M%SZ)
  dest="${dest_dir}/${phase}_${src_base}_${stamp}_$$.txt"
  cp "${src}" "${dest}"
  echo "${dest}"
}

record_manifest_slurm_job() {
  local manifest=$1 job_id=$2
  printf '%s\n' "${job_id}" > "${manifest}.slurm_jobid"
}

is_force_cpu() {
  case "${CYBENCH_FORCE_CPU:-}" in
    1|yes|true|TRUE) return 0 ;;
    *) return 1 ;;
  esac
}

device_mode_label() {
  if is_force_cpu; then
    echo "cpu (CYBENCH_FORCE_CPU)"
  elif [[ "${NEEDS_GPU:-}" == "yes" ]]; then
    echo "cuda"
  else
    echo "cpu"
  fi
}

# Short labels for SLURM --job-name (max 63 chars on most clusters).
slurm_horizon_short() {
  case "${PREDICTION_HORIZON:-eos}" in
    eos) echo "eos" ;;
    middle-of-season | mid-season | mid_season) echo "mid" ;;
    quarter-of-season | quarter-season | quarter_season) echo "qtr" ;;
    eos-*) echo "eos${PREDICTION_HORIZON#eos-}" ;;
    *)
      echo "${PREDICTION_HORIZON}" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '_' | sed 's/^_*//;s/_*$//' | cut -c1-8
      ;;
  esac
}

slurm_crop_short() {
  case "$1" in
    maize) echo "mz" ;;
    wheat) echo "wh" ;;
    *)
      echo "$1" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '' | cut -c1-4
      ;;
  esac
}

# Per-array-task name (model visible in squeue once the task starts).
slurm_task_job_name() {
  local phase=$1
  local p crop_s
  case "${phase}" in
    screening) p="scr" ;;
    walk_forward) p="wf" ;;
    *) p=$(echo "${phase}" | cut -c1-2) ;;
  esac
  crop_s=$(slurm_crop_short "${CROP}")
  echo "cb_${p}_${MODEL}_${crop_s}${COUNTRY}"
}

# JobId for scontrol update: must be ArrayJobId_ArrayTaskId, not ArrayJobId alone
# (updating the parent id renames every array element to the same JobName).
slurm_task_job_id() {
  if [[ -n "${SLURM_ARRAY_JOB_ID:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  else
    echo "${SLURM_JOB_ID:-}"
  fi
}

# Rename this array element so squeue shows crop/country/model (batch stays in logs only).
slurm_update_task_job_name() {
  local phase=$1
  local name job_id
  job_id=$(slurm_task_job_id)
  [[ -n "${job_id}" ]] || return 0
  name=$(slurm_task_job_name "${phase}")
  name="${name:0:63}"
  scontrol update "JobId=${job_id}" "JobName=${name}" 2>/dev/null || true
}

# Infer cpu | naive | gpu from manifest filename when --group is omitted.
infer_manifest_group() {
  local manifest=$1
  local base
  base=$(basename "${manifest}" .txt)
  case "${base}" in
    *naive*) echo "nav" ;;
    *cpu*) echo "cpu" ;;
    *gpu*) echo "gpu" ;;
    *) echo "mix" ;;
  esac
}

# Array-level name (pending / array header). Per-task rename adds model — see slurm_update_task_job_name.
# Examples: cb_scr_cpu_eos | cb_wf_fcp_mid (gpu manifest + --cpu)
build_slurm_job_name() {
  local phase=$1 group=${2:-mix}
  local p h name
  case "${phase}" in
    screening) p="scr" ;;
    walk_forward) p="wf" ;;
    *) p=$(echo "${phase}" | cut -c1-3) ;;
  esac
  case "${group}" in
    cpu) group="cpu" ;;
    naive | nav) group="nav" ;;
    gpu)
      if is_force_cpu; then
        group="fcp"
      else
        group="gpu"
      fi
      ;;
    fcp | mix) ;;
    *) group=$(echo "${group}" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '_' | cut -c1-6) ;;
  esac
  h=$(slurm_horizon_short)
  name="cb_${p}_${group}_${h}"
  echo "${name:0:63}"
}

slurm_setup() {
  module load 2024
  module load Python/3.12.3-GCCcore-13.3.0

  # Repo root = three levels above cybench/runs/slurm.
  # SLURM_DIR is set by screening.sh / walk_forward.sh before sourcing this file.
  local _slurm_dir="${SLURM_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
  REPO_ROOT="${REPO_ROOT:-$(cd "${_slurm_dir}/../../.." && pwd)}"
  CYBENCH_EXPERIMENT_NAME="${CYBENCH_EXPERIMENT_NAME:-baselines}"
  validate_experiment_name "${CYBENCH_EXPERIMENT_NAME}"
  export CYBENCH_EXPERIMENT_NAME
  # Hydra writes to ../output/<experiment.name> relative to repo root (see conf/config.yaml).
  if [[ -n "${CYBENCH_BASELINES_DIR:-}" ]]; then
    BASELINES_DIR="${CYBENCH_BASELINES_DIR}"
  else
    local output_root="${REPO_ROOT}/../output"
    mkdir -p "${output_root}/${CYBENCH_EXPERIMENT_NAME}"
    BASELINES_DIR="$(cd "${output_root}/${CYBENCH_EXPERIMENT_NAME}" && pwd)"
  fi
  export BASELINES_DIR
  JOB_MANIFEST="${JOB_MANIFEST:-${REPO_ROOT}/cybench/runs/slurm/benchmark_jobs.txt}"
  PREDICTION_HORIZON="${PREDICTION_HORIZON:-eos}"
  export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
  cd "${REPO_ROOT}"
  setup_pytorch_cuda_libs
}

# Pip-installed CUDA wheels (nvidia-*) are not on the default loader path.
setup_pytorch_cuda_libs() {
  local site_pkg="${REPO_ROOT}/.venv/lib/python3.12/site-packages"
  local lib_dir
  [[ -d "${site_pkg}/nvidia" ]] || return 0
  shopt -s nullglob
  for lib_dir in "${site_pkg}"/nvidia/*/lib; do
    export LD_LIBRARY_PATH="${lib_dir}:${LD_LIBRARY_PATH:-}"
  done
  shopt -u nullglob
}

# Validate torch/torchvision/HF imports only when this task uses torch or CUDA.
slurm_validate_env() {
  local model=${1:-}
  if [[ "${FRAMEWORK:-}" != "torch" && "${NEEDS_GPU:-}" != "yes" ]]; then
    return 0
  fi
  local check_args=(--check-torch-stack)
  if [[ -n "${model}" ]]; then
    check_args+=(--model "${model}")
  fi
  if ! poetry run python "${SLURM_DIR}/check_env.py" "${check_args[@]}"; then
    echo "[FATAL] Environment check failed (torch/CUDA/HF). See stderr above; often:" >&2
    echo "  poetry run pip install --force-reinstall --no-cache-dir nvidia-nccl-cu12==2.21.5" >&2
    exit 1
  fi
  if [[ "${NEEDS_GPU:-}" == "yes" ]] && ! is_force_cpu; then
    if ! poetry run python "${SLURM_DIR}/check_env.py" --probe-cuda; then
      echo "[WARN] CUDA probe failed on this node; forcing CPU for this task" >&2
      export CYBENCH_FORCE_CPU=1
    fi
    if [[ -n "${model}" && "${model}" == "tabdpt" ]]; then
      if ! poetry run python "${SLURM_DIR}/check_env.py" --probe-tabular-fm "${model}"; then
        echo "[WARN] ${model} CUDA kernel probe failed; forcing CPU for this task" >&2
        export CYBENCH_FORCE_CPU=1
      fi
    fi
  fi
}

horizon_tag() {
  poetry run python -c "from cybench.util.prediction_horizon import prediction_horizon_tag; print(prediction_horizon_tag('${PREDICTION_HORIZON}'))"
}

# Hydra run dirs use model.name from cybench/conf/model/<slug>.yaml (e.g. average -> average_yield).
model_run_name() {
  local slug=$1
  poetry run python -c "
from pathlib import Path
from omegaconf import OmegaConf
slug = '${slug}'
cfg_path = Path('cybench/conf/model') / f'{slug}.yaml'
if not cfg_path.exists():
    print(slug)
else:
    print(OmegaConf.select(OmegaConf.load(cfg_path), 'name', default=slug))
"
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
# Set CYBENCH_FORCE_CPU=1 (submit_array.sh --cpu) to run GPU-manifest jobs on main/CPU.
configure_parallelism() {
  local -n _common=$1
  local force_cpu=false
  if is_force_cpu; then
    force_cpu=true
  fi
  if [[ "${FRAMEWORK}" == "pandas" ]]; then
    _common+=(dataset.framework=pandas)
    if [[ "${FEATURE_DESIGN}" == "yes" ]]; then
      _common+=(dataset/temporal=feature_design)
    fi
    _common+=("dataset.temporal.season.end_of_sequence=${PREDICTION_HORIZON}")
    _common+=(experiment.n_jobs=1)
    if [[ "${NEEDS_GPU}" == "yes" ]]; then
      if [[ "${force_cpu}" == true ]]; then
        # TabPFN on CPU (slow; use for queue bypass / pilot runs).
        _common+=(model.device=cpu model.allow_cpu_fallback=true experiment.device=cpu)
      else
        # Tabular foundation models: try CUDA; fall back to CPU on OOM / arch mismatch.
        _common+=(model.device=cuda model.allow_cpu_fallback=true)
      fi
    else
      _common+=(experiment.device=cpu)
    fi
  else
    if [[ "${force_cpu}" == true ]]; then
      _common+=(dataset.framework=torch experiment.device=cpu experiment.n_jobs=1)
    else
      _common+=(dataset.framework=torch experiment.device=cuda experiment.n_jobs=1)
    fi
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

# Optional Hydra overrides file (one override per line). Set CYBENCH_EXTRA_OVERRIDES_FILE.
append_extra_overrides_file() {
  local -n _extra=$1
  local line
  if [[ -z "${CYBENCH_EXTRA_OVERRIDES_FILE:-}" || ! -f "${CYBENCH_EXTRA_OVERRIDES_FILE}" ]]; then
    return 0
  fi
  while IFS= read -r line || [[ -n "${line}" ]]; do
    line="${line//$'\r'/}"
    [[ -z "${line}" || "${line}" =~ ^[[:space:]]*# ]] && continue
    _extra+=("${line}")
  done < "${CYBENCH_EXTRA_OVERRIDES_FILE}"
}

# Find .../<test_years>/optimal_model.yaml under the latest horizon-tagged screening run.
find_frozen_screening_dir() {
  local crop=$1 country=$2 model_slug=$3
  local htag model_name run_dir frozen
  htag=$(horizon_tag)
  model_name=$(model_run_name "${model_slug}")
  run_dir=$(ls -td "${BASELINES_DIR}/${crop}_${country}_${model_name}_screening_${htag}_"* 2>/dev/null | head -1 || true)
  if [[ -z "${run_dir}" ]]; then
    echo "No screening run for ${crop}/${country}/${model_name} (model=${model_slug}) horizon=${PREDICTION_HORIZON} (tag=${htag})" >&2
    return 1
  fi
  frozen=$(find "${run_dir}" -name optimal_model.yaml -printf '%h\n' 2>/dev/null | head -1)
  if [[ -n "${frozen}" ]]; then
    echo "${frozen}"
    return 0
  fi
  echo "No optimal_model.yaml in ${run_dir}" >&2
  return 1
}

# Latest walk-forward Hydra run folder for a crop/country/model (horizon-tagged).
find_latest_walk_forward_run_dir() {
  local crop=$1 country=$2 model_slug=$3
  local htag model_name
  htag=$(horizon_tag)
  model_name=$(model_run_name "${model_slug}")
  ls -td "${BASELINES_DIR}/${crop}_${country}_${model_name}_walk_forward_${htag}_"* 2>/dev/null | head -1 || true
}

discover_run_seeds_py() {
  local run_dir=$1
  poetry run python -c "
from pathlib import Path
from cybench.runs.analysis.collect_walk_forward_results import discover_run_seeds
for seed in discover_run_seeds(Path('${run_dir}')):
    print(seed)
"
}

# Plan seed schedule for walk-forward. Sets WF_RUN_DIR, WF_START_SEED, WF_RUN_REPS.
# Returns 0 run, 1 skip (all target seeds present), 2 error.
plan_walk_forward_seeds() {
  local crop=$1 country=$2 model_slug=$3
  local total=${WF_REPETITIONS:-1}
  local base=${WF_BASE_SEED:-42}
  local resume=${WF_RESUME:-no}

  validate_wf_repetitions "${total}"

  local -a target=()
  local i
  for ((i = 0; i < total; i++)); do
    target+=($((base + i)))
  done

  local run_dir=""
  local -a existing=()
  if [[ "${resume}" != "no" ]]; then
    run_dir=$(find_latest_walk_forward_run_dir "${crop}" "${country}" "${model_slug}")
    if [[ -n "${run_dir}" && -d "${run_dir}" ]]; then
      mapfile -t existing < <(discover_run_seeds_py "${run_dir}")
    fi
  fi

  local -a missing=()
  local s e found
  for s in "${target[@]}"; do
    found=false
    for e in "${existing[@]}"; do
      if [[ "${s}" == "${e}" ]]; then
        found=true
        break
      fi
    done
    if [[ "${found}" == false ]]; then
      missing+=("${s}")
    fi
  done

  if [[ ${#missing[@]} -eq 0 ]]; then
    if [[ ${#existing[@]} -gt 0 && "${resume}" != "no" ]]; then
      echo "[SKIP] Walk-forward seeds complete | target=${target[*]} | run=${run_dir}" >&2
      return 1
    fi
    WF_RUN_DIR=""
    WF_START_SEED="${base}"
    WF_RUN_REPS="${total}"
    return 0
  fi

  if [[ "${resume}" == "no" || -z "${run_dir}" ]]; then
    WF_RUN_DIR=""
    WF_START_SEED="${base}"
    WF_RUN_REPS="${total}"
    return 0
  fi

  WF_RUN_DIR="$(cd "${run_dir}" && pwd)"
  WF_START_SEED="${missing[0]}"
  WF_RUN_REPS="${#missing[@]}"
  return 0
}
