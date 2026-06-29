#!/usr/bin/env bash
#
# Submit the full benchmark pipeline for one prediction horizon:
#   1) split manifest (cpu / naive / gpu)
#   2) screening arrays (all three)
#   3) walk-forward arrays (optionally chained with SLURM --dependency)
#
# Usage (from repo root):
#   cybench/runs/slurm/submit_benchmark.sh all --horizon eos
#   cybench/runs/slurm/submit_benchmark.sh screening --horizon eos
#   cybench/runs/slurm/submit_benchmark.sh all --horizon middle-of-season --regenerate --countries DE NL
#   cybench/runs/slurm/submit_benchmark.sh screening --horizon eos --array 0 --only gpu
#   cybench/runs/slurm/submit_benchmark.sh all --horizon eos --regenerate --countries DE --only gpu --force-cpu
#
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: submit_benchmark.sh <screening|walk_forward|all> [options]

Phases:
  screening     Submit cpu + naive + gpu screening arrays
  walk_forward  Submit cpu + naive + gpu walk-forward arrays
  all           Screening, then walk-forward with --dependency=afterok:<screening jobs>

Options:
  --horizon H       PREDICTION_HORIZON (default: eos)
  --batch NAME      Hydra experiment.name / output dir under ../output/NAME (default: baselines)
  --regenerate      Run generate_job_manifest.py, then split manifests
  --countries C...  Passed to generate_job_manifest.py (with --regenerate)
  --array RANGE     SLURM array range for every submit (default: full manifest)
  --only GROUP      One group only: cpu | naive | gpu
  --cpu             GPU group on main partition (torch/TabPFN on CPU; slow)
  --force-cpu       Alias for --cpu
  --no-dependency   For "all": submit walk-forward without afterok
  --repetitions N   Walk-forward: experiment.n_repetitions (default: 1; seeds 42..42+N-1)
  --repetitions N   Walk-forward: total seeds from base 42 (default: 1)
  --resume          Walk-forward: append missing seeds into latest run (no re-run of 42)
  --skip-naive      Omit naive (average, trend) manifests
  -n, --dry-run     Print commands without sbatch

Examples:
  cybench/runs/slurm/submit_benchmark.sh all --horizon eos
  cybench/runs/slurm/submit_benchmark.sh all --horizon eos --batch baselines_pilot_2026q2
  cybench/runs/slurm/submit_benchmark.sh screening --horizon eos --array 0 --only gpu
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

PHASE_MODE=$1
shift

case "${PHASE_MODE}" in
  screening|walk_forward|all) ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    echo "Unknown phase mode: ${PHASE_MODE}" >&2
    usage
    exit 1
    ;;
esac

if [[ -f "${SLURM_SUBMIT_DIR:-}/cybench/runs/slurm/slurm_common.sh" ]]; then
  SLURM_DIR="${SLURM_SUBMIT_DIR}/cybench/runs/slurm"
elif [[ -n "${REPO_ROOT:-}" && -f "${REPO_ROOT}/cybench/runs/slurm/slurm_common.sh" ]]; then
  SLURM_DIR="${REPO_ROOT}/cybench/runs/slurm"
else
  SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

SUBMIT_ARRAY="${SLURM_DIR}/submit_array.sh"
SHARED_MANIFEST="${SLURM_DIR}/benchmark_jobs.txt"
MANIFEST_ROOT=""
BASE_MANIFEST=""
MANIFEST_CPU=""
MANIFEST_NAIVE=""
MANIFEST_GPU=""

PREDICTION_HORIZON="${PREDICTION_HORIZON:-eos}"
CYBENCH_EXPERIMENT_NAME="${CYBENCH_EXPERIMENT_NAME:-baselines}"
REGENERATE=false
ARRAY_ARG=()
ONLY_GROUP=""
FORCE_CPU=false
USE_DEPENDENCY=true
SKIP_NAIVE=false
DRY_RUN=false
COUNTRIES=()
WF_REPETITIONS="${WF_REPETITIONS:-1}"
WF_RESUME="${WF_RESUME:-no}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --horizon)
      PREDICTION_HORIZON=$2
      shift 2
      ;;
    --batch)
      CYBENCH_EXPERIMENT_NAME=$2
      shift 2
      ;;
    --regenerate)
      REGENERATE=true
      shift
      ;;
    --countries)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        COUNTRIES+=("$1")
        shift
      done
      ;;
    --array)
      ARRAY_ARG=(--array "$2")
      shift 2
      ;;
    --only)
      ONLY_GROUP=$2
      shift 2
      ;;
    --force-cpu|--cpu)
      FORCE_CPU=true
      shift
      ;;
    --no-dependency)
      USE_DEPENDENCY=false
      shift
      ;;
    --repetitions)
      WF_REPETITIONS=$2
      shift 2
      ;;
    --resume)
      WF_RESUME=yes
      shift
      ;;
    --skip-naive)
      SKIP_NAIVE=true
      shift
      ;;
    -n|--dry-run)
      DRY_RUN=true
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

# shellcheck source=/dev/null
source "${SLURM_DIR}/slurm_common.sh"
validate_experiment_name "${CYBENCH_EXPERIMENT_NAME}"
if [[ "${PHASE_MODE}" == walk_forward || "${PHASE_MODE}" == all ]]; then
  validate_wf_repetitions "${WF_REPETITIONS}"
fi
export PREDICTION_HORIZON
export CYBENCH_EXPERIMENT_NAME

init_manifest_paths() {
  MANIFEST_ROOT="$(manifest_batch_dir "${SLURM_DIR}" "${CYBENCH_EXPERIMENT_NAME}")"
  BASE_MANIFEST="${MANIFEST_ROOT}/benchmark_jobs.txt"
  MANIFEST_CPU="${MANIFEST_ROOT}/benchmark_jobs_cpu.txt"
  MANIFEST_NAIVE="${MANIFEST_ROOT}/benchmark_jobs_naive.txt"
  MANIFEST_GPU="${MANIFEST_ROOT}/benchmark_jobs_gpu.txt"
}

split_manifests() {
  if [[ ! -f "${BASE_MANIFEST}" ]]; then
    echo "Missing ${BASE_MANIFEST}. Run with --regenerate or generate_job_manifest.py" >&2
    exit 1
  fi
  mkdir -p "${MANIFEST_ROOT}"
  awk '$7 == "no" && $6 == "yes"' "${BASE_MANIFEST}" > "${MANIFEST_CPU}"
  awk '$5 == "no" && $6 == "no" && $7 == "no"' "${BASE_MANIFEST}" > "${MANIFEST_NAIVE}"
  awk '$7 == "yes"' "${BASE_MANIFEST}" > "${MANIFEST_GPU}"
  echo "[INFO] Split manifests (batch=${CYBENCH_EXPERIMENT_NAME}, horizon=${PREDICTION_HORIZON}):"
  echo "  root:  ${MANIFEST_ROOT}"
  echo "  cpu:   $(awk '!/^#/ && NF>=7' "${MANIFEST_CPU}" | wc -l) jobs -> ${MANIFEST_CPU}"
  echo "  naive: $(awk '!/^#/ && NF>=7' "${MANIFEST_NAIVE}" | wc -l) jobs -> ${MANIFEST_NAIVE}"
  echo "  gpu:   $(awk '!/^#/ && NF>=7' "${MANIFEST_GPU}" | wc -l) jobs -> ${MANIFEST_GPU}"
}

regenerate_manifest() {
  mkdir -p "${MANIFEST_ROOT}"
  local cmd=(poetry run python "${SLURM_DIR}/generate_job_manifest.py" -o "${BASE_MANIFEST}")
  cmd+=(--horizon "${PREDICTION_HORIZON}")
  if [[ ${#COUNTRIES[@]} -gt 0 ]]; then
    cmd+=(--countries "${COUNTRIES[@]}")
  fi
  if [[ "${DRY_RUN}" == true ]]; then
    echo "[DRY-RUN] ${cmd[*]}"
  else
    "${cmd[@]}"
  fi
  split_manifests
}

ensure_batch_manifests() {
  if [[ ! -f "${BASE_MANIFEST}" ]]; then
    if [[ -f "${SHARED_MANIFEST}" ]]; then
      echo "[INFO] Seeding batch manifests from ${SHARED_MANIFEST} -> ${MANIFEST_ROOT}"
      mkdir -p "${MANIFEST_ROOT}"
      cp "${SHARED_MANIFEST}" "${BASE_MANIFEST}"
    else
      echo "No manifests for batch '${CYBENCH_EXPERIMENT_NAME}' under ${MANIFEST_ROOT}." >&2
      echo "Run with --regenerate or generate_job_manifest.py -o ${BASE_MANIFEST}" >&2
      exit 1
    fi
  fi
  # Always re-split: orchestrate_benchmark_complete overwrites benchmark_jobs.txt
  # with a partial retry list; stale cpu/naive/gpu files must not be reused.
  split_manifests
}

init_manifest_paths

if [[ "${REGENERATE}" == true ]]; then
  regenerate_manifest
else
  ensure_batch_manifests
fi

manifest_for_group() {
  case "$1" in
    cpu) echo "${MANIFEST_CPU}" ;;
    naive) echo "${MANIFEST_NAIVE}" ;;
    gpu) echo "${MANIFEST_GPU}" ;;
    *)
      echo "Unknown group: $1 (use cpu, naive, or gpu)" >&2
      exit 1
      ;;
  esac
}

groups_to_run() {
  if [[ -n "${ONLY_GROUP}" ]]; then
    echo "${ONLY_GROUP}"
    return
  fi
  echo cpu
  if [[ "${SKIP_NAIVE}" != true ]]; then
    echo naive
  fi
  echo gpu
}

count_jobs() {
  awk '!/^#/ && NF >= 7 { n++ } END { print n + 0 }' "$1"
}

submit_one() {
  local phase=$1 manifest=$2 dep=${3:-} group=${4:-}
  if [[ "$(count_jobs "${manifest}")" -lt 1 ]]; then
    echo "[SKIP] ${phase} ${manifest}: no jobs"
    return 0
  fi
  local -a cmd=(env PREDICTION_HORIZON="${PREDICTION_HORIZON}" CYBENCH_EXPERIMENT_NAME="${CYBENCH_EXPERIMENT_NAME}")
  if [[ "${phase}" == walk_forward ]]; then
    cmd+=(WF_REPETITIONS="${WF_REPETITIONS}" WF_RESUME="${WF_RESUME}")
  fi
  cmd+=("${SUBMIT_ARRAY}" "${phase}" "${manifest}")
  if [[ -n "${group}" ]]; then
    cmd+=(--group "${group}")
  fi
  if [[ "${phase}" == walk_forward ]]; then
    cmd+=(--repetitions "${WF_REPETITIONS}")
    if [[ "${WF_RESUME}" == yes ]]; then
      cmd+=(--resume)
    fi
  fi
  if [[ ${#ARRAY_ARG[@]} -gt 0 ]]; then
    cmd+=("${ARRAY_ARG[@]}")
  fi
  if [[ "${FORCE_CPU}" == true && "${group}" == gpu ]]; then
    cmd+=(--cpu)
  fi
  if [[ -n "${dep}" ]]; then
    cmd+=(--dependency "${dep}")
  fi
  if [[ "${DRY_RUN}" == true ]]; then
    echo "[DRY-RUN] ${cmd[*]}" >&2
    return 0
  fi
  local out job_id
  out=$("${cmd[@]}" 2>&1 | tee /dev/stderr)
  job_id=$(echo "$out" | awk -F= '/^job_id=/{print $2; exit}')
  if [[ -z "${job_id}" ]]; then
    echo "[ERROR] No job_id from: ${cmd[*]}" >&2
    exit 1
  fi
  echo "${job_id}"
}

declare -A SCREEN_JOB

run_screening() {
  local group manifest job_id
  echo ""
  echo "=== Screening | horizon=${PREDICTION_HORIZON} | batch=${CYBENCH_EXPERIMENT_NAME} ==="
  while read -r group; do
    manifest=$(manifest_for_group "${group}")
    echo "--- group=${group} ---"
    job_id=$(submit_one screening "${manifest}" "" "${group}")
    if [[ -n "${job_id}" ]]; then
      SCREEN_JOB["${group}"]="${job_id}"
    fi
  done < <(groups_to_run)
}

run_walk_forward() {
  local group manifest dep job_id
  echo ""
  echo "=== Walk-forward | horizon=${PREDICTION_HORIZON} | batch=${CYBENCH_EXPERIMENT_NAME} | repetitions=${WF_REPETITIONS} | resume=${WF_RESUME} ==="
  while read -r group; do
    manifest=$(manifest_for_group "${group}")
    dep=""
    if [[ "${USE_DEPENDENCY}" == true && -n "${SCREEN_JOB[${group}]:-}" ]]; then
      dep="afterok:${SCREEN_JOB[${group}]}"
    fi
    echo "--- group=${group} ---"
    if [[ "${DRY_RUN}" == true ]]; then
      submit_one walk_forward "${manifest}" "${dep}" "${group}"
    else
      submit_one walk_forward "${manifest}" "${dep}" "${group}" >/dev/null
    fi
  done < <(groups_to_run)
}

case "${PHASE_MODE}" in
  screening)
    run_screening
    ;;
  walk_forward)
    USE_DEPENDENCY=false
    run_walk_forward
    ;;
  all)
    run_screening
    run_walk_forward
    ;;
esac

echo ""
echo "[DONE] ${PHASE_MODE} submitted for horizon=${PREDICTION_HORIZON} batch=${CYBENCH_EXPERIMENT_NAME}"
