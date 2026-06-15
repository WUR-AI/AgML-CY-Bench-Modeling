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
  --no-dependency   For "all": submit walk-forward without afterok
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
BASE_MANIFEST="${SLURM_DIR}/benchmark_jobs.txt"
MANIFEST_CPU="${SLURM_DIR}/benchmark_jobs_cpu.txt"
MANIFEST_NAIVE="${SLURM_DIR}/benchmark_jobs_naive.txt"
MANIFEST_GPU="${SLURM_DIR}/benchmark_jobs_gpu.txt"

PREDICTION_HORIZON="${PREDICTION_HORIZON:-eos}"
CYBENCH_EXPERIMENT_NAME="${CYBENCH_EXPERIMENT_NAME:-baselines}"
REGENERATE=false
ARRAY_ARG=()
ONLY_GROUP=""
USE_DEPENDENCY=true
SKIP_NAIVE=false
DRY_RUN=false
COUNTRIES=()

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
    --no-dependency)
      USE_DEPENDENCY=false
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
export PREDICTION_HORIZON
export CYBENCH_EXPERIMENT_NAME

split_manifests() {
  if [[ ! -f "${BASE_MANIFEST}" ]]; then
    echo "Missing ${BASE_MANIFEST}. Run with --regenerate or generate_job_manifest.py" >&2
    exit 1
  fi
  awk '$7 == "no" && $6 == "yes"' "${BASE_MANIFEST}" > "${MANIFEST_CPU}"
  awk '$5 == "no" && $6 == "no" && $7 == "no"' "${BASE_MANIFEST}" > "${MANIFEST_NAIVE}"
  awk '$7 == "yes"' "${BASE_MANIFEST}" > "${MANIFEST_GPU}"
  echo "[INFO] Split manifests (horizon=${PREDICTION_HORIZON}):"
  echo "  cpu:   $(awk '!/^#/ && NF>=7' "${MANIFEST_CPU}" | wc -l) jobs -> ${MANIFEST_CPU}"
  echo "  naive: $(awk '!/^#/ && NF>=7' "${MANIFEST_NAIVE}" | wc -l) jobs -> ${MANIFEST_NAIVE}"
  echo "  gpu:   $(awk '!/^#/ && NF>=7' "${MANIFEST_GPU}" | wc -l) jobs -> ${MANIFEST_GPU}"
}

regenerate_manifest() {
  local cmd=(poetry run python "${SLURM_DIR}/generate_job_manifest.py")
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

if [[ "${REGENERATE}" == true ]]; then
  regenerate_manifest
elif [[ ! -f "${MANIFEST_CPU}" || ! -f "${MANIFEST_GPU}" ]]; then
  echo "[INFO] Split manifests missing; creating from ${BASE_MANIFEST}"
  split_manifests
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
  local phase=$1 manifest=$2 dep=${3:-}
  if [[ "$(count_jobs "${manifest}")" -lt 1 ]]; then
    echo "[SKIP] ${phase} ${manifest}: no jobs"
    return 0
  fi
  local cmd=(env PREDICTION_HORIZON="${PREDICTION_HORIZON}" CYBENCH_EXPERIMENT_NAME="${CYBENCH_EXPERIMENT_NAME}" "${SUBMIT_ARRAY}" "${phase}" "${manifest}")
  if [[ ${#ARRAY_ARG[@]} -gt 0 ]]; then
    cmd+=("${ARRAY_ARG[@]}")
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
    job_id=$(submit_one screening "${manifest}")
    if [[ -n "${job_id}" ]]; then
      SCREEN_JOB["${group}"]="${job_id}"
    fi
  done < <(groups_to_run)
}

run_walk_forward() {
  local group manifest dep job_id
  echo ""
  echo "=== Walk-forward | horizon=${PREDICTION_HORIZON} | batch=${CYBENCH_EXPERIMENT_NAME} ==="
  while read -r group; do
    manifest=$(manifest_for_group "${group}")
    dep=""
    if [[ "${USE_DEPENDENCY}" == true && -n "${SCREEN_JOB[${group}]:-}" ]]; then
      dep="afterok:${SCREEN_JOB[${group}]}"
    fi
    echo "--- group=${group} ---"
    if [[ "${DRY_RUN}" == true ]]; then
      submit_one walk_forward "${manifest}" "${dep}"
    else
      submit_one walk_forward "${manifest}" "${dep}" >/dev/null
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
