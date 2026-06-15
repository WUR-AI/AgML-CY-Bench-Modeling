#!/usr/bin/env bash
#
# Submit screening or walk-forward with --array sized from the job manifest.
#
# Usage (from repo root):
#   cybench/runs/slurm/submit_array.sh screening cybench/runs/slurm/benchmark_jobs_cpu.txt
#   cybench/runs/slurm/submit_array.sh walk_forward cybench/runs/slurm/benchmark_jobs_gpu.txt
#   cybench/runs/slurm/submit_array.sh screening cybench/runs/slurm/benchmark_jobs_gpu.txt --array 0
#
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: submit_array.sh <screening|walk_forward> [manifest] [--array RANGE] [--batch NAME] [--gpu|--cpu] [--dependency SPEC]

  manifest     Job list (default: cybench/runs/slurm/benchmark_jobs.txt)
  --array      SLURM array range (default: 0-(N-1); use 0 for first job only)
  --batch NAME Hydra experiment.name → ../output/NAME (default: baselines)
  --dependency SLURM dependency, e.g. afterok:12345 or afterok:111:222
  --gpu     Force GPU request (optional if manifest is GPU-only)
  --cpu     Force no GPU even for benchmark_jobs_gpu.txt

GPU is enabled automatically when every manifest row has needs_gpu=yes.
Default request: partition gpu + --gpus=1 + --time=2-00:00:00 (WUR lustre GPU cap).
Override with SLURM_GPU_PARTITION / SLURM_GPU_REQUEST / SLURM_GPU_TIME_LIMIT.

Examples:
  cybench/runs/slurm/submit_array.sh screening cybench/runs/slurm/benchmark_jobs_cpu.txt
  cybench/runs/slurm/submit_array.sh walk_forward cybench/runs/slurm/benchmark_jobs_gpu.txt
  cybench/runs/slurm/submit_array.sh screening cybench/runs/slurm/benchmark_jobs_gpu.txt --array 0
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

PHASE=$1
shift

case "${PHASE}" in
  screening) JOB_SCRIPT="screening.sh" ;;
  walk_forward) JOB_SCRIPT="walk_forward.sh" ;;
  *)
    echo "Unknown phase: ${PHASE}" >&2
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

MANIFEST="${SLURM_DIR}/benchmark_jobs.txt"
SBATCH_EXTRA=()
ARRAY_RANGE=""
GPU_MODE=auto
DEPENDENCY=""
CYBENCH_EXPERIMENT_NAME="${CYBENCH_EXPERIMENT_NAME:-baselines}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      GPU_MODE=yes
      shift
      ;;
    --cpu)
      GPU_MODE=no
      shift
      ;;
    --array)
      ARRAY_RANGE=$2
      shift 2
      ;;
    --batch)
      CYBENCH_EXPERIMENT_NAME=$2
      shift 2
      ;;
    --dependency)
      DEPENDENCY=$2
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      MANIFEST=$1
      shift
      ;;
  esac
done

# shellcheck source=/dev/null
source "${SLURM_DIR}/slurm_common.sh"
validate_experiment_name "${CYBENCH_EXPERIMENT_NAME}"
export CYBENCH_EXPERIMENT_NAME

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Manifest not found: ${MANIFEST}" >&2
  exit 1
fi

N=$(awk '!/^#/ && NF >= 7 {print}' "${MANIFEST}" | wc -l)
if [[ "${N}" -lt 1 ]]; then
  echo "No jobs in manifest: ${MANIFEST}" >&2
  exit 1
fi

N_GPU=$(awk '!/^#/ && NF >= 7 && $7 == "yes" { n++ } END { print n + 0 }' "${MANIFEST}")
if [[ "${GPU_MODE}" == auto ]]; then
  if [[ "${N_GPU}" -eq "${N}" ]]; then
    GPU_MODE=yes
  else
    GPU_MODE=no
  fi
fi
if [[ "${GPU_MODE}" == yes ]]; then
  # shellcheck source=/dev/null
  source "${SLURM_DIR}/slurm_common.sh"
  append_gpu_sbatch_args SBATCH_EXTRA
fi
if [[ -n "${DEPENDENCY}" ]]; then
  SBATCH_EXTRA+=(--dependency="${DEPENDENCY}")
fi

MAX=$((N - 1))
if [[ -z "${ARRAY_RANGE}" ]]; then
  ARRAY_RANGE="0-${MAX}"
fi

mkdir -p "output/${PHASE}"
export JOB_MANIFEST="${MANIFEST}"
PREDICTION_HORIZON="${PREDICTION_HORIZON:-eos}"
export PREDICTION_HORIZON

FIRST_JOB=$(awk '!/^#/ && NF >= 7 {print; exit}' "${MANIFEST}")
echo "Submitting ${PHASE} | jobs=${N} | array=${ARRAY_RANGE}"
echo "  manifest: ${MANIFEST}"
echo "  script:   ${SLURM_DIR}/${JOB_SCRIPT}"
echo "  horizon:  ${PREDICTION_HORIZON}"
echo "  batch:    ${CYBENCH_EXPERIMENT_NAME} (../output/${CYBENCH_EXPERIMENT_NAME})"
echo "  gpu:      ${GPU_MODE} ($([[ ${#SBATCH_EXTRA[@]} -gt 0 ]] && gpu_sbatch_summary || echo no))"
if [[ "${ARRAY_RANGE}" == "0" || "${ARRAY_RANGE}" == "0-0" ]]; then
  echo "  job[0]:   ${FIRST_JOB}"
fi
if [[ -n "${DEPENDENCY}" ]]; then
  echo "  depends:  ${DEPENDENCY}"
fi

JOB_ID=$(sbatch \
  --array="${ARRAY_RANGE}" \
  --export=ALL,JOB_MANIFEST="${JOB_MANIFEST}",PREDICTION_HORIZON="${PREDICTION_HORIZON}",CYBENCH_EXPERIMENT_NAME="${CYBENCH_EXPERIMENT_NAME}" \
  "${SBATCH_EXTRA[@]}" \
  "${SLURM_DIR}/${JOB_SCRIPT}" | awk '/Submitted batch job/{print $4}')
echo "job_id=${JOB_ID}"
