#!/usr/bin/env bash
#
# Submit screening or walk-forward with --array sized from the job manifest.
#
# Usage (from repo root):
#   cybench/runs/slurm/submit_array.sh screening cybench/runs/slurm/benchmark_jobs_cpu.txt
#   cybench/runs/slurm/submit_array.sh walk_forward cybench/runs/slurm/benchmark_jobs_gpu.txt --gpu
#
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: submit_array.sh <screening|walk_forward> [manifest] [--gpu]

  manifest  Job list (default: cybench/runs/slurm/benchmark_jobs.txt)
  --gpu     Pass --gres=gpu:1 to sbatch (torch / TabPFN jobs)

Examples:
  cybench/runs/slurm/submit_array.sh screening cybench/runs/slurm/benchmark_jobs_cpu.txt
  cybench/runs/slurm/submit_array.sh walk_forward cybench/runs/slurm/benchmark_jobs_gpu.txt --gpu
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

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      SBATCH_EXTRA+=(--gres=gpu:1)
      shift
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

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Manifest not found: ${MANIFEST}" >&2
  exit 1
fi

N=$(awk '!/^#/ && NF >= 7 {print}' "${MANIFEST}" | wc -l)
if [[ "${N}" -lt 1 ]]; then
  echo "No jobs in manifest: ${MANIFEST}" >&2
  exit 1
fi
MAX=$((N - 1))

mkdir -p "output/${PHASE}"
export JOB_MANIFEST="${MANIFEST}"

echo "Submitting ${PHASE} | jobs=${N} | array=0-${MAX}"
echo "  manifest: ${MANIFEST}"
echo "  script:   ${SLURM_DIR}/${JOB_SCRIPT}"

sbatch \
  --array="0-${MAX}" \
  --export=ALL,JOB_MANIFEST="${JOB_MANIFEST}" \
  "${SBATCH_EXTRA[@]}" \
  "${SLURM_DIR}/${JOB_SCRIPT}"
