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
Usage: submit_array.sh <screening|walk_forward> [manifest] [--array RANGE] [--batch NAME] [--group GROUP] [--gpu|--cpu] [--dependency SPEC] [--repetitions N] [--resume]

  manifest     Job list (default: cybench/runs/slurm/benchmark_jobs.txt)
  --array      SLURM array range (default: 0-(N-1); use 0 for first job only)
  --batch NAME Hydra experiment.name → ../output/NAME (default: baselines)
  --group GROUP  cpu | naive | gpu — used in SLURM job name (default: infer from manifest)
  --dependency SLURM dependency, e.g. afterok:12345 or afterok:111:222
  --repetitions N  Walk-forward: total seeds from base (default: 1; seeds 42..42+N-1)
  --resume     Walk-forward: append missing seeds into latest run dir (skip if complete)
  --gpu     Force GPU partition + CUDA (even for mixed manifests)
  --cpu     Override GPU manifest: main partition, no GPU, CYBENCH_FORCE_CPU=1

GPU is enabled automatically when every manifest row has needs_gpu=yes (unless --cpu).
Default request: partition gpu + --gpus=1 + --time=2-00:00:00 (WUR lustre GPU cap).
Override with SLURM_GPU_PARTITION / SLURM_GPU_REQUEST / SLURM_GPU_TIME_LIMIT.

Examples:
  cybench/runs/slurm/submit_array.sh screening cybench/runs/slurm/benchmark_jobs_cpu.txt
  cybench/runs/slurm/submit_array.sh walk_forward cybench/runs/slurm/benchmark_jobs_gpu.txt
  cybench/runs/slurm/submit_array.sh screening cybench/runs/slurm/benchmark_jobs_gpu.txt --array 0
  cybench/runs/slurm/submit_array.sh screening cybench/runs/slurm/benchmark_jobs_gpu.txt --cpu --array 0
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
FORCE_CPU=false
DEPENDENCY=""
JOB_GROUP=""
CYBENCH_EXPERIMENT_NAME="${CYBENCH_EXPERIMENT_NAME:-baselines}"
WF_REPETITIONS="${WF_REPETITIONS:-1}"
WF_RESUME="${WF_RESUME:-no}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      GPU_MODE=yes
      shift
      ;;
    --cpu)
      GPU_MODE=no
      FORCE_CPU=true
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
    --group)
      JOB_GROUP=$2
      shift 2
      ;;
    --dependency)
      DEPENDENCY=$2
      shift 2
      ;;
    --repetitions)
      WF_REPETITIONS=$2
      shift 2
      ;;
    --resume)
      WF_RESUME=yes
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

# shellcheck source=/dev/null
source "${SLURM_DIR}/slurm_common.sh"

if is_force_cpu; then
  FORCE_CPU=true
  GPU_MODE=no
fi

validate_experiment_name "${CYBENCH_EXPERIMENT_NAME}"
if [[ "${PHASE}" == walk_forward ]]; then
  validate_wf_repetitions "${WF_REPETITIONS}"
fi
export CYBENCH_EXPERIMENT_NAME

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Manifest not found: ${MANIFEST}" >&2
  exit 1
fi

if [[ -z "${JOB_GROUP}" ]]; then
  JOB_GROUP=$(infer_manifest_group "${MANIFEST}")
fi

SUBMIT_MANIFEST="${MANIFEST}"
EXPANDED_MANIFEST=""
PER_YEAR_LARGE=no
should_expand_wf_per_seed() {
  [[ "${PHASE}" == walk_forward ]] || return 1
  [[ "$(basename "${MANIFEST}" .txt)" == *gpu* ]] || return 1
  [[ "${WF_REPETITIONS}" -gt 1 || "${WF_RESUME}" == yes ]] || return 1
  return 0
}

if should_expand_wf_per_seed; then
  EXPANDED_MANIFEST=$(mktemp "${TMPDIR:-/tmp}/cybench_wf_manifest.XXXXXX")
  PER_YEAR_LARGE=no
  if poetry run python -c "
from pathlib import Path
from cybench.runs.slurm.benchmark_completion_lib import manifest_wants_per_year_wf
import sys
sys.exit(0 if manifest_wants_per_year_wf(Path('${MANIFEST}')) else 1)
" 2>/dev/null; then
    PER_YEAR_LARGE=yes
  fi
  EXPAND_ARGS=(
    poetry run python "${SLURM_DIR}/expand_walk_forward_manifest.py"
    --input "${MANIFEST}"
    --output "${EXPANDED_MANIFEST}"
    --batch "${CYBENCH_EXPERIMENT_NAME}"
    --horizon "${PREDICTION_HORIZON:-eos}"
    --repetitions "${WF_REPETITIONS}"
    --per-seed
  )
  if [[ "${PER_YEAR_LARGE}" == yes ]]; then
    EXPAND_ARGS+=(--per-year-large)
  fi
  if [[ "${WF_RESUME}" == yes ]]; then
    EXPAND_ARGS+=(--resume)
  fi
  "${EXPAND_ARGS[@]}"
  SUBMIT_MANIFEST="${EXPANDED_MANIFEST}"
fi

N=$(awk '!/^#/ && NF >= 7 {print}' "${SUBMIT_MANIFEST}" | wc -l)
if [[ "${N}" -lt 1 ]]; then
  if should_expand_wf_per_seed; then
    echo "[WARN] No walk-forward GPU tasks after seed expansion (all seeds complete?)" >&2
    [[ -n "${EXPANDED_MANIFEST}" ]] && rm -f "${EXPANDED_MANIFEST}"
    exit 0
  fi
  echo "No jobs in manifest: ${MANIFEST}" >&2
  [[ -n "${EXPANDED_MANIFEST}" ]] && rm -f "${EXPANDED_MANIFEST}"
  exit 1
fi

N_GPU=$(awk '!/^#/ && NF >= 7 && $7 == "yes" { n++ } END { print n + 0 }' "${SUBMIT_MANIFEST}")
if [[ "${FORCE_CPU}" == true ]]; then
  GPU_MODE=no
elif [[ "${GPU_MODE}" == auto ]]; then
  if [[ "${N_GPU}" -eq "${N}" ]]; then
    GPU_MODE=yes
  else
    GPU_MODE=no
  fi
fi
resolved_mem=$(poetry run python -c "
from pathlib import Path
from cybench.runs.slurm.benchmark_submit_lib import slurm_memory_for_manifest
print(slurm_memory_for_manifest(Path('${SUBMIT_MANIFEST}')) or '')
" 2>/dev/null || true)
if [[ -n "${resolved_mem}" ]]; then
  SLURM_JOB_MEM="${resolved_mem}"
fi
if [[ "${GPU_MODE}" == yes ]]; then
  append_gpu_sbatch_args SBATCH_EXTRA
fi
SLURM_JOB_MEM="${SLURM_JOB_MEM:-${DEFAULT_SLURM_JOB_MEM}}"
export SLURM_JOB_MEM
append_job_mem_sbatch_args SBATCH_EXTRA
if [[ -n "${DEPENDENCY}" ]]; then
  SBATCH_EXTRA+=(--dependency="${DEPENDENCY}")
fi

MAX=$((N - 1))
if [[ -z "${ARRAY_RANGE}" ]]; then
  ARRAY_RANGE="0-${MAX}"
fi

mkdir -p "output/${PHASE}"
PREDICTION_HORIZON="${PREDICTION_HORIZON:-eos}"
export PREDICTION_HORIZON

JOB_MANIFEST=$(snapshot_job_manifest "${SUBMIT_MANIFEST}" "${PHASE}" "${CYBENCH_EXPERIMENT_NAME}" "${SLURM_DIR}")
export JOB_MANIFEST
[[ -n "${EXPANDED_MANIFEST}" ]] && rm -f "${EXPANDED_MANIFEST}"

SLURM_JOB_NAME=$(build_slurm_job_name "${PHASE}" "${JOB_GROUP}")

FIRST_JOB=$(awk '!/^#/ && NF >= 7 {print; exit}' "${JOB_MANIFEST}")
echo "Submitting ${PHASE} | jobs=${N} | array=${ARRAY_RANGE} | job-name=${SLURM_JOB_NAME}"
echo "  manifest: ${MANIFEST} (source)"
echo "  snapshot: ${JOB_MANIFEST}"
echo "  script:   ${SLURM_DIR}/${JOB_SCRIPT}"
echo "  horizon:  ${PREDICTION_HORIZON}"
echo "  batch:    ${CYBENCH_EXPERIMENT_NAME} (../output/${CYBENCH_EXPERIMENT_NAME})"
if [[ "${PHASE}" == walk_forward ]]; then
  echo "  repetitions: ${WF_REPETITIONS} (target seeds 42..$((42 + WF_REPETITIONS - 1)))"
  if [[ "$(basename "${MANIFEST}" .txt)" == *gpu* ]]; then
    if [[ "${PER_YEAR_LARGE:-no}" == yes ]]; then
      echo "  gpu seeds:  one SLURM task per seed×year for large countries (${N} tasks)"
    else
      echo "  gpu seeds:  one SLURM task per seed (${N} tasks from $(basename "${MANIFEST}"))"
    fi
  fi
  if [[ "${WF_RESUME}" == yes ]]; then
    echo "  resume:     yes (skip seeds already on disk)"
  fi
fi
echo "  gpu:      ${GPU_MODE} ($([[ ${#SBATCH_EXTRA[@]} -gt 0 ]] && gpu_sbatch_summary || echo no))"
if [[ "${FORCE_CPU}" == true ]]; then
  echo "  device:   CYBENCH_FORCE_CPU=1 (torch + TabPFN on CPU)"
fi
if [[ "${ARRAY_RANGE}" == "0" || "${ARRAY_RANGE}" == "0-0" ]]; then
  echo "  job[0]:   ${FIRST_JOB}"
fi
if [[ -n "${DEPENDENCY}" ]]; then
  echo "  depends:  ${DEPENDENCY}"
fi

if [[ "${FORCE_CPU}" == true ]]; then
  export CYBENCH_FORCE_CPU=1
fi

SBATCH_EXPORT="ALL,JOB_MANIFEST=${JOB_MANIFEST},PREDICTION_HORIZON=${PREDICTION_HORIZON},CYBENCH_EXPERIMENT_NAME=${CYBENCH_EXPERIMENT_NAME}"
if [[ -n "${HP_TRIALS:-}" ]]; then
  SBATCH_EXPORT+=",HP_TRIALS=${HP_TRIALS}"
fi
if [[ -n "${CYBENCH_EXTRA_OVERRIDES_FILE:-}" ]]; then
  SBATCH_EXPORT+=",CYBENCH_EXTRA_OVERRIDES_FILE=${CYBENCH_EXTRA_OVERRIDES_FILE}"
fi
if [[ "${PHASE}" == walk_forward ]]; then
  SBATCH_EXPORT+=",WF_REPETITIONS=${WF_REPETITIONS},WF_RESUME=${WF_RESUME}"
fi
if [[ "${FORCE_CPU}" == true ]]; then
  SBATCH_EXPORT+=",CYBENCH_FORCE_CPU=1"
fi

JOB_ID=$(sbatch \
  --job-name="${SLURM_JOB_NAME}" \
  --array="${ARRAY_RANGE}" \
  --export="${SBATCH_EXPORT}" \
  "${SBATCH_EXTRA[@]}" \
  "${SLURM_DIR}/${JOB_SCRIPT}" | awk '/Submitted batch job/{print $4}')
record_manifest_slurm_job "${JOB_MANIFEST}" "${JOB_ID}"
echo "job_id=${JOB_ID}"
