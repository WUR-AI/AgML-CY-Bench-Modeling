#!/usr/bin/env bash
#
# Submit benchmark jobs for many countries: discover pending batches, route torch/TabPFN
# to gpu or --cpu by region count, always keep tabular jobs on cpu.
#
# Usage (from repo root on anunna):
#   cybench/runs/slurm/orchestrate_benchmark_submit.sh --list
#   cybench/runs/slurm/orchestrate_benchmark_submit.sh --dry-run
#   cybench/runs/slurm/orchestrate_benchmark_submit.sh --countries PL IT --horizon eos
#   cybench/runs/slurm/orchestrate_benchmark_submit.sh --region-threshold 50 --max 3
#
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: orchestrate_benchmark_submit.sh [options]

Discover countries / horizons without manifests yet, then call submit_benchmark.sh
per batch. Tabular (cpu) and naive jobs always use the cpu/main arrays; only the
gpu manifest group is routed to the gpu partition or --cpu based on region count.

Options:
  --list              Show plan (country, regions, gpu vs cpu) and exit
  --dry-run           Print submit_benchmark.sh commands without sbatch
  --countries CC ...  Limit to these countries (default: all with data on disk)
  --horizon H ...     Default: eos middle-of-season (alias mid → middle-of-season)
  --region-threshold N  gpu partition when country has >= N regions (default: 50)
  --version N         Batch version suffix (default: 3)
  --phase MODE        screening | walk_forward | all (default: all)
  --force             Submit even if manifest batch dir already exists
  --all-countries     Include already-filed batches in --list
  --max N             Submit at most N batches this run
  --manifest-root DIR Override slurm/manifests parent
  --data-dir DIR      Override cybench/data

Examples:
  orchestrate_benchmark_submit.sh --list
  orchestrate_benchmark_submit.sh --countries IN --horizon eos
  orchestrate_benchmark_submit.sh --region-threshold 50 --max 3
EOF
}

if [[ -f "${SLURM_SUBMIT_DIR:-}/cybench/runs/slurm/slurm_common.sh" ]]; then
  SLURM_DIR="${SLURM_SUBMIT_DIR}/cybench/runs/slurm"
elif [[ -n "${REPO_ROOT:-}" && -f "${REPO_ROOT}/cybench/runs/slurm/slurm_common.sh" ]]; then
  SLURM_DIR="${REPO_ROOT}/cybench/runs/slurm"
else
  SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

SUBMIT_BENCHMARK="${SLURM_DIR}/submit_benchmark.sh"
PLAN_PY="${SLURM_DIR}/benchmark_submit_lib.py"

LIST_ONLY=false
DRY_RUN=false
FORCE=false
ALL_COUNTRIES=false
PHASE_MODE="all"
VERSION=3
REGION_THRESHOLD=50
MAX_BATCHES=0
MANIFEST_ROOT="${SLURM_DIR}/manifests"
DATA_DIR=""
COUNTRIES=()
HORIZONS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --list)
      LIST_ONLY=true
      shift
      ;;
    --dry-run|-n)
      DRY_RUN=true
      shift
      ;;
    --countries)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        COUNTRIES+=("$1")
        shift
      done
      ;;
    --horizon|--horizons)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        HORIZONS+=("$1")
        shift
      done
      ;;
    --region-threshold)
      REGION_THRESHOLD=$2
      shift 2
      ;;
    --version)
      VERSION=$2
      shift 2
      ;;
    --phase)
      PHASE_MODE=$2
      shift 2
      ;;
    --force)
      FORCE=true
      shift
      ;;
    --all-countries)
      ALL_COUNTRIES=true
      shift
      ;;
    --max)
      MAX_BATCHES=$2
      shift 2
      ;;
    --manifest-root)
      MANIFEST_ROOT=$2
      shift 2
      ;;
    --data-dir)
      DATA_DIR=$2
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

case "${PHASE_MODE}" in
  screening|walk_forward|all) ;;
  *)
    echo "Invalid --phase: ${PHASE_MODE}" >&2
    exit 1
    ;;
esac

plan_args=(
  poetry run python "${PLAN_PY}"
  --manifest-root "${MANIFEST_ROOT}"
  --version "${VERSION}"
  --region-threshold "${REGION_THRESHOLD}"
)
[[ -n "${DATA_DIR}" ]] && plan_args+=(--data-dir "${DATA_DIR}")
[[ ${#COUNTRIES[@]} -gt 0 ]] && plan_args+=(--countries "${COUNTRIES[@]}")
[[ ${#HORIZONS[@]} -gt 0 ]] && plan_args+=(--horizons "${HORIZONS[@]}")
[[ "${ALL_COUNTRIES}" == true ]] && plan_args+=(--all)
[[ "${FORCE}" == true ]] && plan_args+=(--force)

if [[ "${LIST_ONLY}" == true ]]; then
  "${plan_args[@]}"
  exit 0
fi

mapfile -t PLAN_LINES < <("${plan_args[@]}" --plan-tsv)

if [[ ${#PLAN_LINES[@]} -eq 0 ]]; then
  echo "[DONE] No batches to submit (use --list to inspect)"
  exit 0
fi

echo "[INFO] ${#PLAN_LINES[@]} batch(es) to submit (region threshold=${REGION_THRESHOLD})"
submitted=0

for line in "${PLAN_LINES[@]}"; do
  IFS=$'\t' read -r CC HORIZON BATCH GPU_MODE N_REGIONS <<< "${line}"
  if [[ "${MAX_BATCHES}" -gt 0 && "${submitted}" -ge "${MAX_BATCHES}" ]]; then
    echo "[STOP] --max ${MAX_BATCHES} reached"
    break
  fi

  cmd=(
    "${SUBMIT_BENCHMARK}" "${PHASE_MODE}"
    --horizon "${HORIZON}"
    --batch "${BATCH}"
    --regenerate
    --countries "${CC}"
  )
  if [[ "${GPU_MODE}" == "cpu" ]]; then
    cmd+=(--cpu)
  fi
  if [[ "${DRY_RUN}" == true ]]; then
    cmd+=(--dry-run)
  fi

  echo ""
  echo "=== ${BATCH} | ${CC} | ${HORIZON} | regions=${N_REGIONS} | torch group=${GPU_MODE} ==="
  if [[ "${DRY_RUN}" == true ]]; then
    echo "[DRY-RUN] ${cmd[*]}"
  else
    "${cmd[@]}"
  fi
  submitted=$((submitted + 1))
done

echo ""
echo "[DONE] Submitted ${submitted} batch(es)"
