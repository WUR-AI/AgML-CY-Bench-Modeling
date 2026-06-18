#!/usr/bin/env bash
#
# Complete missing/failed benchmark jobs for an existing batch (partial rerun).
#
# Usage (from repo root on anunna):
#   cybench/runs/slurm/orchestrate_benchmark_complete.sh --batch baselines_DE_eos_v1 --list
#   cybench/runs/slurm/orchestrate_benchmark_complete.sh --country DE --horizons eos mid --list
#   cybench/runs/slurm/orchestrate_benchmark_complete.sh --batch baselines_DE_eos_v1 --submit --dry-run
#
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: orchestrate_benchmark_complete.sh (--batch NAME | --country CC) [options]

Inspect Hydra output for incomplete screening / walk-forward jobs, optionally
submit a partial manifest (does not rerun successful jobs).

If manifests/<batch>/benchmark_jobs.txt is missing, jobs are taken from the
shared benchmark_jobs.txt (filtered by country) or generated via generate_job_manifest.py.

Options:
  --batch NAME        Hydra experiment.name (e.g. baselines_DE_eos_v1)
  --country CC        Country code (use with --horizons for both eos + mid)
  --horizon H         Single horizon (default: eos); repeat or use --horizons
  --horizons H ...    eos, mid, middle-of-season (default: eos)
  --version N         Batch version when using --country (default: 1)
  --manifest PATH     Explicit job list
  --output-root DIR   Parent of baselines_* (default: lustre output or ../output)
  --baselines-dir DIR Override output dir for one batch
  --data-dir DIR      Override cybench/data for year preflight
  --phase MODE        screening | walk_forward | all (default: all)
  --list              Print status table and exit
  --submit            Write retry manifest and call submit_benchmark.sh
  --dry-run           With --submit: no sbatch
  --cpu               GPU manifest group on CPU partition

Examples:
  orchestrate_benchmark_complete.sh --batch baselines_DE_eos_v1 --list
  orchestrate_benchmark_complete.sh --country DE --horizons eos mid --list
  orchestrate_benchmark_complete.sh --country DE --horizons eos mid --submit --dry-run
EOF
}

if [[ -f "${SLURM_SUBMIT_DIR:-}/cybench/runs/slurm/slurm_common.sh" ]]; then
  SLURM_DIR="${SLURM_SUBMIT_DIR}/cybench/runs/slurm"
elif [[ -n "${REPO_ROOT:-}" && -f "${REPO_ROOT}/cybench/runs/slurm/slurm_common.sh" ]]; then
  SLURM_DIR="${REPO_ROOT}/cybench/runs/slurm"
else
  SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

COMPLETE_PY="${SLURM_DIR}/orchestrate_benchmark_complete.py"
BATCH=""
COUNTRY=""
HORIZONS=(eos)
VERSION=""
MANIFEST=""
BASELINES_DIR=""
OUTPUT_ROOT=""
DATA_DIR=""
PHASE="all"
LIST=false
SUBMIT=false
DRY_RUN=false
FORCE_CPU=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --batch)
      BATCH=$2
      shift 2
      ;;
    --country)
      COUNTRY=$2
      shift 2
      ;;
    --horizon)
      HORIZONS=("$2")
      shift 2
      ;;
    --horizons)
      shift
      HORIZONS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        HORIZONS+=("$1")
        shift
      done
      ;;
    --version)
      VERSION=$2
      shift 2
      ;;
    --manifest)
      MANIFEST=$2
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT=$2
      shift 2
      ;;
    --baselines-dir)
      BASELINES_DIR=$2
      shift 2
      ;;
    --data-dir)
      DATA_DIR=$2
      shift 2
      ;;
    --phase)
      PHASE=$2
      shift 2
      ;;
    --list)
      LIST=true
      shift
      ;;
    --submit)
      SUBMIT=true
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --cpu)
      FORCE_CPU=true
      shift
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

if [[ -z "${BATCH}" && -z "${COUNTRY}" ]]; then
  echo "Provide --batch or --country" >&2
  usage
  exit 1
fi

cmd=(poetry run python "${COMPLETE_PY}" --phase "${PHASE}")
[[ -n "${BATCH}" ]] && cmd+=(--batch "${BATCH}")
[[ -n "${COUNTRY}" ]] && cmd+=(--country "${COUNTRY}")
cmd+=(--horizons "${HORIZONS[@]}")
[[ -n "${VERSION}" ]] && cmd+=(--version "${VERSION}")
[[ -n "${MANIFEST}" ]] && cmd+=(--manifest "${MANIFEST}")
[[ -n "${OUTPUT_ROOT}" ]] && cmd+=(--output-root "${OUTPUT_ROOT}")
[[ -n "${BASELINES_DIR}" ]] && cmd+=(--baselines-dir "${BASELINES_DIR}")
[[ -n "${DATA_DIR}" ]] && cmd+=(--data-dir "${DATA_DIR}")
[[ "${LIST}" == true ]] && cmd+=(--list)
[[ "${SUBMIT}" == true ]] && cmd+=(--submit)
[[ "${DRY_RUN}" == true ]] && cmd+=(--dry-run)
[[ "${FORCE_CPU}" == true ]] && cmd+=(--cpu)

"${cmd[@]}"
