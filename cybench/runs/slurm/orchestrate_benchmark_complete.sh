#!/usr/bin/env bash
#
# Complete missing/failed benchmark jobs for an existing batch (partial rerun).
#
# Usage (from repo root on anunna):
#   cybench/runs/slurm/orchestrate_benchmark_complete.sh --country DE --horizons eos mid --list
#   cybench/runs/slurm/orchestrate_benchmark_complete.sh --all-countries --horizons eos mid --list
#   cybench/runs/slurm/orchestrate_benchmark_complete.sh --all-countries --horizons eos mid --submit --dry-run
#
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: orchestrate_benchmark_complete.sh (--batch NAME | --country CC | --countries ... | --all-countries) [options]

Inspect Hydra output for incomplete screening / walk-forward jobs, optionally
submit a partial manifest (does not rerun successful jobs).

Options:
  --batch NAME        Single Hydra experiment.name (e.g. baselines_DE_eos_v1)
  --country CC        Single country code
  --countries CC ...  Multiple countries (e.g. DE FR NL)
  --all-countries     All countries with output dirs, manifest rows, or yield data
  --horizon H         Single horizon (default: eos); repeat or use --horizons
  --horizons H ...    eos, mid, early-season, middle-of-season (default: eos)
  --version N         Batch version suffix (default: 3)
  --max N             Process at most N batch×horizon targets (0 = unlimited)
  --manifest PATH     Explicit job list
  --model MODEL       Limit to model slug (repeatable, e.g. --model lpjml_bc or twso_bc)
  --force-rerun       Include complete jobs in retry manifest (use with --resume)
  --repetitions N     Walk-forward: total target seeds (42..42+N-1)
  --resume            Walk-forward: append missing seeds into latest run dirs
  --output-root DIR   Parent of baselines_* (default: lustre output or ../output)
  --baselines-dir DIR Override output dir for one batch
  --data-dir DIR      Override cybench/data for year preflight
  --phase MODE        screening | walk_forward | all (default: all)
  --list              Print status table and exit
  --submit            Write retry manifest and call submit_benchmark.sh
  --dry-run           With --submit: no sbatch
  --cpu               Force torch/TabPFN group to CPU partition
  --force-gpu         Use gpu partition even when regions < threshold
  --region-threshold N  gpu when country has >= N regions (default: 50)

Examples:
  orchestrate_benchmark_complete.sh --country DE --horizons eos mid --list
  orchestrate_benchmark_complete.sh --all-countries --horizons eos mid --list
  orchestrate_benchmark_complete.sh --countries DE FR NL --horizon eos --submit --dry-run
  orchestrate_benchmark_complete.sh --all-countries --horizons eos mid --max 5 --submit
  orchestrate_benchmark_complete.sh --all-countries --horizon eos --model twso_bc \\
    --phase walk_forward --force-rerun --submit --dry-run
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
COUNTRIES=()
ALL_COUNTRIES=false
HORIZONS=(eos)
VERSION=""
MAX=""
MANIFEST=""
MODELS=()
FORCE_RERUN=false
WF_REPETITIONS=""
WF_RESUME=false
BASELINES_DIR=""
OUTPUT_ROOT=""
DATA_DIR=""
PHASE="all"
LIST=false
SUBMIT=false
DRY_RUN=false
FORCE_CPU=false
FORCE_GPU=false
REGION_THRESHOLD=""

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
    --countries)
      shift
      COUNTRIES=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        COUNTRIES+=("$1")
        shift
      done
      ;;
    --all-countries)
      ALL_COUNTRIES=true
      shift
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
    --max)
      MAX=$2
      shift 2
      ;;
    --manifest)
      MANIFEST=$2
      shift 2
      ;;
    --model)
      MODELS+=("$2")
      shift 2
      ;;
    --force-rerun)
      FORCE_RERUN=true
      shift
      ;;
    --repetitions)
      WF_REPETITIONS=$2
      shift 2
      ;;
    --resume)
      WF_RESUME=true
      shift
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
    --force-gpu)
      FORCE_GPU=true
      shift
      ;;
    --region-threshold)
      REGION_THRESHOLD=$2
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

if [[ -z "${BATCH}" && -z "${COUNTRY}" && ${#COUNTRIES[@]} -eq 0 && "${ALL_COUNTRIES}" != true ]]; then
  echo "Provide --batch, --country, --countries, or --all-countries" >&2
  usage
  exit 1
fi

cmd=(poetry run python "${COMPLETE_PY}" --phase "${PHASE}")
[[ -n "${BATCH}" ]] && cmd+=(--batch "${BATCH}")
[[ -n "${COUNTRY}" ]] && cmd+=(--country "${COUNTRY}")
[[ ${#COUNTRIES[@]} -gt 0 ]] && cmd+=(--countries "${COUNTRIES[@]}")
[[ "${ALL_COUNTRIES}" == true ]] && cmd+=(--all-countries)
cmd+=(--horizons "${HORIZONS[@]}")
[[ -n "${VERSION}" ]] && cmd+=(--version "${VERSION}")
[[ -n "${MAX}" ]] && cmd+=(--max "${MAX}")
[[ -n "${MANIFEST}" ]] && cmd+=(--manifest "${MANIFEST}")
for _model in "${MODELS[@]}"; do
  cmd+=(--model "${_model}")
done
[[ "${FORCE_RERUN}" == true ]] && cmd+=(--force-rerun)
[[ -n "${WF_REPETITIONS}" ]] && cmd+=(--repetitions "${WF_REPETITIONS}")
[[ "${WF_RESUME}" == true ]] && cmd+=(--resume)
[[ -n "${OUTPUT_ROOT}" ]] && cmd+=(--output-root "${OUTPUT_ROOT}")
[[ -n "${BASELINES_DIR}" ]] && cmd+=(--baselines-dir "${BASELINES_DIR}")
[[ -n "${DATA_DIR}" ]] && cmd+=(--data-dir "${DATA_DIR}")
[[ "${LIST}" == true ]] && cmd+=(--list)
[[ "${SUBMIT}" == true ]] && cmd+=(--submit)
[[ "${DRY_RUN}" == true ]] && cmd+=(--dry-run)
[[ "${FORCE_CPU}" == true ]] && cmd+=(--cpu)
[[ "${FORCE_GPU}" == true ]] && cmd+=(--force-gpu)
[[ -n "${REGION_THRESHOLD}" ]] && cmd+=(--region-threshold "${REGION_THRESHOLD}")

"${cmd[@]}"
