#!/usr/bin/env bash
#
# Regenerate full per-country benchmark_jobs.txt from generate_job_manifest.py.
#
# Use after orchestrate_benchmark_complete.sh --submit overwrote a working manifest
# with an "incomplete jobs only" retry list.
#
# Usage (from repo root on anunna):
#   cybench/runs/slurm/refresh_batch_manifests.sh
#   cybench/runs/slurm/refresh_batch_manifests.sh --horizon qtr --version 2
#   cybench/runs/slurm/refresh_batch_manifests.sh --countries BR FR --horizon qtr --version 2
#   cybench/runs/slurm/refresh_batch_manifests.sh --dry-run
#
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: refresh_batch_manifests.sh [options]

Regenerate manifests/baselines_<CC>_<hz>_vN/benchmark_jobs.txt for batches that
exist under the output root (or for explicit --countries).

Options:
  --horizon H       eos | mid | qtr | early-season (default: qtr)
  --version N       Batch version suffix (default: 3)
  --countries CC..  Only these countries (default: all matching batch dirs)
  --models FILE     Model catalogue (default: models.txt)
  --output-root DIR Parent of baselines_* (default: $CYBENCH_OUTPUT_ROOT or lustre)
  --data-dir DIR    Passed to generate_job_manifest.py
  --backup          Keep timestamped copy of existing benchmark_jobs.txt
  --dry-run         Print commands only
  -h, --help        This help

Examples:
  refresh_batch_manifests.sh --horizon qtr --version 2
  refresh_batch_manifests.sh --countries BR BG CN FR --horizon qtr --version 2 --backup
EOF
}

if [[ -f "${SLURM_SUBMIT_DIR:-}/cybench/runs/slurm/slurm_common.sh" ]]; then
  SLURM_DIR="${SLURM_SUBMIT_DIR}/cybench/runs/slurm"
elif [[ -n "${REPO_ROOT:-}" && -f "${REPO_ROOT}/cybench/runs/slurm/slurm_common.sh" ]]; then
  SLURM_DIR="${REPO_ROOT}/cybench/runs/slurm"
else
  SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
# shellcheck source=cybench/runs/slurm/slurm_common.sh
source "${SLURM_DIR}/slurm_common.sh"

REPO_ROOT="${REPO_ROOT:-$(cd "${SLURM_DIR}/../../.." && pwd)}"
GENERATE_PY="${SLURM_DIR}/generate_job_manifest.py"
MODELS_FILE="${SLURM_DIR}/models.txt"

HORIZON_KEY="qtr"
VERSION="3"
COUNTRIES=()
OUTPUT_ROOT="${CYBENCH_OUTPUT_ROOT:-}"
DATA_DIR=""
BACKUP=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --horizon)
      HORIZON_KEY=$2
      shift 2
      ;;
    --version)
      VERSION=$2
      shift 2
      ;;
    --countries)
      shift
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        COUNTRIES+=("$(printf '%s' "$1" | tr '[:lower:]' '[:upper:]')")
        shift
      done
      ;;
    --output-root)
      OUTPUT_ROOT=$2
      shift 2
      ;;
    --data-dir)
      DATA_DIR=$2
      shift 2
      ;;
    --models)
      MODELS_FILE=$2
      shift 2
      ;;
    --backup)
      BACKUP=true
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

case "${HORIZON_KEY}" in
  eos) BATCH_HZ="eos"; SLURM_HORIZON="eos" ;;
  mid | middle-of-season | middle_of_season | mid-season | mid_season)
    BATCH_HZ="mid"
    SLURM_HORIZON="middle-of-season"
    ;;
  qtr | quarter-of-season | quarter_of_season | quarter-season | quarter_season)
    BATCH_HZ="qtr"
    SLURM_HORIZON="quarter-of-season"
    ;;
  early | early-season | early_season)
    BATCH_HZ="early"
    SLURM_HORIZON="early-season"
    ;;
  *)
    echo "Unknown horizon: ${HORIZON_KEY}" >&2
    exit 1
    ;;
esac

if [[ -z "${OUTPUT_ROOT}" ]]; then
  if [[ -d /lustre/backup/SHARED/AIN/agml/output ]]; then
    OUTPUT_ROOT=/lustre/backup/SHARED/AIN/agml/output
  else
    OUTPUT_ROOT="${REPO_ROOT}/../output"
  fi
fi

if [[ ! -d "${OUTPUT_ROOT}" ]]; then
  echo "Output root not found: ${OUTPUT_ROOT}" >&2
  exit 1
fi

parse_country_from_batch_dir() {
  local name=$1
  if [[ "${name}" =~ ^baselines_([A-Za-z]{2})_${BATCH_HZ}_v${VERSION}$ ]]; then
    printf '%s' "${BASH_REMATCH[1]}" | tr '[:lower:]' '[:upper:]'
    return 0
  fi
  return 1
}

declare -a TARGET_COUNTRIES=()
if [[ ${#COUNTRIES[@]} -gt 0 ]]; then
  TARGET_COUNTRIES=("${COUNTRIES[@]}")
else
  shopt -s nullglob
  declare -A seen=()
  for d in "${OUTPUT_ROOT}"/baselines_*_"${BATCH_HZ}"_v"${VERSION}"; do
    [[ -d "${d}" ]] || continue
    cc=$(parse_country_from_batch_dir "$(basename "${d}")") || continue
    if [[ -z "${seen[${cc}]:-}" ]]; then
      seen["${cc}"]=1
      TARGET_COUNTRIES+=("${cc}")
    fi
  done
  shopt -u nullglob
  if [[ ${#TARGET_COUNTRIES[@]} -eq 0 ]]; then
    echo "No baselines_*_${BATCH_HZ}_v${VERSION} directories under ${OUTPUT_ROOT}" >&2
    exit 1
  fi
  IFS=$'\n' TARGET_COUNTRIES=($(printf '%s\n' "${TARGET_COUNTRIES[@]}" | sort))
  unset IFS
fi

run_generate() {
  local cc=$1
  local batch="baselines_${cc}_${BATCH_HZ}_v${VERSION}"
  local manifest_dir
  manifest_dir="$(manifest_batch_dir "${SLURM_DIR}" "${batch}")"
  local out="${manifest_dir}/benchmark_jobs.txt"
  mkdir -p "${manifest_dir}"

  if [[ -f "${out}" && "${BACKUP}" == true ]]; then
    local stamp
    stamp=$(date -u +%Y%m%dT%H%M%SZ)
    local backup="${out}.bak.${stamp}"
    if [[ "${DRY_RUN}" == true ]]; then
      echo "[DRY-RUN] cp ${out} ${backup}"
    else
      cp "${out}" "${backup}"
      echo "[backup] ${backup}"
    fi
  fi

  local cmd=(
    poetry run python "${GENERATE_PY}"
    --countries "${cc}"
    --horizon "${SLURM_HORIZON}"
    --models "${MODELS_FILE}"
    -o "${out}"
  )
  if [[ -n "${DATA_DIR}" ]]; then
    cmd+=(--data-dir "${DATA_DIR}")
  fi

  echo "Generating ${cc} -> ${out}"
  if [[ "${DRY_RUN}" == true ]]; then
    printf '  '; printf '%q ' "${cmd[@]}"; printf '\n'
  else
    (cd "${REPO_ROOT}" && "${cmd[@]}")
  fi
}

echo "Refresh manifests | horizon=${SLURM_HORIZON} (${BATCH_HZ}) | version=${VERSION} | output=${OUTPUT_ROOT}"
for cc in "${TARGET_COUNTRIES[@]}"; do
  run_generate "${cc}"
done
echo "[DONE] Refreshed ${#TARGET_COUNTRIES[@]} manifest(s)"
