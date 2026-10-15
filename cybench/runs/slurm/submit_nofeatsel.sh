#!/usr/bin/env bash
#
# Walk-forward for tabular foundation models (tabpfn, tabicl, tabdpt) without mRMR.
# Model hyperparameters are the ones already frozen in a v4 (or --version) screening
# batch. Feature selection is not re-tuned and is not applied.
#
# Results are written to baselines_{CC}_{hz}_nofeatsel, next to the v4 batch but
# not inside it. v4 folders are only read.
#
# From the cluster checkout (repo parent contains output/, or set CYBENCH_OUTPUT_ROOT):
#   cybench/runs/slurm/submit_nofeatsel.sh --horizon eos --countries DE NL --dry-run
#   cybench/runs/slurm/submit_nofeatsel.sh --horizon eos --repetitions 5
#   cybench/runs/slurm/submit_nofeatsel.sh --horizon middle-of-season --countries FR
#
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: submit_nofeatsel.sh [options]

Walk-forward only. Does not rerun screening or hyperparameter search.
Reads optimal_model.yaml from baselines_{CC}_{hz}_v{version}.
Writes predictions to baselines_{CC}_{hz}_nofeatsel.

Options:
  --horizon H       eos | middle-of-season | quarter-of-season | early-season
                    (default: eos). Repeat for several horizons.
  --countries C...  Country codes. Default: every country that already has a
                    v{version} batch for that horizon.
  --version N       Screening batch version to read (default: 4)
  --repetitions N   Walk-forward seeds from 42 (default: 1). Use 5 to match
                    the 5-seed paper runs.
  --array RANGE     SLURM array range (default: full manifest)
  --cpu             GPU models on the CPU partition (slow)
  --dry-run         Build manifests and print sbatch plans; do not submit
  -h, --help

Environment:
  CYBENCH_OUTPUT_ROOT   Directory that holds baselines_*_v4 (default: ../output,
                        or /lustre/backup/SHARED/AIN/agml/output when that exists)
EOF
}

if [[ -f "${SLURM_SUBMIT_DIR:-}/cybench/runs/slurm/slurm_common.sh" ]]; then
  SLURM_DIR="${SLURM_SUBMIT_DIR}/cybench/runs/slurm"
elif [[ -n "${REPO_ROOT:-}" && -f "${REPO_ROOT}/cybench/runs/slurm/slurm_common.sh" ]]; then
  SLURM_DIR="${REPO_ROOT}/cybench/runs/slurm"
else
  SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
REPO_ROOT="$(cd "${SLURM_DIR}/../../.." && pwd)"

# shellcheck source=/dev/null
source "${SLURM_DIR}/slurm_common.sh"

HORIZONS=()
COUNTRIES=()
VERSION=4
WF_REPETITIONS="${WF_REPETITIONS:-1}"
ARRAY_ARG=()
FORCE_CPU=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --horizon)
      HORIZONS+=("$2")
      shift 2
      ;;
    --countries)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        COUNTRIES+=("$1")
        shift
      done
      ;;
    --version)
      VERSION=$2
      shift 2
      ;;
    --repetitions)
      WF_REPETITIONS=$2
      shift 2
      ;;
    --array)
      ARRAY_ARG=(--array "$2")
      shift 2
      ;;
    --cpu)
      FORCE_CPU=true
      shift
      ;;
    --dry-run|-n)
      DRY_RUN=true
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

if [[ ${#HORIZONS[@]} -eq 0 ]]; then
  HORIZONS=(eos)
fi
validate_wf_repetitions "${WF_REPETITIONS}"

resolve_output_root() {
  if [[ -n "${CYBENCH_OUTPUT_ROOT:-}" ]]; then
    echo "${CYBENCH_OUTPUT_ROOT}"
    return
  fi
  if [[ -d /lustre/backup/SHARED/AIN/agml/output ]]; then
    echo /lustre/backup/SHARED/AIN/agml/output
    return
  fi
  if [[ -d "${REPO_ROOT}/../output" ]]; then
    (cd "${REPO_ROOT}/../output" && pwd)
    return
  fi
  echo "${REPO_ROOT}/../output"
}

OUTPUT_ROOT="$(resolve_output_root)"
if [[ ! -d "${OUTPUT_ROOT}" ]]; then
  echo "Output root not found: ${OUTPUT_ROOT}" >&2
  echo "Set CYBENCH_OUTPUT_ROOT to the directory that contains baselines_*_v${VERSION}." >&2
  exit 1
fi
OUTPUT_ROOT="$(cd "${OUTPUT_ROOT}" && pwd)"

MODELS_FILE="$(mktemp)"
DISCOVER_PY=""
trap 'rm -f "${MODELS_FILE}" ${DISCOVER_PY:+"${DISCOVER_PY}"}' EXIT
grep -E '^(tabpfn|tabicl|tabdpt)[[:space:]]' "${SLURM_DIR}/models.txt" > "${MODELS_FILE}"
if [[ "$(wc -l < "${MODELS_FILE}")" -ne 3 ]]; then
  echo "Expected tabpfn, tabicl, and tabdpt in ${SLURM_DIR}/models.txt" >&2
  exit 1
fi

DISCOVER_PY="$(mktemp)"
cat > "${DISCOVER_PY}" <<'PY'
import sys
from pathlib import Path

from cybench.runs.slurm.benchmark_submit_lib import discover_nofeatsel_sources

output_root = Path(sys.argv[1])
horizon = sys.argv[2]
version = int(sys.argv[3])
countries = sys.argv[4:] or None
plans = discover_nofeatsel_sources(
    output_root, horizon=horizon, version=version, countries=countries
)
found = {cc for cc, _path, _name in plans}
if countries:
    missing = [cc.upper() for cc in countries if cc.upper() not in found]
    if missing:
        print(
            "No baselines_*_v"
            + str(version)
            + " screening batch for horizon "
            + horizon
            + ": "
            + " ".join(missing),
            file=sys.stderr,
        )
        raise SystemExit(1)
for cc, path, name in plans:
    print(f"{cc}\t{path}\t{name}")
PY

submitted=0
for horizon in "${HORIZONS[@]}"; do
  discover_args=("${OUTPUT_ROOT}" "${horizon}" "${VERSION}")
  if [[ ${#COUNTRIES[@]} -gt 0 ]]; then
    discover_args+=("${COUNTRIES[@]}")
  fi
  plan_file="$(mktemp)"
  if ! poetry run python "${DISCOVER_PY}" "${discover_args[@]}" > "${plan_file}"; then
    rm -f "${plan_file}"
    exit 1
  fi
  mapfile -t plans < "${plan_file}"
  rm -f "${plan_file}"

  if [[ ${#plans[@]} -eq 0 ]]; then
    echo "No v${VERSION} screening batches for horizon ${horizon} under ${OUTPUT_ROOT}" >&2
    exit 1
  fi

  for row in "${plans[@]}"; do
    IFS=$'\t' read -r cc screening_dir out_batch <<< "${row}"
    if ! find "${screening_dir}" -name optimal_model.yaml -print -quit | grep -q .; then
      echo "[SKIP] ${cc} ${horizon}: no optimal_model.yaml under ${screening_dir}" >&2
      continue
    fi

    manifest_dir="$(manifest_batch_dir "${SLURM_DIR}" "${out_batch}")"
    mkdir -p "${manifest_dir}"
    manifest="${manifest_dir}/benchmark_jobs_gpu.txt"
    poetry run python "${SLURM_DIR}/generate_job_manifest.py" \
      --crops maize wheat \
      --countries "${cc}" \
      --models "${MODELS_FILE}" \
      --horizon "${horizon}" \
      -o "${manifest}"

    n_jobs=$(awk '!/^#/ && NF >= 7 { n++ } END { print n + 0 }' "${manifest}")
    out_dir="${OUTPUT_ROOT}/${out_batch}"
    echo "nofeatsel | ${cc} | horizon=${horizon} | jobs=${n_jobs} | read=${screening_dir} | write=${out_dir}"
    if [[ "${n_jobs}" -lt 1 ]]; then
      echo "[SKIP] ${cc} ${horizon}: manifest has no crop/country rows (data missing?)" >&2
      continue
    fi

    mkdir -p "${out_dir}"
    # Absolute Hydra dir so results stay in the nofeatsel batch even when
    # the repo's ../output is not the screening tree.
    overrides="${manifest_dir}/hydra_nofeatsel_overrides.txt"
    cat > "${overrides}" <<EOF
hydra.run.dir=${out_dir}/\${run.name}
hydra.sweep.dir=${out_dir}
experiment.results_dir=${out_dir}
EOF
    export CYBENCH_EXTRA_OVERRIDES_FILE="${overrides}"
    export CYBENCH_APPLY_FEATURE_SELECTION=no
    export CYBENCH_SCREENING_BASELINES_DIR="${screening_dir}"
    export CYBENCH_EXPERIMENT_NAME="${out_batch}"
    export CYBENCH_BASELINES_DIR="${out_dir}"
    export PREDICTION_HORIZON="${horizon}"

    cmd=(
      "${SLURM_DIR}/submit_array.sh" walk_forward "${manifest}"
      --batch "${out_batch}"
      --group gpu
      --repetitions "${WF_REPETITIONS}"
    )
    if [[ ${#ARRAY_ARG[@]} -gt 0 ]]; then
      cmd+=("${ARRAY_ARG[@]}")
    fi
    if [[ "${FORCE_CPU}" == true ]]; then
      cmd+=(--cpu)
    fi
    if [[ "${DRY_RUN}" == true ]]; then
      printf '  dry-run:'
      printf ' %q' "${cmd[@]}"
      printf '\n'
      continue
    fi
    "${cmd[@]}"
    submitted=$((submitted + 1))
  done
done

if [[ "${DRY_RUN}" == true ]]; then
  echo "[DONE] dry-run. v${VERSION} screening was not modified."
else
  echo "[DONE] submitted ${submitted} nofeatsel walk-forward array(s)."
fi
