#!/bin/bash
#
# Screening benchmark: one SLURM array task = one (crop, country, model) job.
#
# 1) Generate the job list (only countries with data on disk):
#      poetry run python cybench/runs/slurm/generate_job_manifest.py
#    Or a subset:
#      poetry run python cybench/runs/slurm/generate_job_manifest.py --countries US FR NL
#
# 2) Submit (from repo root):
#      sbatch cybench/runs/slurm/screening.sh
#
# Resource guide:
#   Tabular (pandas):  --cpus-per-task=8, no GPU, experiment.n_jobs=1 (set in slurm_common.sh)
#   Neural (torch):    GPU via submit_array.sh → -p gpu --gpus=1 (see slurm_common.sh)
#
# Split manifests (do not mix CPU and GPU in one array):
#   awk '$7=="no"'  benchmark_jobs.txt > benchmark_jobs_cpu.txt   # RF, ridge, ...
#   awk '$7=="yes"' benchmark_jobs.txt > benchmark_jobs_gpu.txt   # torch + tab foundation models
#
#SBATCH --job-name=cybench_screen
#SBATCH --output=output/screening/out_%A_%a.txt
#SBATCH --error=output/screening/err_%A_%a.txt
#SBATCH --mem-per-cpu=4G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4-00:00:00
## GPU submits via submit_array.sh override this with --time=2-00:00:00 (partition limit)
##SBATCH --mail-user=michiel.kallenberg@wur.nl
##SBATCH --mail-type=ALL
##SBATCH --array=0-99
#SBATCH --array=0
## GPU partition/request are added by submit_array.sh (-p gpu --gpus=1 on WUR lustre)

set -euo pipefail

# SLURM copies this script to a spool dir; BASH_SOURCE then points there, not the repo.
# Resolve cybench/runs/slurm from the directory where sbatch was invoked (repo root).
if [[ -f "${SLURM_SUBMIT_DIR:-}/cybench/runs/slurm/slurm_common.sh" ]]; then
  export SLURM_DIR="${SLURM_SUBMIT_DIR}/cybench/runs/slurm"
elif [[ -f "${SLURM_SUBMIT_DIR:-}/slurm_common.sh" ]]; then
  export SLURM_DIR="${SLURM_SUBMIT_DIR}"
elif [[ -n "${REPO_ROOT:-}" && -f "${REPO_ROOT}/cybench/runs/slurm/slurm_common.sh" ]]; then
  export SLURM_DIR="${REPO_ROOT}/cybench/runs/slurm"
else
  export SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
source "${SLURM_DIR}/slurm_common.sh"
slurm_setup
mkdir -p output/screening

read_benchmark_job
slurm_validate_env "${MODEL}"
slurm_update_task_job_name screening
echo "Screening | ${CROP}/${COUNTRY} | model=${MODEL} | device=$(device_mode_label) | horizon=${PREDICTION_HORIZON} | batch=${CYBENCH_EXPERIMENT_NAME} | out=${BASELINES_DIR}"

if [[ "${MODEL}" == "twso_bc" ]]; then
  set +e
  twso_skip_reason=$(
    poetry run python -c "
from cybench.models.twso_model import twso_screening_viable
ok, msg = twso_screening_viable(
    '${CROP}', '${COUNTRY}', end_of_sequence='${PREDICTION_HORIZON}'
)
if not ok:
    print(msg)
    raise SystemExit(2)
"
  )
  twso_status=$?
  set -e
  if [[ ${twso_status} -eq 2 ]]; then
    echo "[SKIP] TWSO screening | ${CROP}/${COUNTRY} | horizon=${PREDICTION_HORIZON} — ${twso_skip_reason}"
    exit 0
  fi
  if [[ ${twso_status} -ne 0 ]]; then
    exit "${twso_status}"
  fi
fi

COMMON=(
  "dataset/crop=${CROP}"
  "dataset.country=${COUNTRY}"
  dataset.use_cache=true
  validation=screening
  "experiment.name=${CYBENCH_EXPERIMENT_NAME}"
  experiment.n_repetitions=1
  experiment.seed=42
  "model=${MODEL}"
)

configure_parallelism COMMON
EXTRA=()
configure_hpo_extras EXTRA
append_extra_overrides_file EXTRA

poetry run python cybench/runs/run_experiments.py "${COMMON[@]}" "${EXTRA[@]}"
