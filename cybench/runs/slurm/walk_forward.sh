#!/bin/bash
#
# Walk-forward: one array task = one (crop, country, model), using frozen screening artifacts.
# Auto-discovers the latest screening run under CYBENCH_EXPERIMENT_NAME (default: baselines):
#   ../output/<batch>/<crop>_<country>_<model>_screening_<horizon>_<timestamp>/<test_years>/optimal_model.yaml
#
# Submit after screening jobs finished:
#   sbatch cybench/runs/slurm/walk_forward.sh
#
#SBATCH --job-name=cybench_wf
#SBATCH --output=output/walk_forward/out_%A_%a.txt
#SBATCH --error=output/walk_forward/err_%A_%a.txt
#SBATCH --mem-per-cpu=4G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
##SBATCH --array=0-99
#SBATCH --array=0
## GPU: added by submit_array.sh when using benchmark_jobs_gpu.txt

set -euo pipefail

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
mkdir -p output/walk_forward

read_benchmark_job
slurm_update_task_job_name walk_forward
FROZEN_DIR=$(find_frozen_screening_dir "${CROP}" "${COUNTRY}" "${MODEL}")
echo "Walk-forward | ${CROP}/${COUNTRY} | model=${MODEL} | device=$(device_mode_label) | horizon=${PREDICTION_HORIZON} | batch=${CYBENCH_EXPERIMENT_NAME} | frozen=${FROZEN_DIR}"

COMMON=(
  "dataset/crop=${CROP}"
  "dataset.country=${COUNTRY}"
  dataset.use_cache=true
  validation=walk_forward
  "validation.frozen_screening_dir=${FROZEN_DIR}"
  "experiment.name=${CYBENCH_EXPERIMENT_NAME}"
  experiment.n_repetitions=1
  experiment.seed=42
  "model=${MODEL}"
)

configure_parallelism COMMON
EXTRA=()
if [[ "${FEATURE_DESIGN}" == "yes" && "${FRAMEWORK}" == "pandas" ]]; then
  EXTRA+=(+feature_selection=mrmr)
fi

poetry run python cybench/runs/run_experiments.py "${COMMON[@]}" "${EXTRA[@]}"
