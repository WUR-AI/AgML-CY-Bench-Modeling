#!/bin/bash
#
# Walk-forward: one array task = one (crop, country, model), using frozen screening artifacts.
# Auto-discovers the latest screening run (Hydra: ../output/baselines/ from repo root):
#   ../output/baselines/<crop>_<country>_<model>_screening_<horizon>_<timestamp>/<test_years>/optimal_model.yaml
#
# Submit after screening jobs finished:
#   sbatch cybench/runs/slurm/walk_forward.sh
#
#SBATCH --job-name=cybench_wf
#SBATCH --output=output/walk_forward/out_%A_%a.txt
#SBATCH --error=output/walk_forward/err_%A_%a.txt
#SBATCH --mem-per-cpu=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
##SBATCH --array=0-99
#SBATCH --array=0
##SBATCH --gres=gpu:1

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
FROZEN_DIR=$(find_frozen_screening_dir "${CROP}" "${COUNTRY}" "${MODEL}")
echo "Walk-forward | ${CROP}/${COUNTRY} | model=${MODEL} | framework=${FRAMEWORK} | horizon=${PREDICTION_HORIZON} | frozen=${FROZEN_DIR}"

COMMON=(
  "dataset/crop=${CROP}"
  "dataset.country=${COUNTRY}"
  dataset.use_cache=true
  validation=walk_forward
  "validation.frozen_screening_dir=${FROZEN_DIR}"
  experiment.name=baselines
  experiment.n_repetitions=1
  experiment.seed=42
  "model=${MODEL}"
)

if [[ "${FRAMEWORK}" == "pandas" ]]; then
  COMMON+=(dataset.framework=pandas dataset/temporal=feature_design +feature_selection=mrmr)
  COMMON+=(experiment.n_jobs=1)
  if [[ "${NEEDS_GPU}" == "yes" ]]; then
    COMMON+=(model.device=cuda model.allow_cpu_fallback=false)
  else
    COMMON+=(experiment.device=cpu)
  fi
else
  COMMON+=(dataset.framework=torch experiment.device=cuda experiment.n_jobs=1)
fi

poetry run python cybench/runs/run_experiments.py "${COMMON[@]}"
