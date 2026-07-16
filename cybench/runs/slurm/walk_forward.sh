#!/bin/bash
#
# Walk-forward: one array task = one (crop, country, model) [× seed for GPU manifest].
# Auto-discovers the latest screening run under CYBENCH_EXPERIMENT_NAME (default: baselines):
#   ../output/<batch>/<crop>_<country>_<model>_screening_<horizon>_<timestamp>/<test_years>/optimal_model.yaml
#
# GPU manifest rows with an 8th column (seed) run one seed per SLURM task.
# Large countries (≥350 regions): optional 9th column (forecast year) for parallel origins.
# CPU/naive manifests bundle seeds in one task (experiment.n_repetitions).
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
# Single-threaded BLAS during torch walk-forward fits (override to use more CPUs).
export CYBENCH_TORCH_THREADS="${CYBENCH_TORCH_THREADS:-1}"
mkdir -p output/walk_forward

read_benchmark_job
slurm_validate_env "${MODEL}"
slurm_update_task_job_name walk_forward
if ! FROZEN_DIR=$(find_frozen_screening_dir "${CROP}" "${COUNTRY}" "${MODEL}"); then
  echo "[SKIP] Walk-forward | ${CROP}/${COUNTRY} | model=${MODEL} — no successful screening artifact"
  exit 0
fi

WF_REPETITIONS="${WF_REPETITIONS:-1}"
WF_BASE_SEED="${WF_BASE_SEED:-42}"
WF_RESUME="${WF_RESUME:-no}"
WF_RUN_DIR=""
WF_START_SEED="${WF_BASE_SEED}"
WF_RUN_REPS="${WF_REPETITIONS}"

if [[ -n "${WF_SEED:-}" ]]; then
  plan_walk_forward_single_seed "${CROP}" "${COUNTRY}" "${MODEL}" "${WF_SEED}" "${WF_ORIGIN:-}"
  wf_plan_rc=$?
  case "${wf_plan_rc}" in
    0) ;;
    1) exit 0 ;; # seed already present
    *) exit 1 ;; # timeout / planning error — do not mask as success
  esac
elif ! plan_walk_forward_seeds "${CROP}" "${COUNTRY}" "${MODEL}"; then
  exit 0
fi

resume_note=""
if [[ -n "${WF_RUN_DIR}" ]]; then
  resume_note=" | resume=${WF_RUN_DIR}"
fi
seed_note=" | seed=${WF_START_SEED}"
if [[ -n "${WF_SEED:-}" ]]; then
  seed_note=" | task_seed=${WF_SEED}"
  if [[ -n "${WF_ORIGIN:-}" ]]; then
    seed_note+=" | origin=${WF_ORIGIN}"
  fi
elif [[ "${WF_RUN_REPS}" != "1" ]]; then
  seed_note=" | seeds=${WF_START_SEED}..$((WF_START_SEED + WF_RUN_REPS - 1))"
fi
echo "Walk-forward | ${CROP}/${COUNTRY} | model=${MODEL} | device=$(device_mode_label) | horizon=${PREDICTION_HORIZON} | batch=${CYBENCH_EXPERIMENT_NAME} | target_repetitions=${WF_REPETITIONS}${seed_note}${resume_note} | frozen=${FROZEN_DIR}"

COMMON=(
  "dataset/crop=${CROP}"
  "dataset.country=${COUNTRY}"
  dataset.use_cache=false
  validation=walk_forward
  "validation.frozen_screening_dir=${FROZEN_DIR}"
  "experiment.name=${CYBENCH_EXPERIMENT_NAME}"
  "experiment.n_repetitions=${WF_RUN_REPS}"
  "experiment.seed=${WF_START_SEED}"
  "model=${MODEL}"
  store.model=true
)

configure_parallelism COMMON
EXTRA=()
if [[ -n "${WF_ORIGIN:-}" ]]; then
  EXTRA+=("validation.test_years=[${WF_ORIGIN}]")
fi
if [[ -n "${WF_RUN_DIR}" ]]; then
  EXTRA+=("hydra.run.dir=${WF_RUN_DIR}")
fi
if [[ "${FEATURE_DESIGN}" == "yes" && "${FRAMEWORK}" == "pandas" ]]; then
  EXTRA+=(+feature_selection=mrmr)
fi
append_extra_overrides_file EXTRA

poetry run python cybench/runs/run_experiments.py "${COMMON[@]}" "${EXTRA[@]}"
