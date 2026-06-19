#!/bin/bash
#
# Collect walk-forward results for one country × horizon (SLURM array task).
#
# Submit via submit_collect.sh (from repo root).
#
#SBATCH --job-name=cb_collect
#SBATCH --output=output/collect/out_%A_%a.txt
#SBATCH --error=output/collect/err_%A_%a.txt
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --array=0

set -euo pipefail

if [[ -f "${SLURM_SUBMIT_DIR:-}/cybench/runs/slurm/slurm_common.sh" ]]; then
  export SLURM_DIR="${SLURM_SUBMIT_DIR}/cybench/runs/slurm"
elif [[ -n "${REPO_ROOT:-}" && -f "${REPO_ROOT}/cybench/runs/slurm/slurm_common.sh" ]]; then
  export SLURM_DIR="${REPO_ROOT}/cybench/runs/slurm"
else
  export SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
source "${SLURM_DIR}/slurm_common.sh"
slurm_setup
mkdir -p output/collect

JOB_MANIFEST="${JOB_MANIFEST:-${SLURM_DIR}/collect_jobs.txt}"
line=$(awk '!/^#/ && NF >= 3 {print}' "${JOB_MANIFEST}" | awk "NR == ${SLURM_ARRAY_TASK_ID}+1")
if [[ -z "${line}" ]]; then
  echo "No collect job for array task ${SLURM_ARRAY_TASK_ID} in ${JOB_MANIFEST}" >&2
  exit 1
fi
read -r COLLECT_COUNTRY COLLECT_HORIZON COLLECT_PLOT <<< "${line}"

job_id=$(slurm_task_job_id)
if [[ -n "${job_id}" ]]; then
  name="cb_col_${COLLECT_COUNTRY}_${COLLECT_HORIZON}"
  scontrol update "JobId=${job_id}" "JobName=${name:0:63}" 2>/dev/null || true
fi

echo "Collect | ${COLLECT_COUNTRY} | horizon=${COLLECT_HORIZON} | plot=${COLLECT_PLOT}"

cmd=(
  poetry run python cybench/runs/analysis/orchestrate_dashboard_publish.py
  --country "${COLLECT_COUNTRY}"
  --horizon "${COLLECT_HORIZON}"
  --mode all-available
  --stages collect
  --force collect
)
if [[ "${COLLECT_PLOT}" != "yes" ]]; then
  cmd+=(--no-plot)
fi

"${cmd[@]}"
