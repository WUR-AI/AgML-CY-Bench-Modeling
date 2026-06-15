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
#   Neural (torch):    --gres=gpu:1, --cpus-per-task=4, experiment.n_jobs=1
#
# Split manifests (do not mix CPU and GPU in one array):
#   awk '$7=="no"'  benchmark_jobs.txt > benchmark_jobs_cpu.txt   # RF, ridge, ...
#   awk '$7=="yes"' benchmark_jobs.txt > benchmark_jobs_gpu.txt   # torch + tabpfn
#
#SBATCH --job-name=cybench_screen
#SBATCH --output=output/screening/out_%A_%a.txt
#SBATCH --error=output/screening/err_%A_%a.txt
#SBATCH --mem-per-cpu=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4-00:00:00
##SBATCH --mail-user=michiel.kallenberg@wur.nl
##SBATCH --mail-type=ALL
##SBATCH --array=0-99
#SBATCH --array=0
## Torch jobs — enable GPU:
##SBATCH --gres=gpu:1
##SBATCH --partition=gpu

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/slurm_common.sh"
slurm_setup
mkdir -p output/screening

read_benchmark_job
echo "Screening | ${CROP}/${COUNTRY} | model=${MODEL} | framework=${FRAMEWORK}"

COMMON=(
  "dataset/crop=${CROP}"
  "dataset.country=${COUNTRY}"
  dataset.use_cache=true
  validation=screening
  experiment.name=baselines
  experiment.n_repetitions=1
  experiment.seed=42
  "model=${MODEL}"
)

configure_parallelism COMMON
EXTRA=()
configure_hpo_extras EXTRA

poetry run python cybench/runs/run_experiments.py "${COMMON[@]}" "${EXTRA[@]}"
