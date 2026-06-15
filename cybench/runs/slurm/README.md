# SLURM benchmark jobs (WUR lustre)

**screening** + **walk-forward** via `run_experiments.py`.

## Layout

| File | Purpose |
|------|---------|
| `models.txt` | Model catalogue (`needs_gpu=yes` for torch + TabPFN) |
| `generate_job_manifest.py` | Build full `crop × country × model` job list |
| `benchmark_jobs.txt` | **Array manifest** (one row = one SLURM task) |
| `benchmark_jobs.example.txt` | Small test subset |
| `slurm_common.sh` | Modules, paths, HPO/CPU helpers |
| `screening.sh` | Phase A: HPO + held-out test |
| `walk_forward.sh` | Phase B: rolling forecasts (auto-finds screening artifacts) |

## 1. Generate the job list

All crops/countries that have data under `cybench/data/<crop>/<country>/`:

```bash
cd /path/to/your/cybench-clone   # any checkout name works
poetry run python cybench/runs/slurm/generate_job_manifest.py
```

Subset (e.g. pilot before full benchmark):

```bash
poetry run python cybench/runs/slurm/generate_job_manifest.py --countries US FR NL DE
```

With ~10 models and ~40 countries × 2 crops, the full manifest can be **hundreds of jobs**. Use `--array` ranges or split manifests:

```bash
# CPU-only manifest (sklearn / boosting on feature_design)
awk '$7=="no"' cybench/runs/slurm/benchmark_jobs.txt > cybench/runs/slurm/benchmark_jobs_cpu.txt
# GPU manifest (torch + TabPFN — pandas on GPU)
awk '$7=="yes"' cybench/runs/slurm/benchmark_jobs.txt > cybench/runs/slurm/benchmark_jobs_gpu.txt
```

## 2. Submit screening

```bash
mkdir -p output/screening

# Edit #SBATCH --array=0-N  (N = number of lines in manifest minus 1)
# Tabular jobs:
JOB_MANIFEST=cybench/runs/slurm/benchmark_jobs_cpu.txt sbatch cybench/runs/slurm/screening.sh

# Neural jobs (uncomment #SBATCH --gres=gpu:1 in screening.sh):
JOB_MANIFEST=cybench/runs/slurm/benchmark_jobs_gpu.txt sbatch cybench/runs/slurm/screening.sh
```

### Parallelism (inside one job)

| Setting | Meaning |
|---------|---------|
| `experiment.n_jobs=1` | One Optuna trial at a time (default in `slurm_common.sh`) |
| `--cpus-per-task=8` | RF/XGB use all 8 cores **per trial** (`n_jobs=-1` in yaml) |
| `--gres=gpu:1` | One trial at a time on one GPU (torch **and TabPFN**) |

**TabPFN** uses `dataset.framework=pandas` + `feature_design` but sets `model.device=cuda` (see `tabpfn.yaml`). Schedule it in the **GPU array**, not the CPU one.

Optuna does **not** spawn separate SLURM tasks per trial.

## 3. Submit walk-forward

After screening finishes for a row, walk-forward finds the latest run automatically:

```text
output/baselines/<crop>_<country>_<model>_screening_<timestamp>/<test_years>/optimal_model.yaml
```

```bash
mkdir -p output/walk_forward
JOB_MANIFEST=cybench/runs/slurm/benchmark_jobs_cpu.txt sbatch cybench/runs/slurm/walk_forward.sh
```

## Cluster environment

Same as your previous script:

```bash
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
# Optional: export REPO_ROOT=/path/to/clone  (auto-detected from script location if omitted)
```

Submit from anywhere; `screening.sh` / `walk_forward.sh` resolve the repo root from their own path. Ensure Zenodo data is at `cybench/data` and `poetry install` has been run in that clone.

## Outputs

```text
output/baselines/<crop>_<country>_<model>_screening_<timestamp>/<test_years>/
  optimal_model.yaml
  optimal_feature_selection.yaml   # tabular + mRMR
  optimal_epochs.yaml              # neural nets
  42/report_metrics.yaml
```

## Suggested rollout

1. `benchmark_jobs.example.txt` → 6 jobs, verify pipeline  
2. `--countries US` → maize + wheat US, all models  
3. `generate_job_manifest.py` → full benchmark, split CPU/GPU arrays  
