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
| `submit_array.sh` | Count manifest rows and `sbatch --array=0-(N-1)` for you |

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
# Naive baselines (average, trend) — no HPO, no feature_design; submit separately
awk '$5=="no" && $6=="no" && $7=="no"' cybench/runs/slurm/benchmark_jobs.txt > cybench/runs/slurm/benchmark_jobs_naive.txt
# GPU manifest (torch + TabPFN — pandas on GPU)
awk '$7=="yes"' cybench/runs/slurm/benchmark_jobs.txt > cybench/runs/slurm/benchmark_jobs_gpu.txt
```

**Note:** `benchmark_jobs_cpu.txt` from `$7=="no"` includes only models with `feature_design=yes`
(ridge, xgboost, …). **`average` and `trend` are in `benchmark_jobs_naive.txt`** — run those
arrays too if you want the naive baseline in `compare_models.html`.

## 2. Submit screening

Submit **from the repo root** (so `SLURM_SUBMIT_DIR` resolves `cybench/runs/slurm/`).

**Recommended** — array size is computed from the manifest automatically:

```bash
cybench/runs/slurm/submit_array.sh screening cybench/runs/slurm/benchmark_jobs_cpu.txt
cybench/runs/slurm/submit_array.sh screening cybench/runs/slurm/benchmark_jobs_gpu.txt --gpu
```

**Manual** — override `#SBATCH --array` on the command line (no need to edit the script):

```bash
mkdir -p output/screening
N=$(awk '!/^#/ && NF>=7' cybench/runs/slurm/benchmark_jobs_cpu.txt | wc -l)
JOB_MANIFEST=cybench/runs/slurm/benchmark_jobs_cpu.txt \
  sbatch --array=0-$((N - 1)) cybench/runs/slurm/screening.sh
```

GPU jobs: add `--gres=gpu:1` (and uncomment `#SBATCH --partition=gpu` in `screening.sh` if your cluster requires it).

### Parallelism (inside one job)

| Setting | Meaning |
|---------|---------|
| `experiment.n_jobs=1` | One Optuna trial at a time (default in `slurm_common.sh`) |
| `--cpus-per-task=8` | RF/XGB use all 8 cores **per trial** (`n_jobs=-1` in yaml) |
| `--gres=gpu:1` | One trial at a time on one GPU (torch **and TabPFN**) |

**TabPFN** uses `dataset.framework=pandas` + `feature_design` but sets `model.device=cuda` (see `tabpfn.yaml`). Schedule it in the **GPU array**, not the CPU one.

Optuna does **not** spawn separate SLURM tasks per trial.

## 3. Submit walk-forward

After screening finishes for a row, walk-forward finds the latest run under **`../output/baselines/`**
(one level above the repo — same path Hydra uses).

```text
../output/baselines/<crop>_<country>_<model>_screening_<timestamp>/<test_years>/optimal_model.yaml
```

Set the array to match the manifest — or use `submit_array.sh` (same as screening):

```bash
cybench/runs/slurm/submit_array.sh walk_forward cybench/runs/slurm/benchmark_jobs_cpu.txt
cybench/runs/slurm/submit_array.sh walk_forward cybench/runs/slurm/benchmark_jobs_gpu.txt --gpu
```

Manual:

```bash
mkdir -p output/walk_forward
N=$(awk '!/^#/ && NF>=7' cybench/runs/slurm/benchmark_jobs_cpu.txt | wc -l)
JOB_MANIFEST=cybench/runs/slurm/benchmark_jobs_cpu.txt \
  sbatch --array=0-$((N - 1)) cybench/runs/slurm/walk_forward.sh
```

## Cluster environment

Same as your previous script:

```bash
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
# Optional: export REPO_ROOT=/path/to/clone  (auto-detected from script location if omitted)
```

Submit from anywhere; `screening.sh` / `walk_forward.sh` resolve the repo root from their own path.

Fetch data (once per clone, Python 3.10+, ~6.2 GB):

```bash
poetry run python data_preparation/fetch_zenodo_data.py
```

Then `poetry install` if you have not already.

## Prediction horizon (lead time)

Forecast cutoff is `dataset.temporal.season.end_of_sequence`:

| Value | Meaning | Run-name tag |
|-------|---------|--------------|
| `eos` | End-of-season (nowcast) | `eos` |
| `middle-of-season` or `mid-season` | Mid-season forecast | `mid_season` |
| `eos-60` | 60 days before EOS | `eos_60` |

Hydra run folders and prediction CSVs include the tag, e.g.:

```text
../output/baselines/maize_NL_ridge_screening_eos_20260615_120738/
../output/baselines/maize_NL_ridge_walk_forward_mid_season_20260620_093000/
maize_NL_h_eos_year_2016.csv
```

On SLURM, set the horizon for both screening and walk-forward (walk-forward matches screening by horizon):

```bash
PREDICTION_HORIZON=eos cybench/runs/slurm/submit_array.sh screening ...
PREDICTION_HORIZON=middle-of-season cybench/runs/slurm/submit_array.sh screening ...

# walk-forward must use the same horizon as its screening run:
PREDICTION_HORIZON=eos cybench/runs/slurm/submit_array.sh walk_forward ...
```

Local override:

```bash
poetry run python cybench/runs/run_experiments.py \
  dataset/crop=maize dataset.country=NL \
  dataset.temporal.season.end_of_sequence=middle-of-season \
  validation=screening model=ridge ...
```

## Paper reporting (walk-forward)

After walk-forward jobs finish, pool per-year splits into one metrics table and plots.
Requires country shapefiles under ``cybench/data/polygons/<CC>/<CC>.shp`` (see below).

```bash
poetry run python cybench/runs/collect_walk_forward_results.py \
  --baselines-dir ../output/baselines \
  --output-dir ../output/paper_walk_forward \
  --plot --dashboard
```

**Compare models** (from an existing collect output):

```bash
poetry run python cybench/runs/collect_walk_forward_results.py \
  --output-dir ../output/paper_walk_forward_eos \
  --dashboard-only
```

Open `compare_models.html` — heatmap of all models × datasets; click a row for scatter/maps.
Copy `compare_models.html` + `assets/` to your laptop to view offline.

Outputs:

- `walk_forward_summary.csv` — one row per crop/country/model (pooled region-year metrics)
- `compare_models.html` — **all models side-by-side** (use `--dashboard`)
- `preds/<model>_<horizon>/` — year CSVs for `visualize_results_aggregated.py`
- `plots/<model>/report.html` — per-model interactive report (when `--plot`)

Screening runs are only used to freeze HP; the paper table should come from walk-forward.

**Compare run groups** (screening vs walk-forward, eos vs mid_season, etc.):

```bash
# Screening vs walk-forward (full metric series + deltas)
poetry run python cybench/runs/compare_benchmark_runs.py \
  --baselines-dir ../output/baselines \
  --group wf=walk_forward/eos \
  --group scr=screening/eos \
  --output ../output/compare_wf_vs_screen_eos.csv

# End-of-season vs mid-season walk-forward
poetry run python cybench/runs/compare_benchmark_runs.py \
  --baselines-dir ../output/baselines \
  --group eos=walk_forward/eos \
  --group mid=walk_forward/mid_season \
  --output ../output/compare_horizons.csv
```

CSV columns are prefixed per group (`wf__nrmse`, `scr__r2`, …) plus `delta__*` for the first vs second group.
NRMSE is lower-is-better; correlation and R² are higher-is-better.

**Polygons for maps** (if `--plot` fails on shapefiles):

```bash
poetry run python data_preparation/fetch_zenodo_data.py --geometries
# creates cybench/data/polygons/DE/DE.shp, NL/NL.shp, ...
```

Run collect from the same repo clone (paths resolve via ``REPO_DIR``).

## Outputs

Hydra experiment artifacts (screening + walk-forward). Default `store.*` flags skip
duplicate exports; each split keeps `test_preds.csv` and `report_metrics.yaml`.

```text
../output/baselines/<crop>_<country>_<model>_screening_<horizon>_<timestamp>/
  .hydra/                          # full composed config (kept)
  <test_years>/
    optimal_model.yaml             # screening / HPO (needed for walk-forward)
    optimal_feature_selection.yaml # tabular + mRMR
    optimal_epochs.yaml            # neural nets
    screening_partitions.yaml      # train/val/test audit (screening only)
    42/
      test_preds.csv
      report_metrics.yaml
```

Re-enable legacy flat CSVs at run root: `store.export_root_csv=true`  
Re-enable per-split config dumps: `store.save_split_configs=true`

## Suggested rollout

1. `benchmark_jobs.example.txt` → 6 jobs, verify pipeline  
2. `--countries US` → maize + wheat US, all models  
3. `generate_job_manifest.py` → full benchmark, split CPU/GPU arrays  
