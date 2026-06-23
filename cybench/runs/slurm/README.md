# SLURM benchmark jobs (WUR lustre)

**screening** + **walk-forward** via `run_experiments.py`.  
See also [../README.md](../README.md) for the full `cybench/runs/` layout (analysis, viz).

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
| `submit_array.sh` | Submit one phase + one manifest (array sizing, GPU auto) |
| `submit_benchmark.sh` | **Full pipeline**: split manifests → screening → walk-forward |
| `manifests/<batch>/` | Per-batch working manifests + **frozen snapshots** per sbatch |

### Manifest safety

Each `sbatch` copies the manifest to an **immutable snapshot** under
`cybench/runs/slurm/manifests/<batch>/` (timestamped filename). SLURM tasks read only that
snapshot, so `--regenerate` or another country's submit cannot change in-flight jobs.

`submit_benchmark.sh --batch NAME` keeps working manifests in the same folder
(`manifests/NAME/benchmark_jobs_cpu.txt`, …). Use a **distinct `--batch`** per country/run
(e.g. `baselines_us_mid_v1` vs `baselines_de_mid_v1`).

A sidecar `*.slurm_jobid` links each snapshot to its SLURM job id for auditing.

### SLURM job names (`squeue`)

**Array header** (pending / just submitted): `cb_{phase}_{group}_{horizon}`  
e.g. `cb_scr_cpu_eos`, `cb_wf_gpu_mid`, `cb_wf_fcp_eos` (GPU manifest on CPU via `--cpu`).

**Each running array task** is renamed at start to include the model:

`cb_{phase}_{model}_{crop}{country}` → e.g. `cb_scr_tabpfn_mzDE`, `cb_wf_informer_lf_whDE`

(`mz`/`wh` = maize/wheat.) Batch name is only in the job log line, not in `squeue`.

Rename uses `scontrol update JobId=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}` — updating
the array parent id alone would give every task the same name in `squeue`.

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

With ~16 models and ~40 countries × 2 crops, the full manifest can be **1000+ jobs**. Use `--array` ranges or split manifests:

```bash
# CPU tabular (feature_design, no GPU)
awk '$7=="no" && $6=="yes"' cybench/runs/slurm/benchmark_jobs.txt > cybench/runs/slurm/benchmark_jobs_cpu.txt
# Naive / standalone baselines (average, trend, lpjml_bc, twso_bc) — no HPO, no feature_design
awk '$5=="no" && $6=="no" && $7=="no"' cybench/runs/slurm/benchmark_jobs.txt > cybench/runs/slurm/benchmark_jobs_naive.txt
# GPU manifest (torch + TabPFN — pandas on GPU)
awk '$7=="yes"' cybench/runs/slurm/benchmark_jobs.txt > cybench/runs/slurm/benchmark_jobs_gpu.txt
```

**Note:** `benchmark_jobs_cpu.txt` from `$7=="no"` includes only models with `feature_design=yes`
(ridge, xgboost, …). **`average`, `trend`, `lpjml_bc`, and `twso_bc` are in `benchmark_jobs_naive.txt`**
— run that array too if you want those baselines in `compare_models.html`. Rows for `lpjml_bc` /
`twso_bc` are omitted at manifest generation when `lpjml_*.csv` / `twso_*.csv` is missing for a
crop/country.

## One-command pipeline (recommended)

From repo root — splits manifests, submits **cpu + naive + gpu** screening, then walk-forward
with `afterok` on each matching screening job. GPU partition/time inferred automatically for
the gpu manifest.

```bash
# Full eos benchmark (screening → walk-forward)
cybench/runs/slurm/submit_benchmark.sh all --horizon eos

# Pilot: regenerate manifest for DE/NL only, then submit
cybench/runs/slurm/submit_benchmark.sh all --horizon eos --regenerate --countries DE NL

# Isolated output batch (avoids flooding ../output/baselines/)
cybench/runs/slurm/submit_benchmark.sh all --horizon eos --batch baselines_pilot_2026q2

# Mid-season (second pass, after eos completes)
cybench/runs/slurm/submit_benchmark.sh all --horizon middle-of-season

# Screening only, first GPU job (TabPFN maize NL if first in gpu manifest)
cybench/runs/slurm/submit_benchmark.sh screening --horizon eos --array 0 --only gpu

# GPU manifest on CPU (bypass gpu queue; torch screening is very slow)
cybench/runs/slurm/submit_benchmark.sh all --horizon eos --regenerate --countries DE \
  --batch baselines_de_eos_v1 --only gpu --cpu

# Preview without sbatch
cybench/runs/slurm/submit_benchmark.sh all --horizon eos --dry-run
```

## Complete missing jobs (partial rerun)

After a batch finishes, some array tasks may fail (OOM, timeout, missing screening
artifact for walk-forward). SLURM jobs **do not** skip work that already succeeded —
build a **partial manifest** and submit only incomplete rows:

```bash
# Per-job status: ok / MISS / BLOCK (too few yield years for screening split)
cybench/runs/slurm/orchestrate_benchmark_complete.sh \
  --batch baselines_DE_eos_v1 --horizon eos --list

# Submit retries (screening + walk-forward with afterok; skips complete jobs)
cybench/runs/slurm/orchestrate_benchmark_complete.sh \
  --batch baselines_DE_eos_v1 --horizon eos --submit

# Both horizons (auto-resolves manifest; uses lustre output if present)
cybench/runs/slurm/orchestrate_benchmark_complete.sh \
  --country DE --horizons eos mid --list

cybench/runs/slurm/orchestrate_benchmark_complete.sh \
  --all-countries --horizons eos mid --list

cybench/runs/slurm/orchestrate_benchmark_complete.sh \
  --all-countries --horizons eos mid --submit --dry-run

# Walk-forward only (screening already ok everywhere)
cybench/runs/slurm/orchestrate_benchmark_complete.sh \
  --batch baselines_DE_eos_v1 --phase walk_forward --submit

# One model only (e.g. re-run lpjml_bc or twso_bc after a code fix)
cybench/runs/slurm/orchestrate_benchmark_complete.sh \
  --all-countries --horizon eos --model lpjml_bc \
  --phase walk_forward --force-rerun --list

cybench/runs/slurm/orchestrate_benchmark_complete.sh \
  --all-countries --horizon eos --model twso_bc \
  --phase walk_forward --force-rerun --submit --dry-run
```

Preflight marks rows **BLOCK** when yield years cannot satisfy the fixed screening
split (`5-last` test + `2-last` val + train). Those are excluded from the retry
manifest. Use `--cpu` on large GPU batches if the queue is backed up.

**Country code case:** batch folders may be `baselines_de_eos_v1` (manual) or
`baselines_DE_eos_v1` (orchestrated). Completion resolves output/manifest paths
case-insensitively and submits using the on-disk folder name.

Manual per-manifest submits (fine-grained control) are below.

## 2. Submit screening

Submit **from the repo root** (so `SLURM_SUBMIT_DIR` resolves `cybench/runs/slurm/`).

**Recommended** — array size is computed from the manifest automatically:

```bash
cybench/runs/slurm/submit_array.sh screening cybench/runs/slurm/benchmark_jobs_cpu.txt
cybench/runs/slurm/submit_array.sh screening cybench/runs/slurm/benchmark_jobs_gpu.txt
```

**Manual** — override `#SBATCH --array` on the command line (no need to edit the script):

```bash
mkdir -p output/screening
N=$(awk '!/^#/ && NF>=7' cybench/runs/slurm/benchmark_jobs_cpu.txt | wc -l)
JOB_MANIFEST=cybench/runs/slurm/benchmark_jobs_cpu.txt \
  sbatch --array=0-$((N - 1)) cybench/runs/slurm/screening.sh
```

GPU jobs: `submit_array.sh` requests **gpu** partition, **`--gpus=1`**, and
**`--time=2-00:00:00`** (overrides screening.sh’s 4-day default; GPU partition
walltime is shorter). Override:

```bash
export SLURM_GPU_PARTITION=gpu
export SLURM_GPU_REQUEST="--gpus=1"
export SLURM_GPU_TIME_LIMIT=2-00:00:00
```

### Parallelism (inside one job)

| Setting | Meaning |
|---------|---------|
| `experiment.n_jobs=1` | One Optuna trial at a time (default in `slurm_common.sh`) |
| `--cpus-per-task=8` | RF/XGB use all 8 cores **per trial** (`n_jobs=-1` in yaml) |
| `--gpus=1` + `-p gpu` | GPU jobs (via `submit_array.sh` + GPU manifest) |

**TabPFN** uses `dataset.framework=pandas` + `feature_design` but sets `model.device=cuda` (see `tabpfn.yaml`). Schedule it in the **GPU array**, not the CPU one.

### GPU manifest on CPU (`--cpu`)

When the **gpu** queue is backed up, run torch + TabPFN on the **main** partition instead:

```bash
# One manifest / phase
cybench/runs/slurm/submit_array.sh screening \
  cybench/runs/slurm/manifests/baselines_de_eos_v1/benchmark_jobs_gpu.txt --cpu

# Full pipeline (gpu group only)
cybench/runs/slurm/submit_benchmark.sh all --horizon eos \
  --regenerate --countries DE --batch baselines_de_eos_v1 --only gpu --cpu
```

`--cpu` does two things:

1. **SLURM** — no `-p gpu` / `--gpus=1` (uses `main` + `screening.sh` defaults)
2. **Hydra** — `CYBENCH_FORCE_CPU=1` → `experiment.device=cpu` (torch), `model.device=cpu` (TabPFN)

Logs show `device=cpu (CYBENCH_FORCE_CPU)` in the `Screening | …` line. Torch screening with HPO is **much slower** on CPU; TabPFN is often acceptable. Optional: `export HP_TRIALS=5` for pilots.

Optuna does **not** spawn separate SLURM tasks per trial.

## Output batches

By default Hydra writes to **`../output/baselines/`**. Re-runs add new timestamped folders;
`latest_only` discovery keeps analysis sane, but the directory still grows.

Use **`--batch NAME`** (maps to Hydra `experiment.name`) to isolate a benchmark run:

```bash
cybench/runs/slurm/submit_benchmark.sh all --horizon eos --batch baselines_full_eos_v1
```

Results land in **`../output/baselines_full_eos_v1/`**. Screening and walk-forward in one
`submit_benchmark.sh all` call share the same batch automatically. For manual submits:

```bash
cybench/runs/slurm/submit_array.sh screening cybench/runs/slurm/benchmark_jobs_cpu.txt \
  --batch baselines_full_eos_v1
# after screening finishes:
cybench/runs/slurm/submit_array.sh walk_forward cybench/runs/slurm/benchmark_jobs_cpu.txt \
  --batch baselines_full_eos_v1
```

Or export **`CYBENCH_EXPERIMENT_NAME`** (same value for screening and walk-forward).
Override the resolved path with **`CYBENCH_BASELINES_DIR`** if needed.

Point analysis at the batch folder:

```bash
poetry run python cybench/runs/analysis/collect_walk_forward_results.py \
  --baselines-dir ../output/baselines_full_eos_v1 \
  --output-dir ../output/paper_walk_forward_eos_v1
```

## 3. Submit walk-forward

After screening finishes for a row, walk-forward finds the latest run under the active
batch directory (default **`../output/baselines/`**, or `../output/<batch>/` with `--batch`).

```text
../output/<batch>/<crop>_<country>_<model>_screening_<horizon>_<timestamp>/<test_years>/optimal_model.yaml
```

Set the array to match the manifest — or use `submit_array.sh` (same as screening):

```bash
cybench/runs/slurm/submit_array.sh walk_forward cybench/runs/slurm/benchmark_jobs_cpu.txt
cybench/runs/slurm/submit_array.sh walk_forward cybench/runs/slurm/benchmark_jobs_gpu.txt
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
PREDICTION_HORIZON=middle-of-season cybench/runs/slurm/submit_array.sh walk_forward ...
```

`submit_array.sh` prints `horizon: ...` on submit. In the job log, confirm
`horizon=middle-of-season` and run dirs named `*_walk_forward_mid_season_*`
(not `*_eos_*`). `run_experiments.py` also logs `Prediction horizon (end_of_sequence): ...`.

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
poetry run python cybench/runs/analysis/collect_walk_forward_results.py \
  --baselines-dir ../output/baselines \
  --output-dir ../output/paper_walk_forward \
  --plot --dashboard
```

**Compare models** (from an existing collect output):

```bash
poetry run python cybench/runs/analysis/collect_walk_forward_results.py \
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
poetry run python cybench/runs/analysis/compare_benchmark_runs.py \
  --baselines-dir ../output/baselines \
  --group wf=walk_forward/eos \
  --group scr=screening/eos \
  --output ../output/compare_wf_vs_screen_eos.csv

# End-of-season vs mid-season walk-forward
poetry run python cybench/runs/analysis/compare_benchmark_runs.py \
  --baselines-dir ../output/baselines \
  --group eos=walk_forward/eos \
  --group mid=walk_forward/mid_season \
  --output ../output/compare_horizons.csv
```

CSV columns are prefixed per group (`wf__nrmse`, `scr__r2`, …) plus `delta__*` for the first vs second group.
Rows match on `(crop, country, model)`; horizons can differ between groups (`eos__horizon`, `mid__horizon`).
NRMSE is lower-is-better; correlation and R² are higher-is-better.

**Polygons for maps** (if `--plot` fails on shapefiles):

```bash
poetry run python data_preparation/fetch_zenodo_data.py --geometries
# creates cybench/data/polygons/DE/DE.shp, NL/NL.shp, ...
```

**World outline** (grey background in map panels): bundled at
``data_preparation/ne_50m_admin_0_countries/`` (default). Falls back to 110m if missing.
Override with ``CYBENCH_WORLD_MAP_SCALE=10|50|110``.

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
