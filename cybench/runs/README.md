# Experiment runs

Hydra-driven benchmarks and tooling around `../output/baselines/`.

## Layout

| Path | Purpose |
|------|---------|
| `run_experiments.py` | Main entry: screening, walk-forward, rolling (Hydra) |
| `slurm/` | HPC array jobs — see [slurm/README.md](slurm/README.md) |
| `analysis/` | Discover runs, pool walk-forward metrics, compare groups |
| `viz/` | Per-model plots and multi-model HTML dashboard |
| `legacy/` | Older `run_benchmark` scripts (not used by the SLURM pipeline) |

## Workflow

```bash
# 1. Local or SLURM experiment
poetry run python cybench/runs/run_experiments.py dataset/crop=maize dataset.country=NL ...

# 2. Collect walk-forward for the paper
poetry run python cybench/runs/analysis/collect_walk_forward_results.py \
  --baselines-dir ../output/baselines \
  --output-dir ../output/paper_walk_forward_eos \
  --plot --dashboard

# 3. Compare run groups (e.g. eos vs mid_season, screening vs walk-forward)
poetry run python cybench/runs/analysis/compare_benchmark_runs.py \
  --baselines-dir ../output/baselines \
  --group eos=walk_forward/eos \
  --group mid=walk_forward/mid_season \
  --output ../output/compare_horizons.csv
```

Run directories follow  
`{crop}_{country}_{model}_{phase}_{horizon}_{timestamp}/`  
under `../output/baselines/`.
