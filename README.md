# AgML CY-Bench Modeling

Code companion to:

> **Benchmarking the State of AI for Crop Yield Forecasting: A Global Assessment Across Modeling Paradigms**  
> Kallenberg et al., KDD 2027.

This repository implements the modeling and evaluation pipeline used in that study. Interactive results are published at **[cybench.agml.org](https://cybench.agml.org/)**.

The underlying subnational yield dataset and data-preparation protocols are described in the CY-Bench dataset paper ([Kallenberg et al., ESSD 2026](https://essd.copernicus.org/articles/18/3997/2026/)) and released on [Zenodo](https://doi.org/10.5281/zenodo.11502142). Dataset code lives in [WUR-AI/AgML-CY-Bench](https://github.com/WUR-AI/AgML-CY-Bench).

---

## What the paper studies

CY-Bench provides a common evaluation setting for **in-season, subnational maize and wheat yield forecasting**. Building on that resource, this work asks how far today’s AI methods go when compared under a shared protocol—not only on overall accuracy, but also on **spatial**, **temporal**, and **anomaly** skill.

We compare modeling paradigms that are routinely used (or proposed) for crop yield forecasting:

- statistical and process-based baselines  
- conventional machine learning with engineered features  
- deep sequence models (late-fusion architectures over weather and related time series)  
- tabular foundation models  

Forecasts are evaluated at multiple points in the season (e.g. mid-season and end-of-season). Models are selected under a **screening** protocol, then assessed with **walk-forward** evaluation that mirrors operational forecasting.

---

## Key findings (summary)

Results are easiest to explore in the [dashboard](https://cybench.agml.org/). At a high level:

- Feature-engineered ML and tabular foundation models achieve the strongest overall skill in most countries.
- Spatial prediction is substantially easier than temporal and anomaly prediction across modeling paradigms.
- Process-based models show comparatively strong temporal skill despite lower overall accuracy; year-to-year anomalies remain hard.

Country pages on the dashboard break these patterns down by crop, horizon, and metric; global insight pages summarize cross-country comparisons.

---

## This repository

This codebase is the implementation behind the paper experiments and the dashboard: Hydra-configured datasets and models, screening and walk-forward evaluation, and the analysis/visualization used to produce figures and interactive views.

For a map of results rather than a tour of the source tree, start at **[cybench.agml.org](https://cybench.agml.org/)**.

---

## Citation

Please cite the modeling paper and, when using the dataset, the CY-Bench data paper:

```
Kallenberg et al. Benchmarking the State of AI for Crop Yield Forecasting:
A Global Assessment Across Modeling Paradigms. KDD 2027.
```

```
@article{kallenberg_etal2026_cybench,
  title   = {CY-Bench: A comprehensive benchmark dataset for subnational crop yield forecasting},
  author  = {Kallenberg, Michiel and others},
  journal = {Earth System Science Data},
  year    = {2026},
  volume  = {18},
  pages   = {3997--4025},
  doi     = {10.5194/essd-18-3997-2026},
  url     = {https://essd.copernicus.org/articles/18/3997/2026/}
}
```

(Update the modeling bibtex entry with the official proceedings citation when available.)

---

## Links

| Resource | URL |
|----------|-----|
| Results dashboard | https://cybench.agml.org/ |
| AgML | https://www.agml.org/ |
| CY-Bench dataset (ESSD) | https://essd.copernicus.org/articles/18/3997/2026/ |
| Dataset release | https://doi.org/10.5281/zenodo.11502142 |
| Dataset / data-prep repository | https://github.com/WUR-AI/AgML-CY-Bench |
