AgML CY-Bench Modeling
======================

Code companion to:

   | **Benchmarking the State of AI for Crop Yield Forecasting: A Global Assessment Across Modeling Paradigms**
   | Michiel Kallenberg, Philip Janz, Christoph Jörges, Vageesh Saxena, Pratishtha Poudel, Mohammed Musthafa Rafi, and Ioannis N. Athanasiadis
   | (submitted to KDD ’27)

Interactive results: `wur-ai.github.io/AgML-CY-Bench-dashboard <https://wur-ai.github.io/AgML-CY-Bench-dashboard/>`__.

The underlying subnational yield dataset is CY-Bench (`Kallenberg et al., ESSD 2026 <https://essd.copernicus.org/articles/18/3997/2026/>`__; `Zenodo <https://doi.org/10.5281/zenodo.11502142>`__). Dataset and data-preparation code: `WUR-AI/AgML-CY-Bench <https://github.com/WUR-AI/AgML-CY-Bench>`__.

--------------

Summary of the study
--------------------

Accurate crop yield forecasting matters for food security, agricultural policy, and commodity markets. Statistical, process-based, and machine learning approaches have all been proposed, but their relative strengths remain poorly understood without large-scale, standardized comparisons.

This work benchmarks **five modeling paradigms** on CY-Bench—**63 country–crop datasets** (>12,000 administrative regions) across six continents:

1. Statistical baselines
2. Process-based models
3. Conventional feature-engineered machine learning
4. Tabular foundation models
5. Deep sequence models

Beyond pooled accuracy, the paper uses a multi-dimensional evaluation that separately quantifies **overall**, **spatial**, **temporal**, and **anomaly** prediction skill. Models are compared under a shared protocol with in-season forecast horizons (including mid-season and end-of-season).

Main findings
~~~~~~~~~~~~~

-  Feature-engineered ML and tabular foundation models achieve the strongest overall predictive performance, though gains over simple statistical baselines are modest: data-driven models reduce NRMSE in **72%** of maize countries and **78%** of wheat countries (median reductions **8.8%** and **11.9%**).
-  Across all paradigms, **spatial** prediction is substantially easier than **temporal** and **anomaly** prediction.
-  Data-driven models markedly improve temporal and anomaly prediction over statistical baselines, yet accurately forecasting **year-to-year** yield variability remains challenging.
-  Progress at scale will depend more on better characterization of the environmental and management factors that drive inter-annual variability than on paradigm choice alone.

Explore country-level and cross-country results in the `dashboard <https://wur-ai.github.io/AgML-CY-Bench-dashboard/>`__.

--------------

This repository
---------------

Implementation of the modeling and evaluation pipeline behind the paper and dashboard (experiment configuration, model families above, screening and walk-forward evaluation, analysis and visualization).

For results, prefer the `dashboard <https://wur-ai.github.io/AgML-CY-Bench-dashboard/>`__ over browsing the source tree.

Quick start
~~~~~~~~~~~

.. code:: bash

   poetry install
   # put CY-Bench data under cybench/data/ (see Zenodo link above)
   poetry run python cybench/runs/run_experiments.py \
     dataset/crop=maize dataset.country=NL model=ridge \
     dataset/temporal=feature_design validation=single \
     experiment.device=cpu experiment.n_repetitions=1

-  **Models & configs:** ``cybench/models/`` (implementations) and ``cybench/conf/model/`` (Hydra configs; pick a model with ``model=<name>``).
-  **Entry point:** ``cybench/runs/run_experiments.py`` (screening / walk-forward). Cluster jobs: ``cybench/runs/slurm/``.

--------------

Links
-----

+--------------------------------+----------------------------------------------------+
| Resource                       | URL                                                |
+================================+====================================================+
| Results dashboard              | https://wur-ai.github.io/AgML-CY-Bench-dashboard/  |
+--------------------------------+----------------------------------------------------+
| AgML                           | https://www.agml.org/                              |
+--------------------------------+----------------------------------------------------+
| CY-Bench dataset (ESSD)        | https://essd.copernicus.org/articles/18/3997/2026/ |
+--------------------------------+----------------------------------------------------+
| Dataset release                | https://doi.org/10.5281/zenodo.11502142            |
+--------------------------------+----------------------------------------------------+
| Dataset / data-prep repository | https://github.com/WUR-AI/AgML-CY-Bench            |
+--------------------------------+----------------------------------------------------+

A citation for this modeling paper will be added when a preprint or proceedings version is available. Please cite the `CY-Bench dataset paper <https://essd.copernicus.org/articles/18/3997/2026/>`__ when using the data.
