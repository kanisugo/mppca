# MPPCA (Mixture of Probabilistic PCA)

This repository contains a lightweight implementation of MPPCA used for clustering geospatial magnetic data. The codebase is modularized for easier reading and maintenance.

Files:
- `kmeans.py` - Simple K-Means implementation used for initialization.
- `mppca.py` - MPPCA model (trainable, score, BIC, predict).
- `data_utils.py` - Data loading and preprocessing helpers.
- `plot_utils.py` - Plotting helper functions to visualize results.
- `hyperparameter_tuning.py` - Orchestrator script that runs cross-validated grid search.

Quickstart:
1. Create a virtual environment: `python3 -m venv .venv && source .venv/bin/activate`
2. Install dependencies: `pip install -r requirements.txt`
3. Update `CSV_PATH` in `hyperparameter_tuning.py` to point to your `magnetic.csv` file.
4. Run: `python hyperparameter_tuning.py`

Notes:
- Keep large data files out of the repository; use `.gitignore` to exclude them.
- Add a license if you plan to open-source this project.
