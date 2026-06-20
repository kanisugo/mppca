# MPPCA — Mixture of Probabilistic PCA

Overview
- Lightweight implementation of Mixture of Probabilistic PCA (MPPCA) with a K-Means initializer and plotting utilities.
- Implements training, scoring, BIC calculation and visualization.

Repository files
- [mppca.py](mppca.py)
- [MPPCA.py](MPPCA.py)
- [hyperparameter_tuning.py](hyperparameter_tuning.py)

Main API (in this repo)
- [`mppca.KMeans`](mppca.py) — simple K-Means initializer used by the MPPCA implementation.
- [`mppca.MPPCA`](mppca.py) — main MPPCA class: fit, predict, score, get_bic.
- [`mppca.plot_results`](mppca.py) — plotting helpers for clusters, noise ratios and maps.
- [`MPPCA.MPPCA`](MPPCA.py) and [`MPPCA.KMeans`](MPPCA.py) — duplicate/alternative implementation in `MPPCA.py`.
- [`hyperparameter_tuning.MPPCA`](hyperparameter_tuning.py) — MPPCA used by the tuning script.

Quickstart
1. Install dependencies:
   pip install numpy pandas matplotlib scikit-learn scipy
2. Edit the CSV path in the script you want to run:
   - [mppca.py](mppca.py) or [MPPCA.py](MPPCA.py) — set `CSV_PATH`.
   - [hyperparameter_tuning.py](hyperparameter_tuning.py) — set dataset path for tuning.
3. Run:
   - Train & visualize: python mppca.py
   - Hyperparameter search: python hyperparameter_tuning.py

Notes
- The code expects columns: `latitude`, `longitude`, `altitude`, `magnetics`.
- Tweak `MAG_WEIGHT`, `M_CLUSTERS`, `Q_LATENT` at the top of the main scripts to change weighting, number of clusters and latent dim.
- For large datasets use the tuning script's subsampling or increase system resources.