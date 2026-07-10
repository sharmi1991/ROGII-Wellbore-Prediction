# Wellbore Geology Prediction — ROGII Kaggle Competition

## Problem Statement
Predict subsurface geological formation properties along a wellbore trajectory using well-log and drilling data. The goal is to estimate formation values at unseen depths/wells with high accuracy, measured via RMSE.

## Approach
Built an ensemble pipeline combining local and global modeling strategies:

- **Local neighbor-well modeling**: LightGBM + CatBoost trained via K-Fold (K=5) cross-validation on geologically similar neighboring wells (`N_NEIGHBORS=15`)
- **Global model**: A globally pretrained LightGBM (`GLOBAL_N_EST=300`) blended with local predictions using a tuned blending weight (`GLOBAL_LAMBDA=0.55`)
- **U-space polynomial projection** (degree 2): Reframing the coordinate space before modeling — this was the single biggest performance improvement in the entire pipeline
- **Multi-typewell DTW (Dynamic Time Warping)**: Aligning well trajectories against 3 representative "typewells" to capture geological analogs
- **Particle filter**: 8-seed particle filter (`N_PARTICLES=128`) with beam search for trajectory-level smoothing (`beta=0.95`)

## Key Engineering Challenge: Non-Determinism Bug
Discovered that re-running the best-scoring model (V34) produced a *different, worse* score (15.149 vs. the original 14.330 RMSE) on an unchanged codebase. Root-caused this to `n_jobs=-1` / `thread_count=-1` settings in LightGBM and CatBoost causing non-deterministic thread scheduling.

**Fix**: Rebuilt the pipeline (`solution_v34_deterministic.py`) with:
- All thread counts explicitly set to `1`
- `deterministic=True`
- `force_row_wise=True`

This is now the mandatory, trustworthy baseline for all further experimentation — every subsequent change is tested one-at-a-time against this deterministic base, and only trusted once confirmed via leaderboard (LB) submission.

## Results
- Started at ~820 RMSE with a baseline Random Forest
- Iteratively improved across ~40 versioned submissions
- **Best score: 14.330 RMSE (V34)** — currently a strong local optimum; most subsequent architectural changes (lambda CV tuning, multi-seed ensembles, formation features) have underperformed this baseline

## Key Learnings
- U-space polynomial projection >> raw feature space for this problem
- Blending a global model with local neighbor-well models outperforms either alone
- Multi-seed ensembling, counter-intuitively, hurt LB score here
- Cross-validation can silently leak (e.g., global model trained on validation wells) — always audit CV splits carefully
- **Reproducibility is not optional**: non-deterministic training (`n_jobs=-1`) can invalidate weeks of ablation conclusions if left unchecked

## Tech Stack
`Python` · `LightGBM` · `CatBoost` · `scikit-learn` · `NumPy` · `Pandas` · `Dynamic Time Warping` · `Particle Filtering`

## Status
Actively iterating — current experiment testing `N_NEIGHBORS=18` on the deterministic baseline, pending LB confirmation.
