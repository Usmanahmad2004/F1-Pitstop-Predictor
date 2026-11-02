# Predicting F1 Pit Stops (CSE 4820)

Usman Ahmad, Jichen Li, Logan Demianczyk, Kenneth Chow

Predict whether a driver pits on the next lap (`PitNextLap`) from lap-level data, using the Kaggle
Playground S6E5 competition ("Predicting F1 Pit Stops"). The metric is ROC AUC.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download the data from https://www.kaggle.com/competitions/playground-series-s6e5/data (join the
competition first), then put `train.csv` and `test.csv` in `data/`.

## Run

```bash
python midpoint.py                 # full run, about 30 to 45 min on a laptop (Random Forest is the slow part)
QUICK=1 python midpoint.py         # 30% of races, fewer trees, for a fast check
```

Or open `midpoint.ipynb` and run all cells (same code; `midpoint.py` is the source of truth and the
notebook is generated from it with `jupytext --to ipynb midpoint.py`).

## What it does

1. EDA: class balance, pit rate by tyre life / compound / race progress, correlations with the target
2. Feature engineering (`src/features.py`): each driver's laps are grouped by race and sorted by lap, then
   lags, rolling pace, lap time slopes, pace vs stint and best lap, tyre life vs the typical stint length
   for that race and compound, laps remaining, and how many cars in the field have pitted recently.
   Only the current and earlier laps are used.
3. Validation: 5-fold `StratifiedGroupKFold` grouped by race event (Race + Year), so every validation
   fold only has races the model never trained on. LightGBM early stopping uses an inner holdout of the
   training races, never the validation fold.
4. Models: Logistic Regression (raw and engineered features), Random Forest, LightGBM (raw and
   engineered), plus a random K-fold LightGBM run to measure how much a non-grouped split overestimates AUC.

## Outputs

- `outputs/figures/`: EDA plots, ROC curves, model comparison, feature importance
- `outputs/results/cv_results.csv`: out-of-fold AUC, fold mean/std, PR AUC and runtime per model
- `outputs/results/feature_importance.csv`, `feature_family_importance.csv`
- `outputs/results/submission_lgbm.csv`: Kaggle submission from the engineered LightGBM
- `outputs/results/summary.json`: everything above in one file

## Next (after midpoint)

XGBoost and CatBoost, hyperparameter tuning, blending, SHAP analysis, final report and presentation.
