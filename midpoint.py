# %% [markdown]
# # F1 Pit Stop Prediction: Midpoint
# CSE 4820 group project. Kaggle Playground S6E5, predict `PitNextLap` and score on ROC AUC.
#
# Midpoint scope (from the proposal): EDA, logistic regression baseline, Random Forest and LightGBM,
# compared under race-grouped 5-fold CV.

# %%
import json
import os
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path.cwd() / "src"))
from features import build_features, target, id_col, cat_cols

warnings.filterwarnings("ignore")

data_dir = Path(os.environ.get("DATA_DIR", "data"))
fig_dir = Path("outputs/figures")
res_dir = Path("outputs/results")
fig_dir.mkdir(parents=True, exist_ok=True)
res_dir.mkdir(parents=True, exist_ok=True)

seed = 42
n_folds = 5
quick = os.environ.get("QUICK") == "1"

#chart colors (validated categorical slots 1-3, text inks)
blue, orange, aqua = "#2a78d6", "#eb6834", "#1baf7a"
ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": grid,
    "axes.labelcolor": ink2, "xtick.color": ink2, "ytick.color": ink2, "text.color": ink,
    "axes.grid": True, "grid.color": grid, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "savefig.dpi": 200, "savefig.bbox": "tight",
})

# %% [markdown]
# ## 1. Data

# %%
train = pd.read_csv(data_dir / "train.csv")
test = pd.read_csv(data_dir / "test.csv")
if quick:
    keep = train[["Race", "Year"]].drop_duplicates().sample(frac=0.3, random_state=seed)
    train = train.merge(keep, on=["Race", "Year"])
print("train", train.shape, "test", test.shape)
train.head()

# %% [markdown]
# ## 2. Exploratory data analysis

# %%
eda = {}
eda["n_train"] = len(train)
eda["n_test"] = len(test)
eda["pos_rate"] = float(train[target].mean())
eda["n_races"] = int(train[["Race", "Year"]].drop_duplicates().shape[0])
eda["n_race_names"] = int(train["Race"].nunique())
eda["years"] = sorted(int(y) for y in train["Year"].unique())
eda["n_drivers"] = int(train["Driver"].nunique())
eda["missing"] = {k: int(v) for k, v in train.isna().sum().items() if v > 0}
laps_per = train.groupby(["Driver", "Race", "Year"]).size()
eda["laps_per_driver_race_median"] = float(laps_per.median())
eda["compound_share"] = train["Compound"].value_counts(normalize=True).round(4).to_dict()
eda["pit_rate_by_compound"] = train.groupby("Compound")[target].mean().round(4).to_dict()
eda["pit_rate_by_stint"] = train.groupby("Stint")[target].agg(["mean", "size"]).query("size > 500")["mean"].round(4).to_dict()
eda["pit_rate_when_pitstop_now"] = train.groupby("PitStop")[target].mean().round(4).to_dict()
num = train.select_dtypes("number").columns.drop([id_col, target])
corr = train[list(num) + [target]].corr()[target].drop(target).sort_values()
eda["corr_with_target"] = corr.round(4).to_dict()
print(json.dumps(eda, indent=1, default=str))
train.describe().T

# %%
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

ax = axes[0]
tl = train.assign(tl=train["TyreLife"].clip(upper=50).round()).groupby("tl")[target].agg(["mean", "size"])
tl = tl[tl["size"] >= 200]
ax.plot(tl.index, tl["mean"], color=blue, lw=2)
ax.axhline(eda["pos_rate"], color=ink2, lw=1, ls="--")
ax.text(tl.index.max(), eda["pos_rate"], " overall", va="bottom", ha="right", color=ink2, fontsize=8)
ax.set(title="Pit rate vs tyre life", xlabel="TyreLife (laps)", ylabel="P(pit next lap)")

ax = axes[1]
order = train.groupby("Compound")[target].mean().sort_values(ascending=False)
ax.bar(order.index, order.values, color=blue, width=0.6)
for i, v in enumerate(order.values):
    ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=ink2)
ax.set(title="Pit rate by compound", ylabel="P(pit next lap)")
ax.tick_params(axis="x", labelsize=8)

ax = axes[2]
rp = train.assign(b=(train["RaceProgress"].clip(0, 1) * 20).round() / 20).groupby("b")[target].mean()
ax.plot(rp.index, rp.values, color=blue, lw=2)
ax.set(title="Pit rate vs race progress", xlabel="RaceProgress", ylabel="P(pit next lap)")
plt.tight_layout()
plt.savefig(fig_dir / "eda_pit_rates.png")
plt.show()

# %%
fig, ax = plt.subplots(figsize=(6, 3.8))
colors = [blue if v > 0 else orange for v in corr.values]
ax.barh(corr.index, corr.values, color=colors, height=0.6)
ax.axvline(0, color=ink2, lw=0.8)
ax.set(title="Pearson correlation with PitNextLap", xlabel="correlation")
plt.savefig(fig_dir / "eda_correlations.png")
plt.show()

# %% [markdown]
# ## 3. Feature engineering
# Group by (Driver, Race, Year), sort by LapNumber, and build features from the current and earlier laps:
# lags, rolling pace, slopes, pace vs stint/best lap, tyre life vs the typical stint length for that
# race and compound, laps remaining, and how many cars in the field have pitted recently.

# %%
t0 = time.time()
train_fe, test_fe = build_features(train, test)
y = train_fe[target].astype(int).values
raw_feats = [c for c in train.columns if c not in (id_col, target)]
all_feats = [c for c in train_fe.columns if c not in (id_col, target)]
eng_feats = [c for c in all_feats if c not in raw_feats]
print(f"raw {len(raw_feats)}, engineered {len(eng_feats)}, total {len(all_feats)} ({time.time() - t0:.0f}s)")

# %% [markdown]
# ## 4. Validation
# Laps from the same race are highly correlated, so folds are grouped by race event (Race + Year).
# Every validation fold only has races the model never saw in training.

# %%
groups = (train_fe["Race"].astype(str) + "_" + train_fe["Year"].astype(str)).values
sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
folds = list(sgkf.split(train_fe, y, groups))
for k, (tr, va) in enumerate(folds):
    print(f"fold {k + 1}: {len(np.unique(groups[va]))} races, {len(va)} laps, pos rate {y[va].mean():.3f}")


# %%
def num_matrix(df, feats):
    cols = [c for c in feats if c not in cat_cols]
    x = df[cols].astype(float)
    x = pd.concat([x, pd.get_dummies(df["Compound"].astype(str), prefix="cmp", dtype=float)], axis=1)
    return x


def code_matrix(df, feats):
    x = df[feats].copy()
    for c in cat_cols:
        if c in x:
            x[c] = x[c].cat.codes
    return x


def run_cv(name, fit_predict, x, x_test=None, fold_list=None):
    fold_list = fold_list or folds
    oof = np.zeros(len(y))
    test_pred = np.zeros(len(x_test)) if x_test is not None else None
    t0 = time.time()
    aucs = []
    for k, (tr, va) in enumerate(fold_list):
        p_va, p_te = fit_predict(x.iloc[tr], y[tr], x.iloc[va], x_test, k)
        oof[va] = p_va
        if p_te is not None:
            test_pred += p_te / len(fold_list)
        aucs.append(roc_auc_score(y[va], p_va))
        print(f"  {name} fold {k + 1}: auc {aucs[-1]:.4f}")
    res = {"model": name, "oof_auc": roc_auc_score(y, oof), "fold_mean": np.mean(aucs), "fold_std": np.std(aucs),
           "pr_auc": average_precision_score(y, oof), "minutes": (time.time() - t0) / 60}
    print(f"{name}: oof auc {res['oof_auc']:.4f} (folds {res['fold_mean']:.4f} +/- {res['fold_std']:.4f}), {res['minutes']:.1f} min")
    return res, oof, test_pred


results, oofs = [], {}

# %% [markdown]
# ## 5. Models
# ### 5.1 Logistic regression baseline (standardized numeric features + one-hot compound)

# %%
def lr_fit(x_tr, y_tr, x_va, x_te, k):
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    m.fit(x_tr, y_tr)
    return m.predict_proba(x_va)[:, 1], None


for tag, feats in [("raw", raw_feats), ("eng", all_feats)]:
    r, o, _ = run_cv(f"LogReg ({tag})", lr_fit, num_matrix(train_fe, feats))
    results.append(r)
    oofs[r["model"]] = o

# %% [markdown]
# ### 5.2 Random Forest

# %%
def rf_fit(x_tr, y_tr, x_va, x_te, k):
    m = RandomForestClassifier(n_estimators=100 if quick else 300, min_samples_leaf=20, max_features="sqrt",
                               max_samples=0.3, n_jobs=-1, random_state=seed)
    m.fit(x_tr.fillna(-999), y_tr)
    return m.predict_proba(x_va.fillna(-999))[:, 1], None


r, o, _ = run_cv("Random Forest (eng)", rf_fit, code_matrix(train_fe, all_feats))
results.append(r)
oofs[r["model"]] = o

# %% [markdown]
# ### 5.3 LightGBM
# Early stopping uses an inner holdout of training races, so the validation fold is never used for tuning.

# %%
lgb_params = dict(n_estimators=4000, learning_rate=0.05, num_leaves=63, min_child_samples=50, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0, random_state=seed, verbose=-1, n_jobs=-1)
lgb_models = {}


def make_lgb_fit(store=None):
    def fit(x_tr, y_tr, x_va, x_te, k):
        g_tr = groups[x_tr.index]
        inner = StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=seed + k)
        i_tr, i_es = next(inner.split(x_tr, y_tr, g_tr))
        m = lgb.LGBMClassifier(**lgb_params)
        m.fit(x_tr.iloc[i_tr], y_tr[i_tr], eval_set=[(x_tr.iloc[i_es], y_tr[i_es])], eval_metric="auc",
              callbacks=[lgb.early_stopping(100, verbose=False)])
        if store is not None:
            store[k] = m
        p_te = m.predict_proba(x_te)[:, 1] if x_te is not None else None
        return m.predict_proba(x_va)[:, 1], p_te
    return fit


r, o, _ = run_cv("LightGBM (raw)", make_lgb_fit(), train_fe[raw_feats])
results.append(r)
oofs[r["model"]] = o

r, o, lgb_test = run_cv("LightGBM (eng)", make_lgb_fit(lgb_models), train_fe[all_feats], test_fe[all_feats])
results.append(r)
oofs[r["model"]] = o

# %% [markdown]
# ### 5.4 Leakage check: same LightGBM with a random (non-grouped) stratified K-fold

# %%
rand_folds = list(StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(train_fe, y))
r, _, _ = run_cv("LightGBM (eng, random K-fold)", make_lgb_fit(), train_fe[all_feats], fold_list=rand_folds)
results.append(r)

# %% [markdown]
# ## 6. Results

# %%
res = pd.DataFrame(results)
res.to_csv(res_dir / "cv_results.csv", index=False)
print(res.round(4).to_string(index=False))

# %%
fig, ax = plt.subplots(figsize=(5.2, 4.6))
show = [("LogReg (raw)", orange, "--"), ("Random Forest (eng)", aqua, "-"), ("LightGBM (eng)", blue, "-")]
for name, c, ls in show:
    fpr, tpr, _ = roc_curve(y, oofs[name])
    ax.plot(fpr, tpr, color=c, lw=2, ls=ls, label=f"{name}  AUC {roc_auc_score(y, oofs[name]):.3f}")
ax.plot([0, 1], [0, 1], color=grid, lw=1)
ax.set(title="Out-of-fold ROC (race-grouped 5-fold)", xlabel="False positive rate", ylabel="True positive rate")
ax.legend(loc="lower right", frameon=False, fontsize=8)
plt.savefig(fig_dir / "roc_curves.png")
plt.show()

# %%
plot_res = res[~res["model"].str.contains("random")].sort_values("oof_auc")
fig, ax = plt.subplots(figsize=(6.5, 3))
ax.barh(plot_res["model"], plot_res["oof_auc"], color=blue, height=0.55)
for i, v in enumerate(plot_res["oof_auc"]):
    ax.text(v + 0.002, i, f"{v:.4f}", va="center", fontsize=8, color=ink2)
ax.set_xlim(max(0.5, plot_res["oof_auc"].min() - 0.05), 1.0)
ax.set(title="Model comparison, out-of-fold ROC AUC", xlabel="ROC AUC")
plt.savefig(fig_dir / "model_comparison.png")
plt.show()

# %% [markdown]
# ## 7. Most helpful features (LightGBM gain, averaged over folds)

# %%
imp = pd.DataFrame({k: pd.Series(m.booster_.feature_importance("gain"), index=all_feats) for k, m in lgb_models.items()})
imp = imp.mean(axis=1).sort_values(ascending=False)
imp = (imp / imp.sum()).rename("gain_share")
imp.to_csv(res_dir / "feature_importance.csv")


def family(f):
    if f in raw_feats:
        return "raw"
    if f.startswith("field_"):
        return "field context"
    if f in ("typical_stint_len", "tyrelife_vs_typical", "laps_to_typical", "race_laps", "laps_remaining"):
        return "tyre/race context"
    return "lap sequence"


fam = imp.groupby(imp.index.map(family)).sum().sort_values(ascending=False)
fam.to_csv(res_dir / "feature_family_importance.csv")
print(imp.head(20).round(4).to_string())
print(fam.round(4).to_string())

top = imp.head(15)[::-1]
fig, ax = plt.subplots(figsize=(6.5, 4.6))
ax.barh(top.index, top.values * 100, color=[blue if f in raw_feats else aqua for f in top.index], height=0.6)
ax.set(title="Top 15 LightGBM features (share of total gain)", xlabel="% of total gain")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=blue, label="original column"), Patch(color=aqua, label="engineered")],
          frameon=False, fontsize=8, loc="lower right")
plt.savefig(fig_dir / "feature_importance.png")
plt.show()

# %% [markdown]
# ## 8. Save summary and Kaggle submission

# %%
sub = pd.DataFrame({id_col: test_fe[id_col].astype(int), target: lgb_test})
sub.to_csv(res_dir / "submission_lgbm.csv", index=False)
summary = {"eda": eda, "n_features": {"raw": len(raw_feats), "engineered": len(eng_feats)},
           "results": res.round(4).to_dict("records"), "top_features": imp.head(20).round(4).to_dict(),
           "feature_families": fam.round(4).to_dict(), "quick": quick}
with open(res_dir / "summary.json", "w") as f:
    json.dump(summary, f, indent=1, default=str)
print("saved", res_dir / "summary.json")
