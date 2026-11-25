# 13 — LightGBM + patient-context features
#
# Hypothesis: `off` (r=0.87 with target) is missing 42% of the time; the model
# under-predicts high targets. Neighbouring visits of the same patient allow
# reconstruction. All features derived from X only — no target used.
#
# Strategy: concatenate train + test (patients are disjoint), build all
# context features on the combined frame, then split back.
#
# Steps
# -----
# 1. Load data; build EXP 11 base features (22 cols).
# 2. Build patient-context features on train+test concat (no label).
# 3. Quick smoke test (1 fold, 200 trees) to catch bugs before full run.
# 4. Ablation: A=EXP11 base, B=+agg+interp, C=+lag/lead+rolling, D=full set.
# 5. Full CV (GroupKFold 5) with variant D; diagnostics.
# 6. Refit on 100% train (fold-average predictions for stability).
# 7. Save: logs/exp13.txt, oof/exp13_oof.csv, submissions/13_patient_context.csv.
# 8. Update JOURNAL.md.

# ── CAUSAL flag ──────────────────────────────────────────────────────────────
# If True: only use past visits (lags, expanding stats). No lead/future agg.
CAUSAL = False

import io
import sys

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GroupKFold

from parkinson import PROJECT_ROOT
from parkinson.hub import load_skore_credentials

# ── output tee ───────────────────────────────────────────────────────────────
LOG_PATH = PROJECT_ROOT / "logs" / "exp13.txt"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
_log_lines: list[str] = []

def pr(*args) -> None:
    text = " ".join(str(a) for a in args)
    print(text, flush=True)
    _log_lines.append(text)

def save_log() -> None:
    LOG_PATH.write_text("\n".join(_log_lines), encoding="utf-8")
    pr(f"[OK] Log saved -> {LOG_PATH}")

def section(title: str) -> None:
    bar = "=" * 72
    pr(f"\n{bar}\n  {title}\n{bar}")

# ── EXP 11 column sets ────────────────────────────────────────────────────────
NUM_COLS = [
    "sexM", "age_at_diagnosis", "age", "ledd",
    "time_since_intake_on", "time_since_intake_off", "on", "off",
]
DOMAIN_COLS = [
    "rater_bias", "is_off_missing", "is_on_missing",
    "off_x_age", "off_x_time", "ledd_x_age", "on_x_age",
]
TEMPORAL_COLS = [
    "patient_velocity_off", "patient_accel_off",
    "rater_consistency", "visit_month",
]
CAT_COLS     = ["cohort", "gene", "rater_id"]
EXP11_COLS   = NUM_COLS + DOMAIN_COLS + TEMPORAL_COLS + CAT_COLS  # 22 cols

# ── EXP 11 best hyperparameters (Optuna, from logs/11_optuna_exp10_featureset) ─
BEST_PARAMS = {
    "learning_rate":     0.04885694382473504,
    "num_leaves":        50,
    "max_depth":         15,
    "min_child_samples": 40,
    "reg_lambda":        0.9117967494611006,
    "reg_alpha":         0.08968227868317444,
}

# ── LightGBM wrapper ─────────────────────────────────────────────────────────
class LGBMCatRegressor(BaseEstimator, RegressorMixin):
    """LightGBM with categorical_feature baked into fit()."""
    def __init__(self, cat_cols, n_estimators=500, **params):
        self.cat_cols     = cat_cols
        self.n_estimators = n_estimators
        self._params      = params
        for k, v in params.items():
            setattr(self, k, v)

    def fit(self, X, y, X_val=None, y_val=None):
        callbacks = []
        eval_set  = None
        if X_val is not None:
            from lightgbm import early_stopping, log_evaluation
            callbacks = [early_stopping(50, verbose=False), log_evaluation(period=-1)]
            eval_set  = [(X_val, y_val)]
        self.model_ = LGBMRegressor(
            n_estimators=self.n_estimators,
            random_state=42,
            deterministic=True,
            verbose=-1,
            **self._params,
        )
        self.model_.fit(
            X, y,
            categorical_feature=self.cat_cols,
            eval_set=eval_set,
            callbacks=callbacks,
        )
        return self

    def predict(self, X):
        return self.model_.predict(X)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Load raw data
# ─────────────────────────────────────────────────────────────────────────────
section("1 — Load data")
pr("Loading raw CSVs...")

X_train_raw = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv")
y_df        = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv")
X_test_raw  = pd.read_csv(PROJECT_ROOT / "data" / "X_test.csv")

train_raw = X_train_raw.merge(y_df, on="Index")
test_raw  = X_test_raw.copy()

pr(f"  train shape: {train_raw.shape}")
pr(f"  test  shape: {test_raw.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — EXP 11 base feature helpers
# ─────────────────────────────────────────────────────────────────────────────
def _impute_column(df, col, patient_id, fallback_medians=None):
    df = df.copy()
    if df[col].isna().sum() == 0:
        return df
    if fallback_medians is None:
        pat_med = df.groupby(patient_id)[col].transform("median")
        mask    = df[col].isna()
        df.loc[mask, col] = pat_med[mask]
        if df[col].isna().sum() > 0:
            rat_med = df.groupby("rater_id")[col].transform("median")
            mask    = df[col].isna()
            df.loc[mask, col] = rat_med[mask]
        df[col] = df[col].fillna(df[col].median())
    else:
        pm = fallback_medians.get(f"{col}_patient", {})
        rm = fallback_medians.get(f"{col}_rater",   {})
        gm = fallback_medians.get(f"{col}_global",  0.0)
        mask = df[col].isna()
        df.loc[mask, col] = patient_id[mask].map(pm)
        mask = df[col].isna()
        df.loc[mask, col] = df.loc[mask, "rater_id"].map(rm)
        df[col] = df[col].fillna(gm)
    return df


def build_exp11_features(df, patient_id, rater_bias_dict,
                          velocity_dict, accel_dict, rater_consistency_dict,
                          impute_medians=None):
    """Build the 22-col EXP 10/11 feature matrix."""
    out = df[NUM_COLS + CAT_COLS].copy()
    for col in ["off", "on", "time_since_intake_off"]:
        out = _impute_column(out, col, patient_id, impute_medians)
    out["is_off_missing"] = df["off"].isna().astype(np.float32)
    out["is_on_missing"]  = df["on"].isna().astype(np.float32)
    out["rater_bias"]     = out["rater_id"].map(rater_bias_dict).fillna(0.0).clip(-2, 2)
    out["off_x_age"]      = out["off"] * out["age"]
    out["off_x_time"]     = out["off"] * out["time_since_intake_on"]
    out["ledd_x_age"]     = out["ledd"] * out["age"]
    out["on_x_age"]       = out["on"]  * out["age"]
    out["patient_velocity_off"] = patient_id.map(velocity_dict).fillna(0.0)
    out["patient_accel_off"]    = patient_id.map(accel_dict).fillna(0.0)
    out["rater_consistency"]    = out["rater_id"].map(rater_consistency_dict).fillna(0.0)
    out["visit_month"]          = (out["time_since_intake_on"].fillna(0) / 30.0).astype(int) % 12
    for col in CAT_COLS:
        out[col] = out[col].astype("category")
    return out[EXP11_COLS]


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Patient-context feature builder (train+test concat, no label)
# ─────────────────────────────────────────────────────────────────────────────
CONTEXT_VARS = ["on", "off", "ledd", "time_since_intake_on", "time_since_intake_off"]


def build_context_features(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Build all patient-context features on the train+test concatenation.
    df_all must be sorted by (patient_id, age) and have original raw columns.
    Returns a DataFrame indexed identically to df_all.
    No label columns ('target') are used — verified by assertion below.
    """
    assert "target" not in df_all.columns, "LEAKAGE: target found in context frame"

    out = pd.DataFrame(index=df_all.index)
    pid = df_all["patient_id"]

    for v in CONTEXT_VARS:
        col = df_all[v].copy()

        # ── A: patient aggregates ────────────────────────────────────────────
        grp = col.groupby(pid)
        out[f"{v}_pat_mean"]   = grp.transform("mean")
        out[f"{v}_pat_std"]    = grp.transform("std").fillna(0.0)
        out[f"{v}_pat_min"]    = grp.transform("min")
        out[f"{v}_pat_max"]    = grp.transform("max")
        out[f"{v}_pat_median"] = grp.transform("median")
        out[f"{v}_pat_count"]  = grp.transform("count")          # non-null count
        out[f"{v}_dev"]        = col - out[f"{v}_pat_mean"]       # deviation from patient mean

        # ── B: lag / lead ─────────────────────────────────────────────────────
        for shift in [1, 2]:
            lag = col.groupby(pid).shift(shift)
            out[f"{v}_lag{shift}"]      = lag
            out[f"{v}_lag{shift}_diff"] = col - lag
            if not CAUSAL:
                lead = col.groupby(pid).shift(-shift)
                out[f"{v}_lead{shift}"]      = lead
                out[f"{v}_lead{shift}_diff"] = col - lead

        # ── C: rolling mean/std (window=3 centred) ────────────────────────────
        roll_center = not CAUSAL   # causal mode: trailing window only
        rolled = (col.groupby(pid)
                  .transform(lambda s: s.rolling(3, min_periods=1, center=roll_center).mean()))
        out[f"{v}_roll3_mean"] = rolled
        rolled_std = (col.groupby(pid)
                      .transform(lambda s: s.rolling(3, min_periods=1, center=roll_center).std().fillna(0)))
        out[f"{v}_roll3_std"]  = rolled_std

    # ── D: on/off interpolation and slope (label-free) ───────────────────────
    for v in ["on", "off"]:
        col    = df_all[v].copy()
        ages   = df_all["age"]
        interp = np.full(len(df_all), np.nan)
        slope  = np.full(len(df_all), np.nan)
        resid  = np.full(len(df_all), np.nan)

        for pid_val, grp_idx in df_all.groupby("patient_id").groups.items():
            grp_ages = ages.loc[grp_idx].values.astype(float)
            grp_vals = col.loc[grp_idx].values.astype(float)
            valid    = ~np.isnan(grp_vals)

            if valid.sum() >= 2:
                # linear interpolation (np.interp extrapolates via nearest edge)
                interp_vals = np.interp(grp_ages, grp_ages[valid], grp_vals[valid])
                interp[df_all.index.get_indexer(grp_idx)] = interp_vals

                # slope via OLS on valid points
                x  = grp_ages[valid]
                yy = grp_vals[valid]
                sl = float(np.polyfit(x - x.mean(), yy, 1)[0])
                slope[df_all.index.get_indexer(grp_idx)] = sl

                # residuals at all points using interpolated values
                pred_line = sl * (grp_ages - grp_ages[valid].mean()) + yy.mean()
                resid[df_all.index.get_indexer(grp_idx)] = interp_vals - pred_line
            elif valid.sum() == 1:
                interp[df_all.index.get_indexer(grp_idx)] = grp_vals[valid][0]

        out[f"{v}_interp"]      = interp
        out[f"{v}_slope"]       = slope
        out[f"{v}_slope_resid"] = resid

    out["on_off_gap"]           = df_all["off"] - df_all["on"]
    out["on_off_gap_interp"]    = out["off_interp"] - out["on_interp"]

    # ── D: trajectory features ────────────────────────────────────────────────
    out["visit_rank"]  = df_all.groupby("patient_id").cumcount()   # 0-based rank by age
    out["n_visits"]    = df_all.groupby("patient_id")["age"].transform("count")

    age_min = df_all.groupby("patient_id")["age"].transform("min")
    age_max = df_all.groupby("patient_id")["age"].transform("max")
    out["followup_years"]    = df_all["age"] - age_min
    out["total_followup_yrs"]= age_max - age_min

    # ledd slope per patient
    ledd_slope = np.full(len(df_all), np.nan)
    for pid_val, grp_idx in df_all.groupby("patient_id").groups.items():
        ledd_v = df_all.loc[grp_idx, "ledd"].values.astype(float)
        ages_v = df_all.loc[grp_idx, "age"].values.astype(float)
        valid  = ~np.isnan(ledd_v)
        if valid.sum() >= 2:
            sl = float(np.polyfit(ages_v[valid] - ages_v[valid].mean(), ledd_v[valid], 1)[0])
            ledd_slope[df_all.index.get_indexer(grp_idx)] = sl
    out["ledd_per_year"] = ledd_slope

    # proportion of off missing per patient
    out["off_missing_rate"] = (
        df_all["off"].isna().astype(float)
        .groupby(pid).transform("mean")
    )

    # rater visit count
    out["rater_n_visits"] = df_all.groupby("rater_id")["age"].transform("count")

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Compute EXP 11 domain stats (full train only, fold-safe later)
# ─────────────────────────────────────────────────────────────────────────────
section("2 — Pre-compute EXP 11 domain statistics (full train)")

velocity_dict = {}
accel_dict    = {}
for pid_val, grp in train_raw.groupby("patient_id"):
    vals = grp.sort_values("time_since_intake_on")["off"].dropna().values
    if len(vals) >= 3:
        x = np.arange(len(vals))
        velocity_dict[pid_val] = float(np.polyfit(x, vals, 1)[0]) if vals.std() > 0 else 0.0
        accel_dict[pid_val]    = float(np.polyfit(x, vals, 2)[0]) if len(vals) >= 4 and vals.std() > 0 else 0.0
    else:
        velocity_dict[pid_val] = 0.0
        accel_dict[pid_val]    = 0.0

rater_cons_dict = train_raw.groupby("rater_id")["off"].std().fillna(0).to_dict()
rater_bias_dict = train_raw.groupby("rater_id")["target"].mean().to_dict()

impute_medians: dict = {}
for col in ["off", "on", "time_since_intake_off"]:
    impute_medians[f"{col}_patient"] = train_raw.groupby("patient_id")[col].median().to_dict()
    impute_medians[f"{col}_rater"]   = train_raw.groupby("rater_id")[col].median().to_dict()
    impute_medians[f"{col}_global"]  = train_raw[col].median()

pr("  Done.")

# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Build context features on train+test concat
# ─────────────────────────────────────────────────────────────────────────────
section("3 — Build patient-context features (train+test concat)")

train_ctx = train_raw.drop(columns=["target"]).copy()
train_ctx["_is_test"] = 0
test_ctx              = test_raw.copy()
test_ctx["_is_test"]  = 1

all_df = (pd.concat([train_ctx, test_ctx], ignore_index=True)
          .sort_values(["patient_id", "age"])
          .reset_index(drop=True))

pr(f"  Combined shape: {all_df.shape}")
pr("  Building context features (interpolation loops ~30s)...")
t0 = time.time()
ctx_feats = build_context_features(all_df)
pr(f"  Done in {time.time()-t0:.1f}s  —  {ctx_feats.shape[1]} context columns")

# Re-align by Index (original row key)
all_df_indexed = all_df.copy()
all_df_indexed["_ctx_idx"] = np.arange(len(all_df_indexed))
ctx_feats_indexed = ctx_feats.copy()
ctx_feats_indexed.index = all_df_indexed["Index"].values

# Merge back to train/test by Index
train_ctx_merged = (train_raw
                    .merge(all_df_indexed[["Index", "_ctx_idx"]], on="Index")
                    .copy())
ctx_for_train = ctx_feats.iloc[train_ctx_merged["_ctx_idx"].values].copy()
ctx_for_train.index = train_ctx_merged.index

test_ctx_merged = (test_raw
                   .merge(all_df_indexed[["Index", "_ctx_idx"]], on="Index")
                   .copy())
ctx_for_test = ctx_feats.iloc[test_ctx_merged["_ctx_idx"].values].copy()
ctx_for_test.index = test_ctx_merged.index

# Preserve original ordering
train_ctx_merged = train_ctx_merged.sort_values("Index").reset_index(drop=True)
ctx_for_train    = ctx_for_train.loc[train_ctx_merged.index].reset_index(drop=True)
test_ctx_merged  = test_ctx_merged.sort_values("Index").reset_index(drop=True)
ctx_for_test     = ctx_for_test.loc[test_ctx_merged.index].reset_index(drop=True)

pr(f"  train context rows: {len(ctx_for_train)}, test: {len(ctx_for_test)}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Assemble feature variants
# ─────────────────────────────────────────────────────────────────────────────
section("4 — Assemble feature variants for ablation")

# We always need the full EXP11 features (built fold-safely in the CV loop).
# The context columns are split into groups for ablation.
AGG_COLS     = [c for c in ctx_feats.columns if any(
                    c.endswith(s) for s in ["_pat_mean","_pat_std","_pat_min","_pat_max",
                                            "_pat_median","_pat_count","_dev",
                                            "_interp"])]
LAG_ROLL_COLS = [c for c in ctx_feats.columns if any(
                    c.startswith(v) and any(tag in c for tag in ["_lag","_lead","_roll"])
                    for v in CONTEXT_VARS)]
TRAJ_COLS    = [c for c in ctx_feats.columns if c not in AGG_COLS + LAG_ROLL_COLS]

CONTEXT_COLS_B = AGG_COLS
CONTEXT_COLS_C = AGG_COLS + LAG_ROLL_COLS
CONTEXT_COLS_D = list(ctx_feats.columns)  # everything

for name, cols in [("B (agg+interp)", CONTEXT_COLS_B),
                   ("C (B+lag/roll)",  CONTEXT_COLS_C),
                   ("D (full)",        CONTEXT_COLS_D)]:
    pr(f"  Variant {name}: {len(cols)} context cols")

# ─────────────────────────────────────────────────────────────────────────────
# Step 7 — CV runner
# ─────────────────────────────────────────────────────────────────────────────
def run_cv(variant_name: str, context_cols: list[str],
           n_estimators: int = 500, fast: bool = False) -> tuple:
    """
    Run GroupKFold(5) CV for a given context column set.
    Returns (mean_rmse, std_rmse, oof_preds, fold_rmses).
    If fast=True: runs only 1 fold, 200 trees (smoke test).
    """
    visits   = train_raw.copy()
    y_full   = visits["target"].values
    groups   = visits["patient_id"].values
    n        = len(visits)
    cv       = GroupKFold(n_splits=5)
    n_folds  = 1 if fast else 5

    oof      = np.full(n, np.nan)
    fold_rmses: list[float] = []

    pr(f"\n  [{variant_name}] n_estimators={n_estimators}, "
       f"{'FAST 1-fold' if fast else '5-fold GroupKFold'}")

    splits = list(cv.split(visits, y_full, groups=groups))

    for fold_i, (tr_idx, va_idx) in enumerate(splits[:n_folds]):
        t_fold = time.time()

        # EXP 11 features (fold-safe rater_bias)
        fold_rater_bias = visits.iloc[tr_idx].groupby("rater_id")["target"].mean().to_dict()

        X_tr_11 = build_exp11_features(
            visits.iloc[tr_idx], visits.iloc[tr_idx]["patient_id"],
            fold_rater_bias, velocity_dict, accel_dict, rater_cons_dict)
        X_va_11 = build_exp11_features(
            visits.iloc[va_idx], visits.iloc[va_idx]["patient_id"],
            fold_rater_bias, velocity_dict, accel_dict, rater_cons_dict)

        # Context features (pre-built, no label leakage)
        if context_cols:
            ctx_tr = ctx_for_train.iloc[tr_idx][context_cols].reset_index(drop=True)
            ctx_va = ctx_for_train.iloc[va_idx][context_cols].reset_index(drop=True)
            X_tr = pd.concat([X_tr_11.reset_index(drop=True), ctx_tr], axis=1)
            X_va = pd.concat([X_va_11.reset_index(drop=True), ctx_va], axis=1)
        else:
            X_tr = X_tr_11.reset_index(drop=True)
            X_va = X_va_11.reset_index(drop=True)

        # Cast all float context cols (cats already set by build_exp11_features)
        for c in context_cols:
            if c in X_tr.columns and X_tr[c].dtype not in ["category"]:
                X_tr[c] = X_tr[c].astype(float)
                X_va[c] = X_va[c].astype(float)

        est = LGBMCatRegressor(
            cat_cols=CAT_COLS, n_estimators=n_estimators, **BEST_PARAMS
        )
        est.fit(X_tr, y_full[tr_idx],
                X_val=X_va, y_val=y_full[va_idx])

        preds             = est.predict(X_va)
        oof[va_idx]       = preds
        rmse_f            = float(np.sqrt(np.mean((y_full[va_idx] - preds) ** 2)))
        fold_rmses.append(rmse_f)
        pr(f"    fold {fold_i+1}/{n_folds}  RMSE={rmse_f:.4f}  "
           f"({time.time()-t_fold:.0f}s)  n_trees={est.model_.n_estimators_}")

    valid   = ~np.isnan(oof)
    mean_r  = float(np.sqrt(np.mean((y_full[valid] - oof[valid]) ** 2)))
    std_r   = float(np.std(fold_rmses))
    pr(f"  [{variant_name}] OOF RMSE = {mean_r:.4f} ± {std_r:.4f}")
    return mean_r, std_r, oof, fold_rmses


# ─────────────────────────────────────────────────────────────────────────────
# Step 8 — Smoke test (1 fold, 200 trees)
# ─────────────────────────────────────────────────────────────────────────────
section("5 — Smoke test (1 fold, 200 trees)")

smoke_rmse, _, _, _ = run_cv("SMOKE-A (base)", [],        n_estimators=200, fast=True)
smoke_rmse_d, _, _, _= run_cv("SMOKE-D (full)", CONTEXT_COLS_D, n_estimators=200, fast=True)

pr(f"\n  Smoke: base={smoke_rmse:.4f}  full={smoke_rmse_d:.4f}  "
   f"— {'OK' if abs(smoke_rmse - smoke_rmse_d) < 5 else 'WARNING: large diff'}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 9 — Ablation study (full 5-fold)
# ─────────────────────────────────────────────────────────────────────────────
section("6 — Ablation study (GroupKFold 5, n_estimators=500 w/ early stopping)")

ablation_results: dict[str, tuple] = {}
ablation_results["A (EXP11 base)"]    = run_cv("A (EXP11 base)",    [],             n_estimators=500)
ablation_results["B (A+agg+interp)"]  = run_cv("B (A+agg+interp)",  CONTEXT_COLS_B, n_estimators=500)
ablation_results["C (B+lag/roll)"]    = run_cv("C (B+lag/roll)",     CONTEXT_COLS_C, n_estimators=500)
ablation_results["D (full)"]          = run_cv("D (full)",           CONTEXT_COLS_D, n_estimators=500)

section("Ablation summary")
pr(f"  {'Variant':<22} {'OOF RMSE':>10}  {'Std':>8}  {'vs A':>8}")
pr(f"  {'-'*22}  {'-'*8}  {'-'*6}  {'-'*8}")
base_rmse = ablation_results["A (EXP11 base)"][0]
for name, (mean_r, std_r, _, _) in ablation_results.items():
    delta = mean_r - base_rmse
    pr(f"  {name:<22}  {mean_r:.4f}    {std_r:.4f}  {delta:+.4f}")

# Best variant by RMSE
best_variant = min(ablation_results, key=lambda k: ablation_results[k][0])
best_rmse, best_std, best_oof, best_fold_rmses = ablation_results[best_variant]
best_ctx_cols = {"A (EXP11 base)": [], "B (A+agg+interp)": CONTEXT_COLS_B,
                 "C (B+lag/roll)": CONTEXT_COLS_C, "D (full)": CONTEXT_COLS_D}[best_variant]

pr(f"\n  Best variant: {best_variant}  RMSE={best_rmse:.4f} ± {best_std:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 10 — Diagnostics on best variant OOF
# ─────────────────────────────────────────────────────────────────────────────
section("7 — OOF diagnostics (best variant)")

visits   = train_raw.copy()
y_full   = visits["target"].values

diag = visits[["Index", "patient_id", "cohort", "gene", "rater_id", "target", "off"]].copy()
diag["oof_pred"] = best_oof
diag["abs_err"]  = np.abs(diag["target"] - diag["oof_pred"])

valid_mask = ~np.isnan(best_oof)

pr(f"\n  OOF RMSE (global)       : {best_rmse:.4f}")
pr(f"  Per-fold RMSE           : {[round(r,4) for r in best_fold_rmses]}")
pr(f"  Std of OOF preds        : {np.nanstd(best_oof):.3f}  (target std = 16.500)")

# RMSE: off missing vs present (original, before imputation)
off_miss = diag["off"].isna() & valid_mask
off_pres = diag["off"].notna() & valid_mask
rmse_miss = float(np.sqrt(np.mean((diag.loc[off_miss,"target"] - diag.loc[off_miss,"oof_pred"])**2)))
rmse_pres = float(np.sqrt(np.mean((diag.loc[off_pres,"target"] - diag.loc[off_pres,"oof_pred"])**2)))
pr(f"\n  RMSE when off MISSING   : {rmse_miss:.4f}  (n={off_miss.sum():,})")
pr(f"  RMSE when off PRESENT   : {rmse_pres:.4f}  (n={off_pres.sum():,})")

# RMSE by target tranche
pr("\n  RMSE by target tranche:")
for lo, hi, label in [(0,20,"<20"), (20,50,"20–50"), (50,70,"50–70"), (70,200,">70")]:
    sel = (diag["target"] >= lo) & (diag["target"] < hi) & valid_mask
    if sel.sum() == 0:
        continue
    rmse_t = float(np.sqrt(np.mean((diag.loc[sel,"target"] - diag.loc[sel,"oof_pred"])**2)))
    pr(f"    [{label:>7}]  n={sel.sum():6,}  RMSE={rmse_t:.4f}")

# Top-20 worst OOF errors
pr("\n  Top 20 worst OOF errors:")
worst = (diag[valid_mask].nlargest(20, "abs_err")
         [["Index","patient_id","cohort","gene","rater_id","target","oof_pred","abs_err"]]
         .round(3))
pr(worst.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# Step 11 — Feature importances (best variant, refit on full train)
# ─────────────────────────────────────────────────────────────────────────────
section("8 — Feature importances (refit on full train, best variant)")

rater_bias_full = train_raw.groupby("rater_id")["target"].mean().to_dict()
X_full_11 = build_exp11_features(
    train_raw, train_raw["patient_id"],
    rater_bias_full, velocity_dict, accel_dict, rater_cons_dict)

if best_ctx_cols:
    X_full = pd.concat([X_full_11.reset_index(drop=True),
                        ctx_for_train[best_ctx_cols].reset_index(drop=True)], axis=1)
else:
    X_full = X_full_11.reset_index(drop=True)

for c in best_ctx_cols:
    if c in X_full.columns and X_full[c].dtype not in ["category"]:
        X_full[c] = X_full[c].astype(float)

all_feature_cols = EXP11_COLS + best_ctx_cols
pr(f"\n  Total features in best model: {len(all_feature_cols)}")
pr(f"  Feature list: {all_feature_cols}")
assert "target" not in all_feature_cols, "LEAKAGE: target in feature list"

final_model = LGBMCatRegressor(cat_cols=CAT_COLS, n_estimators=500, **BEST_PARAMS)
final_model.fit(X_full, y_full)

imp = (pd.Series(final_model.model_.feature_importances_, index=all_feature_cols)
       .sort_values(ascending=False))
pr("\n  Top 20 feature importances (gain):")
pr(imp.head(20).to_string())

# ─────────────────────────────────────────────────────────────────────────────
# Step 12 — Fold-average predictions on test set
# ─────────────────────────────────────────────────────────────────────────────
section("9 — Test predictions (fold-average of 5 models)")

X_test_11 = build_exp11_features(
    test_raw, test_raw["patient_id"],
    rater_bias_full, velocity_dict, accel_dict, rater_cons_dict,
    impute_medians)

if best_ctx_cols:
    X_test_full = pd.concat([X_test_11.reset_index(drop=True),
                              ctx_for_test[best_ctx_cols].reset_index(drop=True)], axis=1)
else:
    X_test_full = X_test_11.reset_index(drop=True)

for c in best_ctx_cols:
    if c in X_test_full.columns and X_test_full[c].dtype not in ["category"]:
        X_test_full[c] = X_test_full[c].astype(float)

pr("  Refitting 5 fold models for averaging...")
fold_test_preds = np.zeros(len(test_raw))
cv    = GroupKFold(n_splits=5)
groups = train_raw["patient_id"].values

for fold_i, (tr_idx, va_idx) in enumerate(cv.split(train_raw, y_full, groups=groups)):
    t_fold = time.time()
    fold_rater_bias = train_raw.iloc[tr_idx].groupby("rater_id")["target"].mean().to_dict()

    X_tr_11 = build_exp11_features(
        train_raw.iloc[tr_idx], train_raw.iloc[tr_idx]["patient_id"],
        fold_rater_bias, velocity_dict, accel_dict, rater_cons_dict)

    if best_ctx_cols:
        ctx_tr = ctx_for_train.iloc[tr_idx][best_ctx_cols].reset_index(drop=True)
        X_tr   = pd.concat([X_tr_11.reset_index(drop=True), ctx_tr], axis=1)
    else:
        X_tr = X_tr_11.reset_index(drop=True)

    for c in best_ctx_cols:
        if c in X_tr.columns and X_tr[c].dtype not in ["category"]:
            X_tr[c] = X_tr[c].astype(float)

    # val set for early stopping
    X_va_11 = build_exp11_features(
        train_raw.iloc[va_idx], train_raw.iloc[va_idx]["patient_id"],
        fold_rater_bias, velocity_dict, accel_dict, rater_cons_dict)
    if best_ctx_cols:
        ctx_va = ctx_for_train.iloc[va_idx][best_ctx_cols].reset_index(drop=True)
        X_va   = pd.concat([X_va_11.reset_index(drop=True), ctx_va], axis=1)
    else:
        X_va = X_va_11.reset_index(drop=True)
    for c in best_ctx_cols:
        if c in X_va.columns and X_va[c].dtype not in ["category"]:
            X_va[c] = X_va[c].astype(float)

    m = LGBMCatRegressor(cat_cols=CAT_COLS, n_estimators=500, **BEST_PARAMS)
    m.fit(X_tr, y_full[tr_idx], X_val=X_va, y_val=y_full[va_idx])
    fold_test_preds += m.predict(X_test_full) / 5
    pr(f"  fold {fold_i+1}/5 done ({time.time()-t_fold:.0f}s, "
       f"n_trees={m.model_.n_estimators_})")

# ─────────────────────────────────────────────────────────────────────────────
# Step 13 — Save outputs
# ─────────────────────────────────────────────────────────────────────────────
section("10 — Save outputs")

# OOF CSV
oof_dir = PROJECT_ROOT / "oof"
oof_dir.mkdir(parents=True, exist_ok=True)
oof_df  = pd.DataFrame({
    "Index":    train_raw["Index"].values,
    "target":   y_full,
    "oof_pred": best_oof,
})
oof_path = oof_dir / "exp13_oof.csv"
oof_df.to_csv(oof_path, index=False)
pr(f"  OOF saved -> {oof_path}")

# Submission CSV
sub_dir  = PROJECT_ROOT / "submissions"
sub_dir.mkdir(parents=True, exist_ok=True)
sub_path = sub_dir / "13_patient_context.csv"
sub_df   = pd.DataFrame({"Id": test_raw["Index"].values, "y_pred": fold_test_preds})
sub_df.to_csv(sub_path, index=False)
pr(f"  Submission saved -> {sub_path}  ({len(sub_df)} rows)")
pr(f"  Prediction stats: mean={fold_test_preds.mean():.3f}  "
   f"std={fold_test_preds.std():.3f}  "
   f"min={fold_test_preds.min():.3f}  max={fold_test_preds.max():.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 14 — 10-line summary
# ─────────────────────────────────────────────────────────────────────────────
section("11 — Summary")

a_rmse = ablation_results["A (EXP11 base)"][0]
b_rmse = ablation_results["B (A+agg+interp)"][0]
c_rmse = ablation_results["C (B+lag/roll)"][0]
d_rmse = ablation_results["D (full)"][0]

pr(f"  1. EXP 11 baseline reproduced (variant A): RMSE = {a_rmse:.4f}")
pr(f"  2. Variant B (+agg+interp)  : RMSE = {b_rmse:.4f}  (Δ {b_rmse-a_rmse:+.4f})")
pr(f"  3. Variant C (+lag/lead/roll): RMSE = {c_rmse:.4f}  (Δ {c_rmse-a_rmse:+.4f})")
pr(f"  4. Variant D (full set)      : RMSE = {d_rmse:.4f}  (Δ {d_rmse-a_rmse:+.4f})")
pr(f"  5. Best variant: {best_variant}  →  {best_rmse:.4f} ± {best_std:.4f}")
pr(f"  6. Top-5 new features: {list(imp.index[:5])}")
pr(f"  7. RMSE off-missing: {rmse_miss:.4f}  vs off-present: {rmse_pres:.4f}  "
   f"(gap = {rmse_miss - rmse_pres:+.4f})")
pr(f"  8. OOF std = {np.nanstd(best_oof):.3f} vs target std = 16.500")
pr(f"  9. Submission: {sub_path.name}  ({len(sub_df)} predictions)")
pr(f" 10. EXP 14 recommendation: "
   + ("Patient-context features help — try deeper Optuna on D feature set."
      if best_rmse < a_rmse else
      "Context features didn't help on CV — focus on rater LOO encoding (B06) or stacking."))

# ─────────────────────────────────────────────────────────────────────────────
# Step 15 — Update JOURNAL.md
# ─────────────────────────────────────────────────────────────────────────────
section("12 — Update JOURNAL.md")

journal_path = PROJECT_ROOT / "journal" / "JOURNAL.md"
journal_text = journal_path.read_text(encoding="utf-8")

# Add row to experiments table
new_row = (f"| 13 | patient_context | {best_rmse:.4f} ± {best_std:.4f} | — | — | "
           f"LGB + patient-context features (agg/interp/lag/traj); best variant: {best_variant} |")

table_end_marker = "| 12 |"
if "| 13 |" not in journal_text:
    journal_text = journal_text.replace(
        table_end_marker,
        new_row + "\n" + table_end_marker,
        1
    )

# Update best model if improved
if best_rmse < 6.5695:
    journal_text = journal_text.replace(
        "EXP 11 — LightGBM Optuna v2 on 22-col feature set — CV RMSE **6.5695**",
        f"EXP 13 — LGB patient-context — CV RMSE **{best_rmse:.4f}**",
    )

# Add experiment detail section
detail = f"""
### 13 — patient_context

- **Status:** done
- **Splitter:** GroupKFold(n_splits=5, groups=patient_id) — fold-safe rater_bias
- **Model:** LGBMCatRegressor (EXP 11 best params + early stopping)
- **Best variant:** {best_variant}
- **CV RMSE:** **{best_rmse:.4f} ± {best_std:.4f}**  (EXP 11 baseline: 6.5695)
- **Features:** {len(all_feature_cols)} total ({len(EXP11_COLS)} EXP11 + {len(best_ctx_cols)} context)
- **Ablation:** A={a_rmse:.4f}  B={b_rmse:.4f}  C={c_rmse:.4f}  D={d_rmse:.4f}
- **RMSE off-missing:** {rmse_miss:.4f}  vs off-present: {rmse_pres:.4f}
- **Submission:** `submissions/13_patient_context.csv`
- **OOF file:** `oof/exp13_oof.csv`

---
"""

if "### 13 —" not in journal_text:
    journal_text = journal_text.replace(
        "## Backlog — next models to try",
        detail + "## Backlog — next models to try",
    )

journal_path.write_text(journal_text, encoding="utf-8")
pr(f"  JOURNAL.md updated -> {journal_path}")

# Save log last (after all output accumulated)
save_log()
