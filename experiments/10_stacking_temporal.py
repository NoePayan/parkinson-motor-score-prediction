# 10 — Stacking ensemble + temporal features
#
# Three diverse base learners (LightGBM, HistGBR, Ridge) stacked with a Ridge
# meta-learner. Extends EXP 09 domain features with 4 temporal features:
#   - patient_velocity_off : linear slope of OFF scores over visit sequence
#   - patient_accel_off    : quadratic coefficient of OFF trajectory
#   - rater_consistency    : per-rater std of 'off' scores
#   - visit_month          : time_since_intake_on / 30 % 12
#
# Stacking CV: out-of-fold predictions from GroupKFold(5) fed to meta-Ridge.
# rater_bias is recomputed per fold to stay honest.
#
# Steps
# -----
# 1. Load data; compute temporal + domain statistics from full train.
# 2. Build LGB and numeric feature matrices (22 cols + OHE variant for Ridge).
# 3. Manual stacking CV (GroupKFold 5-fold) — OOF predictions per base learner.
# 4. Fit meta-Ridge on OOF stack; report stacking CV RMSE ± std.
# 5. Fit all base learners + meta on 100 % of training data.
# 6. Predict on X_test and write submissions/10_stacking_temporal.csv.
# 7. Push EstimatorReport (train[:2000] / test[:500]) to Skore Hub.

import io
import sys

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import skore
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from parkinson import PROJECT_ROOT
from parkinson.hub import load_skore_credentials

# ---------------------------------------------------------------------------
# Best hyperparams from EXP 08 Optuna
# ---------------------------------------------------------------------------
BEST_PARAMS = {
    "learning_rate":     0.04885694382473504,
    "num_leaves":        50,
    "max_depth":         15,
    "min_child_samples": 40,
    "reg_lambda":        0.9117967494611006,
    "reg_alpha":         0.08968227868317444,
}

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
CAT_COLS = ["cohort", "gene", "rater_id"]
ALL_NUM_COLS = NUM_COLS + DOMAIN_COLS + TEMPORAL_COLS
FEATURE_COLS = ALL_NUM_COLS + CAT_COLS


# ---------------------------------------------------------------------------
# LightGBM wrapper (EXP 06-09 pattern)
# ---------------------------------------------------------------------------
class LGBMCatRegressor(BaseEstimator, RegressorMixin):
    """LightGBM with categorical_feature baked into fit()."""

    def __init__(self, cat_cols, n_estimators=150, learning_rate=0.05,
                 num_leaves=31, max_depth=-1, min_child_samples=20,
                 reg_lambda=0.0, reg_alpha=0.0, random_state=42, verbose=-1):
        self.cat_cols = cat_cols
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        self.model_ = LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            min_child_samples=self.min_child_samples,
            reg_lambda=self.reg_lambda,
            reg_alpha=self.reg_alpha,
            random_state=self.random_state,
            verbose=self.verbose,
        )
        self.model_.fit(X, y, categorical_feature=self.cat_cols)
        return self

    def predict(self, X):
        return self.model_.predict(X)


# ---------------------------------------------------------------------------
# Feature engineering helpers (from EXP 09 + new temporal)
# ---------------------------------------------------------------------------
def _impute_column(df, col, patient_id, fallback_medians=None):
    df = df.copy()
    if df[col].isna().sum() == 0:
        return df
    if fallback_medians is None:
        pat_med = df.groupby(patient_id)[col].transform("median")
        mask = df[col].isna()
        df.loc[mask, col] = pat_med[mask]
        if df[col].isna().sum() > 0:
            rater_med = df.groupby("rater_id")[col].transform("median")
            mask = df[col].isna()
            df.loc[mask, col] = rater_med[mask]
        df[col] = df[col].fillna(df[col].median())
    else:
        pat_med_map  = fallback_medians.get(f"{col}_patient", {})
        rater_med_map = fallback_medians.get(f"{col}_rater", {})
        global_med   = fallback_medians.get(f"{col}_global", 0.0)
        mask = df[col].isna()
        df.loc[mask, col] = patient_id[mask].map(pat_med_map)
        mask = df[col].isna()
        df.loc[mask, col] = df.loc[mask, "rater_id"].map(rater_med_map)
        df[col] = df[col].fillna(global_med)
    return df


def build_features(df, patient_id, rater_bias_dict,
                   velocity_dict, accel_dict, rater_consistency_dict,
                   impute_medians=None, for_lgb=True):
    """Build full feature matrix. for_lgb=True casts cat cols to 'category'."""
    out = df[NUM_COLS + CAT_COLS].copy()

    for col in ["off", "on", "time_since_intake_off"]:
        out = _impute_column(out, col, patient_id, impute_medians)

    # Missing indicators
    out["is_off_missing"] = df["off"].isna().astype(np.float32)
    out["is_on_missing"]  = df["on"].isna().astype(np.float32)

    # Rater bias
    out["rater_bias"] = out["rater_id"].map(rater_bias_dict).fillna(0.0).clip(-2, 2)

    # Interactions
    out["off_x_age"]  = out["off"] * out["age"]
    out["off_x_time"] = out["off"] * out["time_since_intake_on"]
    out["ledd_x_age"] = out["ledd"] * out["age"]
    out["on_x_age"]   = out["on"]  * out["age"]

    # Temporal features
    out["patient_velocity_off"] = patient_id.map(velocity_dict).fillna(0.0)
    out["patient_accel_off"]    = patient_id.map(accel_dict).fillna(0.0)
    out["rater_consistency"]    = out["rater_id"].map(rater_consistency_dict).fillna(0.0)
    out["visit_month"]          = (out["time_since_intake_on"].fillna(0) / 30.0).astype(int) % 12

    if for_lgb:
        for col in CAT_COLS:
            out[col] = out[col].astype("category")
    else:
        # For Ridge / HistGBR: OHE cats, result is purely numeric
        dummies = pd.get_dummies(out[CAT_COLS], drop_first=True, dtype=float)
        out = pd.concat([out[ALL_NUM_COLS], dummies], axis=1)

    return out[FEATURE_COLS] if for_lgb else out


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("Step 1: loading data...")
sys.stdout.flush()

X_train_raw = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv")
y_df        = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv")
X_test_raw  = pd.read_csv(PROJECT_ROOT / "data" / "X_test.csv")

visits = X_train_raw.merge(y_df, on="Index")
patient_id_train = visits["patient_id"]
y_full = visits["target"].values

print(f"  visits shape: {visits.shape}")
sys.stdout.flush()

# ---------------------------------------------------------------------------
# 2. Compute per-patient temporal features
# ---------------------------------------------------------------------------
print("Step 2: computing temporal + domain features...")
sys.stdout.flush()

velocity_dict = {}
accel_dict = {}
for pid, grp in visits.groupby("patient_id"):
    grp_s = grp.sort_values("time_since_intake_on")
    vals = grp_s["off"].dropna().values
    if len(vals) >= 3:
        x = np.arange(len(vals))
        velocity_dict[pid] = float(np.polyfit(x, vals, 1)[0]) if vals.std() > 0 else 0.0
        accel_dict[pid] = float(np.polyfit(x, vals, 2)[0]) if len(vals) >= 4 and vals.std() > 0 else 0.0
    else:
        velocity_dict[pid] = 0.0
        accel_dict[pid] = 0.0

rater_consistency_dict = visits.groupby("rater_id")["off"].std().fillna(0).to_dict()
rater_bias_dict        = visits.groupby("rater_id")["target"].mean().to_dict()

impute_medians = {}
for col in ["off", "on", "time_since_intake_off"]:
    impute_medians[f"{col}_patient"] = visits.groupby("patient_id")[col].median().to_dict()
    impute_medians[f"{col}_rater"]   = visits.groupby("rater_id")[col].median().to_dict()
    impute_medians[f"{col}_global"]  = visits[col].median()

print(f"  velocity non-zero: {sum(v != 0 for v in velocity_dict.values())}")
sys.stdout.flush()

# ---------------------------------------------------------------------------
# 3. Build full train feature matrices
# ---------------------------------------------------------------------------
X_lgb  = build_features(visits, patient_id_train, rater_bias_dict,
                         velocity_dict, accel_dict, rater_consistency_dict,
                         for_lgb=True)
X_num  = build_features(visits, patient_id_train, rater_bias_dict,
                         velocity_dict, accel_dict, rater_consistency_dict,
                         for_lgb=False)

print(f"  X_lgb shape : {X_lgb.shape} (LightGBM)")
print(f"  X_num shape : {X_num.shape} (Ridge/HistGBR after OHE)")
sys.stdout.flush()

# ---------------------------------------------------------------------------
# 4. Manual stacking with GroupKFold (so we control groups)
# ---------------------------------------------------------------------------
print("Step 3: manual stacking CV (GroupKFold n_splits=5)...")
sys.stdout.flush()

cv = GroupKFold(n_splits=5)
groups = patient_id_train.values
n = len(y_full)

# Out-of-fold prediction arrays for each base learner
oof_lgb  = np.zeros(n)
oof_hgb  = np.zeros(n)
oof_ridge = np.zeros(n)

fold_rmse_lgb   = []
fold_rmse_hgb   = []
fold_rmse_ridge = []

for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(X_lgb, y_full, groups=groups)):
    fold_visits_tr = visits.iloc[tr_idx]
    fold_visits_va = visits.iloc[va_idx]

    # Recompute rater_bias on fold-train only (no leakage)
    fold_rater_bias = fold_visits_tr.groupby("rater_id")["target"].mean().to_dict()

    X_tr_lgb = build_features(fold_visits_tr, fold_visits_tr["patient_id"],
                               fold_rater_bias, velocity_dict, accel_dict,
                               rater_consistency_dict, for_lgb=True)
    X_va_lgb = build_features(fold_visits_va, fold_visits_va["patient_id"],
                               fold_rater_bias, velocity_dict, accel_dict,
                               rater_consistency_dict, for_lgb=True)
    X_tr_num = build_features(fold_visits_tr, fold_visits_tr["patient_id"],
                               fold_rater_bias, velocity_dict, accel_dict,
                               rater_consistency_dict, for_lgb=False)
    X_va_num = build_features(fold_visits_va, fold_visits_va["patient_id"],
                               fold_rater_bias, velocity_dict, accel_dict,
                               rater_consistency_dict, for_lgb=False)

    y_tr = y_full[tr_idx]
    y_va = y_full[va_idx]

    # LightGBM
    lgb_est = LGBMCatRegressor(cat_cols=CAT_COLS, n_estimators=150,
                                random_state=42, verbose=-1, **BEST_PARAMS)
    lgb_est.fit(X_tr_lgb, y_tr)
    oof_lgb[va_idx] = lgb_est.predict(X_va_lgb)
    fold_rmse_lgb.append(np.sqrt(np.mean((y_va - oof_lgb[va_idx])**2)))

    # HistGBR (handles NaN natively, categoricals via dtype)
    hgb_est = HistGradientBoostingRegressor(
        max_iter=150, learning_rate=0.05, max_depth=6,
        l2_regularization=1.0, random_state=42,
        categorical_features=CAT_COLS,
    )
    hgb_est.fit(X_tr_lgb, y_tr)  # uses category dtype cols
    oof_hgb[va_idx] = hgb_est.predict(X_va_lgb)
    fold_rmse_hgb.append(np.sqrt(np.mean((y_va - oof_hgb[va_idx])**2)))

    # Ridge (numeric only, imputer inside pipeline)
    ridge_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("reg", Ridge(alpha=1.0)),
    ])
    ridge_pipe.fit(X_tr_num, y_tr)
    oof_ridge[va_idx] = ridge_pipe.predict(X_va_num)
    fold_rmse_ridge.append(np.sqrt(np.mean((y_va - oof_ridge[va_idx])**2)))

    print(f"  fold {fold_idx+1}/5 — LGB: {fold_rmse_lgb[-1]:.4f} | "
          f"HGB: {fold_rmse_hgb[-1]:.4f} | Ridge: {fold_rmse_ridge[-1]:.4f}")
    sys.stdout.flush()

print(f"\n  Base learner mean RMSE:")
print(f"    LightGBM : {np.mean(fold_rmse_lgb):.4f} +/- {np.std(fold_rmse_lgb):.4f}")
print(f"    HistGBR  : {np.mean(fold_rmse_hgb):.4f} +/- {np.std(fold_rmse_hgb):.4f}")
print(f"    Ridge    : {np.mean(fold_rmse_ridge):.4f} +/- {np.std(fold_rmse_ridge):.4f}")
sys.stdout.flush()

# ---------------------------------------------------------------------------
# 5. Fit meta-learner on OOF predictions
# ---------------------------------------------------------------------------
print("\nStep 4: fitting meta-learner (Ridge) on OOF stack...")
sys.stdout.flush()

S_train = np.column_stack([oof_lgb, oof_hgb, oof_ridge])
meta = Ridge(alpha=0.1)
meta.fit(S_train, y_full)

y_meta_oof = meta.predict(S_train)
meta_rmse  = np.sqrt(np.mean((y_full - y_meta_oof)**2))
print(f"  Meta Ridge OOF RMSE (train refit, optimistic): {meta_rmse:.4f}")
print(f"  Meta coefs — LGB: {meta.coef_[0]:.4f}, HGB: {meta.coef_[1]:.4f}, Ridge: {meta.coef_[2]:.4f}")
sys.stdout.flush()

# Proper stacking CV RMSE: use fold-level meta predictions
fold_meta_rmses = []
for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(X_lgb, y_full, groups=groups)):
    S_tr = S_train[tr_idx]
    S_va = S_train[va_idx]
    m = Ridge(alpha=0.1)
    m.fit(S_tr, y_full[tr_idx])
    pred = m.predict(S_va)
    fold_meta_rmses.append(np.sqrt(np.mean((y_full[va_idx] - pred)**2)))

mean_stack_rmse = np.mean(fold_meta_rmses)
std_stack_rmse  = np.std(fold_meta_rmses)
print(f"  Stacking CV RMSE (proper): {mean_stack_rmse:.4f} +/- {std_stack_rmse:.4f}")
sys.stdout.flush()

# ---------------------------------------------------------------------------
# 6. Fit final models on 100 % of training data
# ---------------------------------------------------------------------------
print("\nStep 5: fitting all models on 100 % of training data...")
sys.stdout.flush()

final_lgb = LGBMCatRegressor(cat_cols=CAT_COLS, n_estimators=150,
                              random_state=42, verbose=-1, **BEST_PARAMS)
final_lgb.fit(X_lgb, y_full)

final_hgb = HistGradientBoostingRegressor(
    max_iter=150, learning_rate=0.05, max_depth=6,
    l2_regularization=1.0, random_state=42,
    categorical_features=CAT_COLS,
)
final_hgb.fit(X_lgb, y_full)

final_ridge = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("reg", Ridge(alpha=1.0)),
])
final_ridge.fit(X_num, y_full)

print("  Done.")
sys.stdout.flush()

# ---------------------------------------------------------------------------
# 7. Predict on X_test and save submission
# ---------------------------------------------------------------------------
print("Step 6: predicting on X_test and saving submission CSV...")
sys.stdout.flush()

X_test_lgb = build_features(X_test_raw, X_test_raw["patient_id"],
                             rater_bias_dict, velocity_dict, accel_dict,
                             rater_consistency_dict, impute_medians, for_lgb=True)
X_test_num = build_features(X_test_raw, X_test_raw["patient_id"],
                             rater_bias_dict, velocity_dict, accel_dict,
                             rater_consistency_dict, impute_medians, for_lgb=False)

# Reindex numeric matrix to match train columns
X_test_num = X_test_num.reindex(columns=X_num.columns, fill_value=0.0)

pred_lgb   = np.array(final_lgb.predict(X_test_lgb))
pred_hgb   = np.array(final_hgb.predict(X_test_lgb))
pred_ridge = np.array(final_ridge.predict(X_test_num))

S_test = np.column_stack([pred_lgb, pred_hgb, pred_ridge])
y_pred = meta.predict(S_test)

sub_path = PROJECT_ROOT / "submissions" / "10_stacking_temporal.csv"
submission = pd.DataFrame({"Id": X_test_raw["Index"], "y_pred": y_pred})
submission.to_csv(sub_path, index=False)

print(f"  Saved {len(submission)} predictions -> {sub_path}")
print(f"  Prediction sample (first 3): {y_pred[:3].round(4).tolist()}")
sys.stdout.flush()

# ---------------------------------------------------------------------------
# 8. Push EstimatorReport to Skore Hub (EXP 06-09 pattern)
#    Pass unfitted LGBMCatRegressor (best single model) with X_train/y_train sample
# ---------------------------------------------------------------------------
print("Step 7: pushing EstimatorReport to Skore Hub...")
sys.stdout.flush()

try:
    cfg = load_skore_credentials()
    skore.login(mode="hub")

    project = skore.Project(
        name="bobathon-esilv",
        mode="hub",
        workspace=cfg["workspace"],
    )

    idx_tr, idx_te = train_test_split(
        np.arange(len(X_lgb)), test_size=0.2, random_state=42
    )
    X_tr_s = X_lgb.iloc[idx_tr[:2000]]
    y_tr_s = y_full[idx_tr[:2000]]
    X_te_s = X_lgb.iloc[idx_te[:500]]
    y_te_s = y_full[idx_te[:500]]

    report = skore.EstimatorReport(
        estimator=LGBMCatRegressor(
            cat_cols=CAT_COLS, n_estimators=150, random_state=42, verbose=-1,
            **BEST_PARAMS,
        ),
        X_train=X_tr_s,
        y_train=y_tr_s,
        X_test=X_te_s,
        y_test=y_te_s,
    )

    project.put("10_stacking_temporal", report)

    print("[OK] Pushed EstimatorReport to SkoreHub")
    sys.stdout.flush()
    print("[REPORT] https://skore.probabl.ai/projects/bobathon-esilv/items/10_stacking_temporal")
    sys.stdout.flush()
except Exception as exc:
    print(f"[WARN] Skore Hub push failed: {exc}")
    sys.stdout.flush()

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
print("\n=== Final results ===")
print(f"  Base LightGBM   : {np.mean(fold_rmse_lgb):.4f} +/- {np.std(fold_rmse_lgb):.4f}")
print(f"  Base HistGBR    : {np.mean(fold_rmse_hgb):.4f} +/- {np.std(fold_rmse_hgb):.4f}")
print(f"  Base Ridge      : {np.mean(fold_rmse_ridge):.4f} +/- {np.std(fold_rmse_ridge):.4f}")
print(f"  Stacking CV RMSE: {mean_stack_rmse:.4f} +/- {std_stack_rmse:.4f}")
print(f"  Meta coefs      : LGB={meta.coef_[0]:.4f} HGB={meta.coef_[1]:.4f} Ridge={meta.coef_[2]:.4f}")
print(f"  Submission      : {sub_path}")
sys.stdout.flush()
