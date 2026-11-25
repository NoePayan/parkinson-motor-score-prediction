# 11 — LightGBM Optuna v2 on the full 22-col feature set
#
# EXP 08 tuned on 11 features (8 numeric + 3 cats).
# This experiment re-tunes on the full EXP 10 feature set:
#   8 numeric + 7 domain (EXP 09) + 4 temporal (EXP 10) + 3 cats = 22 cols
#
# rater_bias is recomputed inside each fold to avoid target leakage.
# Best params from this run are the starting point for EXP 12 (XGBoost).
#
# Steps
# -----
# 1. Load data; compute domain + temporal statistics from full train.
# 2. Build 22-col feature matrix.
# 3. Run Optuna study (30 trials, GroupKFold 5-fold, fold-safe rater_bias).
# 4. Fit final model with best params on 100 % of training data.
# 5. Print top-10 feature importances.
# 6. Predict on X_test and write submissions/11_optuna_exp10_featureset.csv.
# 7. Push EstimatorReport (train[:2000] / test[:500]) to Skore Hub.

import io
import sys

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import optuna
import pandas as pd
import skore
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GroupKFold, cross_val_score, train_test_split

from parkinson import PROJECT_ROOT
from parkinson.hub import load_skore_credentials

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Column definitions (same as EXP 10)
# ---------------------------------------------------------------------------
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
FEATURE_COLS = NUM_COLS + DOMAIN_COLS + TEMPORAL_COLS + CAT_COLS  # 22 cols


# ---------------------------------------------------------------------------
# LightGBM wrapper (EXP 06-10 pattern)
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
# Feature engineering helpers (copied from EXP 09/10)
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
        pat_med_map   = fallback_medians.get(f"{col}_patient", {})
        rater_med_map = fallback_medians.get(f"{col}_rater", {})
        global_med    = fallback_medians.get(f"{col}_global", 0.0)
        mask = df[col].isna()
        df.loc[mask, col] = patient_id[mask].map(pat_med_map)
        mask = df[col].isna()
        df.loc[mask, col] = df.loc[mask, "rater_id"].map(rater_med_map)
        df[col] = df[col].fillna(global_med)
    return df


def build_features(df, patient_id, rater_bias_dict,
                   velocity_dict, accel_dict, rater_consistency_dict,
                   impute_medians=None):
    """Build 22-col feature matrix (EXP 10 set)."""
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

    return out[FEATURE_COLS]


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
# 2. Compute per-patient / per-rater statistics (from full train)
# ---------------------------------------------------------------------------
print("Step 2: computing domain + temporal feature statistics...")
sys.stdout.flush()

velocity_dict = {}
accel_dict = {}
for pid, grp in visits.groupby("patient_id"):
    vals = grp.sort_values("time_since_intake_on")["off"].dropna().values
    if len(vals) >= 3:
        x = np.arange(len(vals))
        velocity_dict[pid] = float(np.polyfit(x, vals, 1)[0]) if vals.std() > 0 else 0.0
        accel_dict[pid]    = float(np.polyfit(x, vals, 2)[0]) if len(vals) >= 4 and vals.std() > 0 else 0.0
    else:
        velocity_dict[pid] = 0.0
        accel_dict[pid]    = 0.0

rater_consistency_dict = visits.groupby("rater_id")["off"].std().fillna(0).to_dict()
rater_bias_dict        = visits.groupby("rater_id")["target"].mean().to_dict()

impute_medians = {}
for col in ["off", "on", "time_since_intake_off"]:
    impute_medians[f"{col}_patient"] = visits.groupby("patient_id")[col].median().to_dict()
    impute_medians[f"{col}_rater"]   = visits.groupby("rater_id")[col].median().to_dict()
    impute_medians[f"{col}_global"]  = visits[col].median()

# ---------------------------------------------------------------------------
# 3. Build full feature matrix
# ---------------------------------------------------------------------------
print("Step 3: building feature matrix...")
sys.stdout.flush()

X_full = build_features(visits, patient_id_train, rater_bias_dict,
                         velocity_dict, accel_dict, rater_consistency_dict)
groups = patient_id_train.values

print(f"  X shape: {X_full.shape}, columns: {FEATURE_COLS}")
sys.stdout.flush()

# ---------------------------------------------------------------------------
# 4. Optuna objective (fold-safe rater_bias per fold)
# ---------------------------------------------------------------------------
cv = GroupKFold(n_splits=5)


def objective(trial: optuna.Trial) -> float:
    params = {
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves":        trial.suggest_int("num_leaves", 20, 100),
        "max_depth":         trial.suggest_int("max_depth", 5, 20),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 2.0, log=True),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
    }

    # Manual fold loop so rater_bias is refitted per fold (no leakage)
    fold_rmses = []
    for tr_idx, va_idx in cv.split(X_full, y_full, groups=groups):
        fold_visits_tr = visits.iloc[tr_idx]
        fold_visits_va = visits.iloc[va_idx]
        fold_rater_bias = fold_visits_tr.groupby("rater_id")["target"].mean().to_dict()

        X_tr = build_features(fold_visits_tr, fold_visits_tr["patient_id"],
                               fold_rater_bias, velocity_dict, accel_dict,
                               rater_consistency_dict)
        X_va = build_features(fold_visits_va, fold_visits_va["patient_id"],
                               fold_rater_bias, velocity_dict, accel_dict,
                               rater_consistency_dict)

        est = LGBMCatRegressor(
            cat_cols=CAT_COLS, n_estimators=150, random_state=42, verbose=-1,
            **params,
        )
        est.fit(X_tr, y_full[tr_idx])
        preds = est.predict(X_va)
        fold_rmses.append(np.sqrt(np.mean((y_full[va_idx] - preds) ** 2)))

    return float(np.mean(fold_rmses))


# ---------------------------------------------------------------------------
# Step 4: Run Optuna study (30 trials)
# ---------------------------------------------------------------------------
print("Step 4: running Optuna study (30 trials on 22-col feature set)...")
sys.stdout.flush()

study = optuna.create_study(
    direction="minimize",
    pruner=optuna.pruners.MedianPruner(),
)
study.optimize(objective, n_trials=30, n_jobs=1, show_progress_bar=True)

best_params  = study.best_params
best_cv_rmse = study.best_value

print(f"  Best CV RMSE : {best_cv_rmse:.4f}")
print(f"  Best params  : {best_params}")
sys.stdout.flush()

# ---------------------------------------------------------------------------
# Step 5: Fit final model with best params on 100 % of training data
# ---------------------------------------------------------------------------
print("Step 5: fitting final model with best params on 100 % of training data...")
sys.stdout.flush()

best_model = LGBMCatRegressor(
    cat_cols=CAT_COLS, n_estimators=150, random_state=42, verbose=-1,
    **best_params,
)
best_model.fit(X_full, y_full)
print("  Done.")
sys.stdout.flush()

# Feature importances
feat_imp = pd.Series(
    best_model.model_.feature_importances_, index=FEATURE_COLS
).sort_values(ascending=False)
print("  Top-10 feature importances:")
print(feat_imp.head(10).to_string())
sys.stdout.flush()

# ---------------------------------------------------------------------------
# Step 6: Predict on X_test and save submission
# ---------------------------------------------------------------------------
print("Step 6: predicting on X_test and saving submission CSV...")
sys.stdout.flush()

X_test_feats = build_features(
    X_test_raw, X_test_raw["patient_id"],
    rater_bias_dict, velocity_dict, accel_dict,
    rater_consistency_dict, impute_medians,
)
y_pred = np.array(best_model.predict(X_test_feats))

sub_path = PROJECT_ROOT / "submissions" / "11_optuna_exp10_featureset.csv"
submission = pd.DataFrame({"Id": X_test_raw["Index"], "y_pred": y_pred})
submission.to_csv(sub_path, index=False)

print(f"  Saved {len(submission)} predictions -> {sub_path}")
print(f"  Prediction sample (first 3): {y_pred[:3].round(4).tolist()}")
sys.stdout.flush()

# ---------------------------------------------------------------------------
# Step 7: Push EstimatorReport to Skore Hub (EXP 06-10 pattern)
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
        np.arange(len(X_full)), test_size=0.2, random_state=42
    )
    X_tr_s = X_full.iloc[idx_tr[:2000]]
    y_tr_s = y_full[idx_tr[:2000]]
    X_te_s = X_full.iloc[idx_te[:500]]
    y_te_s = y_full[idx_te[:500]]

    report = skore.EstimatorReport(
        estimator=LGBMCatRegressor(
            cat_cols=CAT_COLS, n_estimators=150, random_state=42, verbose=-1,
            **best_params,
        ),
        X_train=X_tr_s,
        y_train=y_tr_s,
        X_test=X_te_s,
        y_test=y_te_s,
    )

    project.put("11_optuna_exp10_featureset", report)

    print("[OK] Pushed EstimatorReport to SkoreHub")
    sys.stdout.flush()
    print("[REPORT] https://skore.probabl.ai/projects/bobathon-esilv/items/11_optuna_exp10_featureset")
    sys.stdout.flush()
except Exception as exc:
    print(f"[WARN] Skore Hub push failed: {exc}")
    sys.stdout.flush()

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
print("\n=== Final results ===")
print(f"  Best CV RMSE : {best_cv_rmse:.4f}")
print(f"  Best params  : {best_params}")
print(f"  Submission   : {sub_path}")
sys.stdout.flush()
