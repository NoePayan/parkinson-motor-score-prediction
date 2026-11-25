# 09 — LightGBM + domain feature engineering
#
# Builds on EXP 06 (LightGBM + categoricals) using EXP 08 Optuna best params.
# Adds 7 domain-driven features:
#   - rater_bias      : per-rater mean target (fold-safe — refitted per CV fold)
#   - is_off_missing  : missingness indicator for 'off'
#   - is_on_missing   : missingness indicator for 'on'
#   - off_x_age       : interaction off × age
#   - off_x_time      : interaction off × time_since_intake_on
#   - ledd_x_age      : interaction ledd × age
#   - on_x_age        : interaction on × age
#
# Imputation strategy: per-patient median → per-rater median → global median.
# rater_bias is recomputed inside each CV fold to avoid target leakage.
#
# Steps
# -----
# 1. Load data; compute rater_bias and imputation medians from full train.
# 2. Build feature matrix (18 cols: 8 numeric + 7 domain + 3 categorical).
# 3. GroupKFold(n_splits=5) CV with fold-safe rater_bias — compute RMSE ± std.
# 4. Fit final model on 100 % of training data; print top-10 importances.
# 5. Predict on X_test and write submissions/09_lightgbm_domain_features.csv.
# 6. Push EstimatorReport (train[:2000] / test[:500]) to Skore Hub.

import io
import sys

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import skore
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import GroupKFold, cross_val_score, train_test_split

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
CAT_COLS = ["cohort", "gene", "rater_id"]
FEATURE_COLS = NUM_COLS + DOMAIN_COLS + CAT_COLS


# ---------------------------------------------------------------------------
# LightGBM wrapper (same pattern as EXP 06/07/08)
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
# Feature engineering helpers
# ---------------------------------------------------------------------------
def _impute_column(df: pd.DataFrame, col: str, patient_id: pd.Series,
                   fallback_medians: dict | None = None) -> pd.DataFrame:
    """Impute col in-place: per-patient median → per-rater median → global.

    fallback_medians: pre-computed medians from train (used for test).
    """
    df = df.copy()
    if df[col].isna().sum() == 0:
        return df

    if fallback_medians is None:
        # Fit: compute per-patient and per-rater medians from this df
        pat_med = df.groupby(patient_id)[col].transform("median")
        mask = df[col].isna()
        df.loc[mask, col] = pat_med[mask]

        if df[col].isna().sum() > 0:
            rater_med = df.groupby("rater_id")[col].transform("median")
            mask = df[col].isna()
            df.loc[mask, col] = rater_med[mask]

        global_med = df[col].median()
        df[col] = df[col].fillna(global_med)
    else:
        # Apply: use pre-computed medians from train
        pat_med_map  = fallback_medians.get(f"{col}_patient", {})
        rater_med_map = fallback_medians.get(f"{col}_rater", {})
        global_med   = fallback_medians.get(f"{col}_global", 0.0)

        mask = df[col].isna()
        df.loc[mask, col] = patient_id[mask].map(pat_med_map)

        mask = df[col].isna()
        df.loc[mask, col] = df.loc[mask, "rater_id"].map(rater_med_map)

        df[col] = df[col].fillna(global_med)
    return df


def build_features(df: pd.DataFrame, patient_id: pd.Series,
                   rater_bias_dict: dict,
                   impute_medians: dict | None = None) -> pd.DataFrame:
    """Return feature matrix for df with domain features added."""
    out = df[NUM_COLS + CAT_COLS].copy()

    # Domain imputation on numeric columns with high missingness
    for col in ["off", "on", "time_since_intake_off"]:
        out = _impute_column(out, col, patient_id, impute_medians)

    # Missing indicators (computed BEFORE imputation on original df)
    out["is_off_missing"] = df["off"].isna().astype(np.float32)
    out["is_on_missing"]  = df["on"].isna().astype(np.float32)

    # Rater bias (pre-computed from train targets)
    out["rater_bias"] = out["rater_id"].map(rater_bias_dict).fillna(0.0).clip(-2, 2)

    # Interaction features
    out["off_x_age"]  = out["off"] * out["age"]
    out["off_x_time"] = out["off"] * out["time_since_intake_on"]
    out["ledd_x_age"] = out["ledd"] * out["age"]
    out["on_x_age"]   = out["on"]  * out["age"]

    # Cast categoricals
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
print(f"  target mean={y_full.mean():.3f}, std={y_full.std():.3f}")
sys.stdout.flush()

# ---------------------------------------------------------------------------
# 2. Compute rater_bias on full train (used for final model; CV recomputes per fold)
# ---------------------------------------------------------------------------
# Rater bias = per-rater mean target (proxy for systematic over/under-scoring).
rater_bias_dict = visits.groupby("rater_id")["target"].mean().to_dict()

# Pre-compute imputation medians from full train for test-set application
impute_medians: dict = {}
for col in ["off", "on", "time_since_intake_off"]:
    impute_medians[f"{col}_patient"] = (
        visits.groupby("patient_id")[col].median().to_dict()
    )
    impute_medians[f"{col}_rater"] = (
        visits.groupby("rater_id")[col].median().to_dict()
    )
    impute_medians[f"{col}_global"] = visits[col].median()

# ---------------------------------------------------------------------------
# 3. Build full train feature matrix
# ---------------------------------------------------------------------------
print("Step 2: building feature matrix (domain features)...")
sys.stdout.flush()

X_full = build_features(visits, patient_id_train, rater_bias_dict)
print(f"  X shape: {X_full.shape}")
print(f"  columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print(f"  unique patients: {patient_id_train.nunique()}")
sys.stdout.flush()

# ---------------------------------------------------------------------------
# 4. GroupKFold cross-validation
#    rater_bias is refitted inside each fold to avoid leakage
# ---------------------------------------------------------------------------
print("Step 3: running 5-fold GroupKFold cross-validation (fold-level rater bias)...")
sys.stdout.flush()

groups = patient_id_train
cv = GroupKFold(n_splits=5)

fold_rmses = []
for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_full, y_full, groups=groups)):
    fold_visits_tr = visits.iloc[train_idx]
    fold_visits_va = visits.iloc[val_idx]
    y_tr = y_full[train_idx]
    y_va = y_full[val_idx]

    # Recompute rater_bias on fold-train only
    fold_rater_bias = fold_visits_tr.groupby("rater_id")["target"].mean().to_dict()

    # Build fold features
    X_tr = build_features(fold_visits_tr, fold_visits_tr["patient_id"], fold_rater_bias)
    X_va = build_features(fold_visits_va, fold_visits_va["patient_id"], fold_rater_bias)

    est = LGBMCatRegressor(
        cat_cols=CAT_COLS, n_estimators=150, random_state=42, verbose=-1,
        **BEST_PARAMS,
    )
    est.fit(X_tr, y_tr)
    y_pred_va = est.predict(X_va)
    rmse = np.sqrt(np.mean((y_va - y_pred_va) ** 2))
    fold_rmses.append(rmse)
    print(f"  fold {fold_idx + 1}/5 RMSE: {rmse:.4f}")
    sys.stdout.flush()

mean_rmse = np.mean(fold_rmses)
std_rmse  = np.std(fold_rmses)
print(f"  Per-fold RMSE: {[round(r, 4) for r in fold_rmses]}")
print(f"  Mean CV RMSE : {mean_rmse:.4f} +/- {std_rmse:.4f}")
sys.stdout.flush()

# ---------------------------------------------------------------------------
# 5. Fit final model on 100 % of training data
# ---------------------------------------------------------------------------
print("Step 4: fitting final model on 100 % of training data...")
sys.stdout.flush()

final_model = LGBMCatRegressor(
    cat_cols=CAT_COLS, n_estimators=150, random_state=42, verbose=-1,
    **BEST_PARAMS,
)
final_model.fit(X_full, y_full)
print("  Done.")
sys.stdout.flush()

# Feature importances
feat_imp = pd.Series(
    final_model.model_.feature_importances_, index=FEATURE_COLS
).sort_values(ascending=False)
print("  Top-10 feature importances:")
print(feat_imp.head(10).to_string())
sys.stdout.flush()

# ---------------------------------------------------------------------------
# 6. Predict on X_test and save submission
# ---------------------------------------------------------------------------
print("Step 5: predicting on X_test and saving submission CSV...")
sys.stdout.flush()

X_test_feats = build_features(
    X_test_raw, X_test_raw["patient_id"], rater_bias_dict, impute_medians
)
y_pred = np.array(final_model.predict(X_test_feats))

sub_path = PROJECT_ROOT / "submissions" / "09_lightgbm_domain_features.csv"
submission = pd.DataFrame({"Id": X_test_raw["Index"], "y_pred": y_pred})
submission.to_csv(sub_path, index=False)

print(f"  Saved {len(submission)} predictions -> {sub_path}")
print(f"  Prediction sample (first 3): {y_pred[:3].round(4).tolist()}")
sys.stdout.flush()

# ---------------------------------------------------------------------------
# 7. Push EstimatorReport to Skore Hub (same pattern as EXP 06/07/08)
# ---------------------------------------------------------------------------
print("Step 6: pushing EstimatorReport to Skore Hub...")
sys.stdout.flush()

try:
    cfg = load_skore_credentials()
    skore.login(mode="hub")

    project = skore.Project(
        name="bobathon-esilv",
        mode="hub",
        workspace=cfg["workspace"],
    )

    # Small sample for the report — unfitted wrapper so skore calls fit() itself
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
            **BEST_PARAMS,
        ),
        X_train=X_tr_s,
        y_train=y_tr_s,
        X_test=X_te_s,
        y_test=y_te_s,
    )

    project.put("09_lightgbm_domain_features", report)

    print("[OK] Pushed EstimatorReport to SkoreHub")
    sys.stdout.flush()
    print("[REPORT] https://skore.probabl.ai/projects/bobathon-esilv/items/09_lightgbm_domain_features")
    sys.stdout.flush()
except Exception as exc:
    print(f"[WARN] Skore Hub push failed: {exc}")
    sys.stdout.flush()

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
print("\n=== Final results ===")
print(f"  Mean CV RMSE : {mean_rmse:.4f} +/- {std_rmse:.4f}")
print(f"  Submission   : {sub_path}")
sys.stdout.flush()
