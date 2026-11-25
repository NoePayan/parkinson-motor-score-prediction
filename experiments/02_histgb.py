# 02 — HistGradientBoosting: first real model
#
# Replace the dummy mean with a gradient-boosted tree.
# HistGBR handles missing values natively — no imputation needed.
# GroupKFold on patient_id ensures no patient leaks across folds,
# matching Kaggle's patient-level holdout strategy.
#
# Steps
# -----
# 1. Load data and build the labelled visits table.
# 2. Define feature columns and extract patient groups.
# 3. Cross-validate with GroupKFold(n_splits=5) — compute mean RMSE ± std.
# 4. Fit final model on 100 % of training data.
# 5. Predict on X_test and write submissions/02_histgb.csv.
# 6. Push EstimatorReport (train[:2000] / test[:500]) to Skore Hub.

import numpy as np
import pandas as pd
import skore
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_val_score, train_test_split

from parkinson import PROJECT_ROOT
from parkinson.hub import load_skore_credentials

# 1 — Load data

print("Step 1: loading data...")

X_train_raw = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv")
y_train_df  = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv")
X_test_raw  = pd.read_csv(PROJECT_ROOT / "data" / "X_test.csv")

# One row per visit with all predictors and the outcome.
visits = X_train_raw.merge(y_train_df, on="Index")

print(f"  visits shape: {visits.shape}")
print(f"  target mean={visits['target'].mean():.3f}, std={visits['target'].std():.3f}")

# 2 — Feature matrix and patient groups
#
# HistGBR handles NaN natively — include all 8 numeric predictors.
# Index is a row key (no signal); patient_id is used only for grouping.

print("Step 2: building feature matrix...")

FEATURE_COLS = [
    "sexM",
    "age_at_diagnosis",
    "age",
    "ledd",
    "time_since_intake_on",
    "time_since_intake_off",
    "on",
    "off",
]

X      = visits[FEATURE_COLS]
y      = visits["target"]
groups = visits["patient_id"]  # one group per patient — no patient leaks across folds

print(f"  X shape: {X.shape}, unique patients: {groups.nunique()}")

# 3 — GroupKFold cross-validation
#
# Each fold keeps all visits of a given patient on the same side of the split,
# mirroring Kaggle's patient-level holdout.

print("Step 3: running 5-fold GroupKFold cross-validation...")

model = HistGradientBoostingRegressor(
    max_iter=100,
    learning_rate=0.1,
    random_state=42,
)

cv = GroupKFold(n_splits=5)

cv_scores = cross_val_score(
    model, X, y,
    cv=cv,
    groups=groups,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
)

rmse_scores   = -cv_scores
mean_rmse     = rmse_scores.mean()
std_rmse      = rmse_scores.std()
baseline_rmse = float(y.std())  # dummy-mean RMSE floor (~16.5)

print(f"  Per-fold RMSE: {[round(r, 4) for r in rmse_scores]}")
print(f"  Mean CV RMSE : {mean_rmse:.4f} ± {std_rmse:.4f}")
print(f"  Baseline RMSE: {baseline_rmse:.4f}  (improvement: {baseline_rmse - mean_rmse:.4f})")

# 4 — Fit final model on 100 % of training data

print("Step 4: fitting final model on 100 % of training data...")

model.fit(X, y)
print("  Done.")

# 5 — Predict on X_test and write submission CSV

print("Step 5: predicting on X_test and saving submission CSV...")

X_test_feats = X_test_raw[FEATURE_COLS]
y_pred_test  = model.predict(X_test_feats)

submission = pd.DataFrame({
    "Index": X_test_raw["Index"],
    "target": y_pred_test,
})

sub_path = PROJECT_ROOT / "submissions" / "02_histgb.csv"
submission.to_csv(sub_path, index=False)

print(f"  Saved {len(submission)} predictions -> {sub_path}")
print(f"  Prediction sample (first 3): {y_pred_test[:3].round(4).tolist()}")

# 6 — Push EstimatorReport to Skore Hub
#
# CrossValidationReport on 44K rows serialises to ~130 MB and times out.
# EstimatorReport on train[:2000] / test[:500] stays under 2 MB.

print("Step 6: pushing EstimatorReport to Skore Hub...")

try:
    cfg = load_skore_credentials()
    skore.login(mode="hub")

    project = skore.Project(
        name="bobathon-esilv",
        mode="hub",
        workspace=cfg["workspace"],
    )

    X_tr_s, X_te_s, y_tr_s, y_te_s = train_test_split(X, y, test_size=0.2, random_state=42)
    X_tr_s = X_tr_s.iloc[:2000]
    y_tr_s = y_tr_s.iloc[:2000]
    X_te_s = X_te_s.iloc[:500]
    y_te_s = y_te_s.iloc[:500]

    report = skore.EstimatorReport(
        estimator=HistGradientBoostingRegressor(
            max_iter=100,
            learning_rate=0.1,
            random_state=42,
        ),
        X_train=X_tr_s,
        y_train=y_tr_s,
        X_test=X_te_s,
        y_test=y_te_s,
    )

    project.put("02_histgb", report)

    print("[OK] Pushed EstimatorReport to Skore Hub")
    print("[REPORT] https://skore.probabl.ai/projects/bobathon-esilv/items/02_histgb")
except Exception as exc:
    print(f"[WARN] Skore Hub push failed: {exc}")

print("\n=== Final results ===")
print(f"  Mean CV RMSE : {mean_rmse:.4f} ± {std_rmse:.4f}")
print(f"  Submission   : {sub_path}")
