# 05 — Ridge regression with GroupKFold + ColumnTransformer
#
# Fixes the methodological gap in EXP 03: uses GroupKFold(patient_id)
# instead of plain KFold so CV is honest — no patient leakage across folds.
# Features: 8 numeric (StandardScaler) + 3 categorical (OHE drop='first').
# Direct comparison vs EXP 04 under identical splitter and feature set.
#
# Steps
# -----
# 1. Load X_train / y_train / X_test.
# 2. Build feature matrix: 8 numeric + 3 OHE categorical columns.
# 3. GroupKFold(n_splits=5) CV on patient_id — compute mean RMSE ± std.
# 4. Fit final model on 100 % of training data.
# 5. Predict on X_test and write submissions/05_ridge_groupkfold.csv.
# 6. Push EstimatorReport (train[:2000] / test[:500]) to Skore Hub.

import sys

import pandas as pd
import skore
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: F401 (kept for reference)
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from parkinson import PROJECT_ROOT
from parkinson.hub import load_skore_credentials

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("Step 1: loading data...")
sys.stdout.flush()

X_train_raw = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv")
y_df = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv")
X_test_raw = pd.read_csv(PROJECT_ROOT / "data" / "X_test.csv")

visits = X_train_raw.merge(y_df, on="Index")

print(f"  visits shape: {visits.shape}")
print(f"  target mean={visits['target'].mean():.3f}, std={visits['target'].std():.3f}")
sys.stdout.flush()

# ---------------------------------------------------------------------------
# 2. Feature matrix: 8 numeric + 3 categorical (raw strings for OHE)
# ---------------------------------------------------------------------------
print("Step 2: building feature matrix...")
sys.stdout.flush()

NUM_COLS = [
    "sexM",
    "age_at_diagnosis",
    "age",
    "ledd",
    "time_since_intake_on",
    "time_since_intake_off",
    "on",
    "off",
]
CAT_COLS = ["cohort", "gene", "rater_id"]
FEATURE_COLS = NUM_COLS + CAT_COLS

X = visits[FEATURE_COLS]
y = visits["target"].values
groups = visits["patient_id"]

print(f"  X shape (pre-OHE): {X.shape}")
print(f"  columns: {FEATURE_COLS}")
print(f"  unique patients: {groups.nunique()}")
sys.stdout.flush()

# ---------------------------------------------------------------------------
# 3. Pipeline: ColumnTransformer + Ridge
# ---------------------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), NUM_COLS),
        ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), CAT_COLS),
    ]
)

pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("ridge", Ridge(alpha=1.0)),
])

# ---------------------------------------------------------------------------
# 4. GroupKFold cross-validation
# ---------------------------------------------------------------------------
print("Step 3: running 5-fold GroupKFold cross-validation...")
sys.stdout.flush()

cv = GroupKFold(n_splits=5)

cv_scores = cross_val_score(
    pipe,
    X,
    y,
    cv=cv,
    groups=groups,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
)

rmse_scores = -cv_scores
mean_rmse = rmse_scores.mean()
std_rmse = rmse_scores.std()

print(f"  Per-fold RMSE: {[round(r, 4) for r in rmse_scores]}")
print(f"  Mean CV RMSE : {mean_rmse:.4f} +/- {std_rmse:.4f}")
sys.stdout.flush()

# ---------------------------------------------------------------------------
# 5. Fit final model on 100 % of training data
# ---------------------------------------------------------------------------
print("Step 4: fitting final model on 100 % of training data...")
sys.stdout.flush()

pipe.fit(X, y)

# Report post-OHE dimensionality
n_features_out = pipe.named_steps["preprocessor"].transform(X.iloc[:1]).shape[1]
print(f"  Done. Post-OHE feature count: {n_features_out}")
sys.stdout.flush()

# ---------------------------------------------------------------------------
# 6. Predict on X_test and save submission
# ---------------------------------------------------------------------------
print("Step 5: predicting on X_test and saving submission CSV...")
sys.stdout.flush()

X_test_feats = X_test_raw[FEATURE_COLS]
y_pred = pipe.predict(X_test_feats)

sub_path = PROJECT_ROOT / "submissions" / "05_ridge_groupkfold.csv"
submission = pd.DataFrame({"Id": X_test_raw["Index"], "y_pred": y_pred})
submission.to_csv(sub_path, index=False)

print(f"  Saved {len(submission)} predictions -> {sub_path}")
print(f"  Prediction sample (first 3): {y_pred[:3].round(4).tolist()}")
sys.stdout.flush()

# ---------------------------------------------------------------------------
# 7. Push EstimatorReport to Skore Hub (sampled to stay under 2 MB)
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

    X_tr_s, X_te_s, y_tr_s, y_te_s = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_tr_s = X_tr_s.iloc[:2000]
    y_tr_s = y_tr_s[:2000]
    X_te_s = X_te_s.iloc[:500]
    y_te_s = y_te_s[:500]

    report = skore.EstimatorReport(
        estimator=Pipeline([
            ("preprocessor", ColumnTransformer(
                transformers=[
                    ("num", Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]), NUM_COLS),
                    ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), CAT_COLS),
                ]
            )),
            ("ridge", Ridge(alpha=1.0)),
        ]),
        X_train=X_tr_s,
        y_train=y_tr_s,
        X_test=X_te_s,
        y_test=y_te_s,
    )

    project.put("05_ridge_groupkfold", report)

    print("[OK] Pushed EstimatorReport to SkoreHub")
    sys.stdout.flush()
    print("[REPORT] https://skore.probabl.ai/projects/bobathon-esilv/items/05_ridge_groupkfold")
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
