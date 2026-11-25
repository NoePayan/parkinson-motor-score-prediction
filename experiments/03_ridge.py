# 03 — Ridge regression with one-hot encoding
#
# Encode manageable categorical columns (cohort, gene, rater_id),
# drop patient_id entirely (5 576 unique values would explode dummies),
# impute missing numerics with the column median, scale, then fit Ridge.
# Note: uses plain KFold (no group) — EXP 05 fixes this with GroupKFold.
#
# Steps
# -----
# 1. Load X_train / y_train / X_test.
# 2. Separate numeric and categorical columns; drop patient_id.
# 3. One-hot-encode cohort, gene, rater_id (drop_first=True).
# 4. Build Pipeline: SimpleImputer(median) → StandardScaler → Ridge(alpha=1.0).
# 5. 5-fold KFold cross-validation — report mean/std RMSE.
# 6. Fit final model on 100 % of training data.
# 7. Encode X_test; reindex on train columns.
# 8. Predict and write submissions/03_ridge.csv.
# 9. Push EstimatorReport (train[:2000] / test[:500]) to Skore Hub.

import sys

import numpy as np
import pandas as pd
import skore
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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

# Align target with features using the shared row key.
train = X_train_raw.merge(y_df, on="Index")

print(f"  train shape: {train.shape}")
print(f"  target mean={train['target'].mean():.3f}, std={train['target'].std():.3f}")
sys.stdout.flush()

# ---------------------------------------------------------------------------
# 2. Define column groups; drop patient_id and Index
# ---------------------------------------------------------------------------
print("Step 2: separating numeric and categorical columns (dropping patient_id)...")
sys.stdout.flush()

# patient_id has 5 576 unique values — one-hot-encoding it would create
# an unmanageable number of sparse dummies and leak patient identity.
CAT_ENCODE = ["cohort", "gene", "rater_id"]   # low-cardinality, safe to encode
DROP_COLS  = ["patient_id", "Index"]          # row key + high-cardinality id

NUM_COLS = [
    c for c in X_train_raw.columns
    if c not in CAT_ENCODE + DROP_COLS
]

print(f"  numeric cols ({len(NUM_COLS)}): {NUM_COLS}")
print(f"  encoded cats ({len(CAT_ENCODE)}): {CAT_ENCODE}")
sys.stdout.flush()

# ---------------------------------------------------------------------------
# 3. One-hot-encode categorical columns on the training set
# ---------------------------------------------------------------------------
def encode(df: pd.DataFrame) -> pd.DataFrame:
    """Return a new DataFrame with CAT_ENCODE columns one-hot-encoded."""
    base = df[NUM_COLS].copy()
    dummies = pd.get_dummies(df[CAT_ENCODE], drop_first=True, dtype=float)
    return pd.concat([base, dummies], axis=1)

X_enc = encode(train)
y = train["target"].values

print(f"  encoded train shape: {X_enc.shape}")
print(f"  columns: {list(X_enc.columns)}")
sys.stdout.flush()

TRAIN_COLS = X_enc.columns.tolist()  # reference column order for test alignment

# ---------------------------------------------------------------------------
# 4. Build the pipeline
# ---------------------------------------------------------------------------
pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
    ("ridge",   Ridge(alpha=1.0)),
])

# ---------------------------------------------------------------------------
# 5. 5-fold cross-validation
# ---------------------------------------------------------------------------
print("Step 5: running 5-fold cross-validation (n_jobs=4, verbose=1)...")
sys.stdout.flush()

kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(
    pipe,
    X_enc,
    y,
    cv=kf,
    scoring="neg_root_mean_squared_error",
    n_jobs=4,
    verbose=1,
)

rmse_scores = -cv_scores
mean_rmse   = rmse_scores.mean()
std_rmse    = rmse_scores.std()

print(f"  Per-fold RMSE : {[round(r, 4) for r in rmse_scores]}")
print(f"  Mean CV RMSE  : {mean_rmse:.4f} ± {std_rmse:.4f}")
sys.stdout.flush()

# ---------------------------------------------------------------------------
# 6. Fit final model on 100 % of training data
# ---------------------------------------------------------------------------
print("Step 6: fitting final model on 100 % of training data...")
sys.stdout.flush()

pipe.fit(X_enc, y)
print("  Done.")
sys.stdout.flush()

# ---------------------------------------------------------------------------
# 7. Encode test set; reindex to match training columns
# ---------------------------------------------------------------------------
print("Step 7: encoding X_test and aligning to train columns...")
sys.stdout.flush()

X_test_enc = encode(X_test_raw).reindex(columns=TRAIN_COLS, fill_value=0.0)

print(f"  X_test encoded shape: {X_test_enc.shape}")
sys.stdout.flush()

# ---------------------------------------------------------------------------
# 8. Predict and save submission
# ---------------------------------------------------------------------------
print("Step 8: predicting on X_test and saving submission CSV...")
sys.stdout.flush()

y_pred = pipe.predict(X_test_enc)

sub_path = PROJECT_ROOT / "submissions" / "03_ridge.csv"
submission = pd.DataFrame({"Id": X_test_raw["Index"], "y_pred": y_pred})
submission.to_csv(sub_path, index=False)

print(f"  Saved {len(submission)} predictions -> {sub_path}")
print(f"  Prediction sample (first 3): {y_pred[:3].round(4).tolist()}")
sys.stdout.flush()

# ---------------------------------------------------------------------------
# 9. Push EstimatorReport to Skore Hub
# ---------------------------------------------------------------------------
# CrossValidationReport serialises all fold data (~130 MB) and times out on
# the Hub upload. Instead we build an EstimatorReport on a small stratified
# sample (2 000 train / 500 test) so the payload stays under 2 MB.
print("Step 9: pushing EstimatorReport to Skore Hub...")
sys.stdout.flush()

try:
    cfg = load_skore_credentials()
    skore.login(mode="hub")

    project = skore.Project(
        name="bobathon-esilv",
        mode="hub",
        workspace=cfg["workspace"],
    )

    # Small sample for the report — the final predictions come from the full
    # model fitted in step 6, not from this subset.
    X_tr_s, X_te_s, y_tr_s, y_te_s = train_test_split(
        X_enc, y, test_size=0.2, random_state=42
    )
    X_tr_s = X_tr_s.iloc[:2000]
    y_tr_s = y_tr_s[:2000]
    X_te_s = X_te_s.iloc[:500]
    y_te_s = y_te_s[:500]

    # EstimatorReport expects an *unfitted* estimator — it fits internally.
    report = skore.EstimatorReport(
        estimator=Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("ridge",   Ridge(alpha=1.0)),
        ]),
        X_train=X_tr_s,
        y_train=y_tr_s,
        X_test=X_te_s,
        y_test=y_te_s,
    )

    project.put("03_ridge", report)

    print("[OK] Pushed EstimatorReport to SkoreHub")
    sys.stdout.flush()
    print("[REPORT] Consult your report at: https://skore.probabl.ai/projects/bobathon-esilv/items/03_ridge")
    sys.stdout.flush()
except Exception as exc:
    print(f"[WARN] Skore Hub push failed: {exc}")
    sys.stdout.flush()

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
print("\n=== Final results ===")
print(f"  Mean CV RMSE : {mean_rmse:.4f} ± {std_rmse:.4f}")
print(f"  Submission   : {sub_path}")
sys.stdout.flush()
