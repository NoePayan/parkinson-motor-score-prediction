# 06 — LightGBM with native categorical features
#
# Mirrors EXP 04 but swaps HistGBR for LightGBM.
# LightGBM handles categoricals natively via its optimal-split histogram —
# faster and often more accurate than OHE on high-cardinality columns.
#
# LightGBM specifics vs HistGBR:
#   - categorical_feature is passed to fit(), not the constructor
#   - Categorical columns must be cast to pandas 'category' dtype beforehand
#   - verbose=-1 suppresses training logs
#
# Steps
# -----
# 1. Load X_train / y_train / X_test.
# 2. Build feature matrix: 8 numeric + 3 categorical (cast to 'category').
# 3. GroupKFold(n_splits=5) CV on patient_id — compute mean RMSE ± std.
# 4. Fit final model on 100 % of training data.
# 5. Predict on X_test and write submissions/06_lightgbm_cats.csv.
# 6. Push EstimatorReport (train[:2000] / test[:500]) to Skore Hub.

import io
import sys

# Force UTF-8 stdout to handle Unicode spinner characters from skore's progress output
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import skore
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GroupKFold, cross_val_score, train_test_split

from parkinson import PROJECT_ROOT
from parkinson.hub import load_skore_credentials


class LGBMCatRegressor(BaseEstimator, RegressorMixin):
    """Thin sklearn wrapper that always passes categorical_feature to LightGBM fit().

    Needed because skore.EstimatorReport calls fit() internally without
    fit_params, so categorical columns must be declared at construction time.
    """

    def __init__(self, cat_cols, n_estimators=150, learning_rate=0.05, random_state=42, verbose=-1):
        self.cat_cols = cat_cols
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        self.model_ = LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            verbose=self.verbose,
        )
        self.model_.fit(X, y, categorical_feature=self.cat_cols)
        return self

    def predict(self, X):
        return self.model_.predict(X)

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
# 2. Feature matrix: 8 numeric + 3 categorical (cast to 'category' for LGBM)
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


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Return feature slice with cat columns cast to 'category' dtype."""
    out = df[FEATURE_COLS].copy()
    for col in CAT_COLS:
        out[col] = out[col].astype("category")
    return out


X = prepare(visits)
y = visits["target"].values
groups = visits["patient_id"]

print(f"  X shape: {X.shape}")
print(f"  columns: {FEATURE_COLS}")
print(f"  unique patients: {groups.nunique()}")
sys.stdout.flush()

# ---------------------------------------------------------------------------
# 3. GroupKFold cross-validation
# ---------------------------------------------------------------------------
print("Step 3: running 5-fold GroupKFold cross-validation...")
sys.stdout.flush()

model = LGBMCatRegressor(
    cat_cols=CAT_COLS,
    n_estimators=150,
    learning_rate=0.05,
    random_state=42,
    verbose=-1,
)

cv = GroupKFold(n_splits=5)

cv_scores = cross_val_score(
    model,
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
# 4. Fit final model on 100 % of training data
# ---------------------------------------------------------------------------
print("Step 4: fitting final model on 100 % of training data...")
sys.stdout.flush()

model.fit(X, y)
print("  Done.")
sys.stdout.flush()

# ---------------------------------------------------------------------------
# 5. Predict on X_test and save submission
# ---------------------------------------------------------------------------
print("Step 5: predicting on X_test and saving submission CSV...")
sys.stdout.flush()

X_test_feats = prepare(X_test_raw)
y_pred = np.array(model.predict(X_test_feats))

sub_path = PROJECT_ROOT / "submissions" / "06_lightgbm_cats.csv"
submission = pd.DataFrame({"Id": X_test_raw["Index"], "y_pred": y_pred})
submission.to_csv(sub_path, index=False)

print(f"  Saved {len(submission)} predictions -> {sub_path}")
print(f"  Prediction sample (first 3): {y_pred[:3].round(4).tolist()}")
sys.stdout.flush()

# ---------------------------------------------------------------------------
# 6. Push EstimatorReport to Skore Hub (sampled to stay under 2 MB)
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

    # Use LGBMCatRegressor (unfitted) so skore can call fit() internally
    # with categorical_feature already baked into the wrapper.
    report = skore.EstimatorReport(
        estimator=LGBMCatRegressor(
            cat_cols=CAT_COLS,
            n_estimators=150,
            learning_rate=0.05,
            random_state=42,
            verbose=-1,
        ),
        X_train=X_tr_s,
        y_train=y_tr_s,
        X_test=X_te_s,
        y_test=y_te_s,
    )

    project.put("06_lightgbm_cats", report)

    print("[OK] Pushed EstimatorReport to SkoreHub")
    sys.stdout.flush()
    print("[REPORT] https://skore.probabl.ai/projects/bobathon-esilv/items/06_lightgbm_cats")
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
