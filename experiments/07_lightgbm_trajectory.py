# 07 — LightGBM + visit trajectory features
#
# Extends EXP 06 with 4 per-patient temporal features:
#   - visit_count  : total visits per patient
#   - time_span    : range of time_since_intake across visits
#   - off_slope    : OLS slope of off ~ time_since_intake_off (0 if < 2 valid rows)
#   - on_slope     : OLS slope of on  ~ time_since_intake_on  (0 if < 2 valid rows)
#
# Hypothesis: the rate of change of motor scores across visits captures
# disease progression signal that single-visit features cannot.
#
# Steps
# -----
# 1. Load X_train / y_train / X_test.
# 2. Compute per-patient trajectory features from the training visits table.
# 3. Build feature matrix: 12 numeric + 3 categorical.
# 4. GroupKFold(n_splits=5) CV on patient_id — compute mean RMSE ± std.
# 5. Fit final model on 100 % of training data.
# 6. Print top-10 feature importances.
# 7. Predict on X_test and write submissions/07_lightgbm_trajectory.csv.
# 8. Push EstimatorReport (train[:2000] / test[:500]) to Skore Hub.

import io
import sys

# Force UTF-8 stdout to handle Unicode spinner characters from skore output
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import skore
from lightgbm import LGBMRegressor
from scipy.stats import linregress
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GroupKFold, cross_val_score, train_test_split

from parkinson import PROJECT_ROOT
from parkinson.hub import load_skore_credentials


# ---------------------------------------------------------------------------
# sklearn wrapper: bakes categorical_feature into fit() so skore can call
# fit() without fit_params (same pattern as 06_lightgbm_cats.py)
# ---------------------------------------------------------------------------
class LGBMCatRegressor(BaseEstimator, RegressorMixin):
    """LightGBM wrapper that passes categorical_feature at fit() time."""

    def __init__(self, cat_cols, n_estimators=150, learning_rate=0.05,
                 random_state=42, verbose=-1):
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
# 2. Compute per-patient trajectory features
# ---------------------------------------------------------------------------
print("Step 2: computing trajectory features...")
sys.stdout.flush()


def _slope(x_series: pd.Series, y_series: pd.Series) -> float:
    """Return OLS slope of y ~ x, using only rows where both are finite."""
    mask = x_series.notna() & y_series.notna()
    x = x_series[mask].values
    y = y_series[mask].values
    if len(x) < 2:
        return 0.0
    return float(linregress(x, y).slope)


# Build one-row-per-patient summary
traj_rows = []
for pid, grp in visits.groupby("patient_id"):
    traj_rows.append({
        "patient_id": pid,
        "visit_count": len(grp),
        "time_span": (
            grp["time_since_intake_off"].max() - grp["time_since_intake_off"].min()
            if grp["time_since_intake_off"].notna().sum() >= 2
            else 0.0
        ),
        "off_slope": _slope(grp["time_since_intake_off"], grp["off"]),
        "on_slope":  _slope(grp["time_since_intake_on"],  grp["on"]),
    })

traj_df = pd.DataFrame(traj_rows)

# Merge back into visits (one trajectory value per visit of that patient)
visits = visits.merge(traj_df, on="patient_id", how="left")

print(f"  trajectory features computed for {len(traj_df)} patients")
print(f"  visit_count: mean={traj_df['visit_count'].mean():.1f}, max={traj_df['visit_count'].max()}")
print(f"  off_slope   non-zero: {(traj_df['off_slope'] != 0).sum()}")
sys.stdout.flush()

# ---------------------------------------------------------------------------
# 3. Feature matrix: 12 numeric + 3 categorical
# ---------------------------------------------------------------------------
print("Step 3: building feature matrix...")
sys.stdout.flush()

NUM_COLS = [
    "sexM", "age_at_diagnosis", "age", "ledd",
    "time_since_intake_on", "time_since_intake_off", "on", "off",
    "visit_count", "time_span", "off_slope", "on_slope",
]
CAT_COLS = ["cohort", "gene", "rater_id"]
FEATURE_COLS = NUM_COLS + CAT_COLS


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Return feature slice with categorical columns cast to 'category' dtype."""
    out = df[FEATURE_COLS].copy()
    for col in CAT_COLS:
        out[col] = out[col].astype("category")
    return out


X = prepare(visits)
y = visits["target"].values
groups = visits["patient_id"]

print(f"  X shape: {X.shape}")
print(f"  columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print(f"  unique patients: {groups.nunique()}")
sys.stdout.flush()

# ---------------------------------------------------------------------------
# 4. GroupKFold cross-validation
# ---------------------------------------------------------------------------
print("Step 4: running 5-fold GroupKFold cross-validation...")
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
# 5. Fit final model on 100 % of training data
# ---------------------------------------------------------------------------
print("Step 5: fitting final model on 100 % of training data...")
sys.stdout.flush()

model.fit(X, y)
print("  Done.")
sys.stdout.flush()

# ---------------------------------------------------------------------------
# 6. Feature importances (top 10)
# ---------------------------------------------------------------------------
print("Step 6: top-10 feature importances (gain)...")
sys.stdout.flush()

importances = model.model_.feature_importances_
feat_imp = pd.Series(importances, index=FEATURE_COLS).sort_values(ascending=False)
print(feat_imp.head(10).to_string())
sys.stdout.flush()

# ---------------------------------------------------------------------------
# 7. Predict on X_test and save submission
# ---------------------------------------------------------------------------
print("Step 7: predicting on X_test and saving submission CSV...")
sys.stdout.flush()

# Compute trajectory features for test set using training statistics.
# Test visits don't have targets, so we merge traj_df (from train) by patient_id.
# Patients unseen in train get NaN trajectory features → LightGBM handles NaN.
X_test_with_traj = X_test_raw.merge(traj_df, on="patient_id", how="left")
X_test_feats = prepare(X_test_with_traj)

y_pred = np.array(model.predict(X_test_feats))

sub_path = PROJECT_ROOT / "submissions" / "07_lightgbm_trajectory.csv"
submission = pd.DataFrame({"Id": X_test_raw["Index"], "y_pred": y_pred})
submission.to_csv(sub_path, index=False)

print(f"  Saved {len(submission)} predictions -> {sub_path}")
print(f"  Prediction sample (first 3): {y_pred[:3].round(4).tolist()}")
sys.stdout.flush()

# ---------------------------------------------------------------------------
# 8. Push EstimatorReport to Skore Hub (same pattern as 06_lightgbm_cats)
# ---------------------------------------------------------------------------
print("Step 8: pushing EstimatorReport to Skore Hub...")
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

    # Pass unfitted LGBMCatRegressor — skore calls fit() internally, which
    # routes categorical_feature through the wrapper (same as EXP 06).
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

    project.put("07_lightgbm_trajectory", report)

    print("[OK] Pushed EstimatorReport to SkoreHub")
    sys.stdout.flush()
    print("[REPORT] https://skore.probabl.ai/projects/bobathon-esilv/items/07_lightgbm_trajectory")
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
