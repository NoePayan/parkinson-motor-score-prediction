# 08 — LightGBM + Optuna hyperparameter tuning
#
# Same 11-col feature set as EXP 06 (8 numeric + 3 categorical natives).
# Optuna searches 30 trials with GroupKFold CV to find the best LightGBM
# hyperparameters, then fits the final model on 100 % of training data.
# Best params from this experiment are reused in EXP 09 and EXP 10.
#
# Steps
# -----
# 1. Load X_train / y_train / X_test; build feature matrix (11 cols).
# 2. Run Optuna study (30 trials, MedianPruner, direction=minimize).
# 3. Print best params + best CV RMSE.
# 4. Fit final model with best params on 100 % of training data.
# 5. Predict on X_test and write submissions/08_lightgbm_optuna.csv.
# 6. Push EstimatorReport (train[:2000] / test[:500]) to Skore Hub.

import io
import sys

# Force UTF-8 stdout to handle Unicode characters from skore/Optuna output
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

# Silence Optuna's per-trial INFO logs — progress bar is enough
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# LightGBM wrapper (same as EXP 06/07): bakes categorical_feature into fit()
# ---------------------------------------------------------------------------
class LGBMCatRegressor(BaseEstimator, RegressorMixin):
    """LightGBM wrapper that passes categorical_feature at fit() time."""

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

NUM_COLS = [
    "sexM", "age_at_diagnosis", "age", "ledd",
    "time_since_intake_on", "time_since_intake_off", "on", "off",
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

print(f"  X shape: {X.shape}, columns: {FEATURE_COLS}")
print(f"  unique patients: {groups.nunique()}")
sys.stdout.flush()

# ---------------------------------------------------------------------------
# 2. Optuna objective
# ---------------------------------------------------------------------------
cv = GroupKFold(n_splits=5)


def objective(trial: optuna.Trial) -> float:
    params = {
        "learning_rate":      trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves":         trial.suggest_int("num_leaves", 20, 100),
        "max_depth":          trial.suggest_int("max_depth", 5, 15),
        "min_child_samples":  trial.suggest_int("min_child_samples", 5, 50),
        "reg_lambda":         trial.suggest_float("reg_lambda", 0.0, 1.0),
        "reg_alpha":          trial.suggest_float("reg_alpha", 0.0, 0.5),
    }
    est = LGBMCatRegressor(
        cat_cols=CAT_COLS,
        n_estimators=150,
        random_state=42,
        verbose=-1,
        **params,
    )
    scores = cross_val_score(
        est, X, y,
        cv=cv,
        groups=groups,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    return float((-scores).mean())


# ---------------------------------------------------------------------------
# Step 2: Run Optuna study
# ---------------------------------------------------------------------------
print("Step 2: running Optuna study (30 trials)...")
sys.stdout.flush()

study = optuna.create_study(
    direction="minimize",
    pruner=optuna.pruners.MedianPruner(),
)
study.optimize(objective, n_trials=30, n_jobs=1, show_progress_bar=True)

best_params = study.best_params
best_cv_rmse = study.best_value

print(f"  Best CV RMSE : {best_cv_rmse:.4f}")
print(f"  Best params  : {best_params}")
sys.stdout.flush()

# ---------------------------------------------------------------------------
# Step 3: Fit final model with best params on 100 % of training data
# ---------------------------------------------------------------------------
print("Step 3: fitting final model with best params on 100 % of training data...")
sys.stdout.flush()

best_model = LGBMCatRegressor(
    cat_cols=CAT_COLS,
    n_estimators=150,
    random_state=42,
    verbose=-1,
    **best_params,
)
best_model.fit(X, y)
print("  Done.")
sys.stdout.flush()

# ---------------------------------------------------------------------------
# Step 4: Predict on X_test and save submission
# ---------------------------------------------------------------------------
print("Step 4: predicting on X_test and saving submission CSV...")
sys.stdout.flush()

X_test_feats = prepare(X_test_raw)
y_pred = np.array(best_model.predict(X_test_feats))

sub_path = PROJECT_ROOT / "submissions" / "08_lightgbm_optuna.csv"
submission = pd.DataFrame({"Id": X_test_raw["Index"], "y_pred": y_pred})
submission.to_csv(sub_path, index=False)

print(f"  Saved {len(submission)} predictions -> {sub_path}")
print(f"  Prediction sample (first 3): {y_pred[:3].round(4).tolist()}")
sys.stdout.flush()

# ---------------------------------------------------------------------------
# Step 5: Push EstimatorReport to Skore Hub (same pattern as EXP 06/07)
# ---------------------------------------------------------------------------
print("Step 5: pushing EstimatorReport to Skore Hub...")
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

    # Unfitted wrapper with best params — skore calls fit() internally
    report = skore.EstimatorReport(
        estimator=LGBMCatRegressor(
            cat_cols=CAT_COLS,
            n_estimators=150,
            random_state=42,
            verbose=-1,
            **best_params,
        ),
        X_train=X_tr_s,
        y_train=y_tr_s,
        X_test=X_te_s,
        y_test=y_te_s,
    )

    project.put("08_lightgbm_optuna", report)

    print("[OK] Pushed EstimatorReport to SkoreHub")
    sys.stdout.flush()
    print("[REPORT] https://skore.probabl.ai/projects/bobathon-esilv/items/08_lightgbm_optuna")
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
