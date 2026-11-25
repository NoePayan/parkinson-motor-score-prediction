# %% [markdown]
# # 01 — Dummy baseline: predict the training mean
#
# The laziest possible model: always predict the mean of y_train.
# This sets the RMSE floor (~16.5) that every real model must beat.
#
# Steps
# -----
# 1. Load data and build the labelled visits table.
# 2. Fit a DummyRegressor(strategy="mean") on the feature subset.
# 3. Evaluate with skore.evaluate (random 20 % holdout — sufficient for a floor).
# 4. Write test predictions to submissions/01_dummy.csv.
# 5. Push the EstimatorReport to Skore Hub under key "01_dummy".

# %%
from pathlib import Path

import numpy as np
import pandas as pd
import skore
from sklearn.dummy import DummyRegressor

from parkinson import PROJECT_ROOT
from parkinson.hub import load_skore_credentials

# %% [markdown]
# ## 1 — Load data

# %%
X_train_raw = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv")
y_train_df = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv")
X_test_raw = pd.read_csv(PROJECT_ROOT / "data" / "X_test.csv")

# Merge features + target on the row key.
visits = X_train_raw.merge(y_train_df, on="Index")

# Features used for the dummy (Index and patient_id are excluded — see EDA).
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

X = visits[FEATURE_COLS]
y = visits["target"]

{  # noqa: B018
    "train_shape": X.shape,
    "target_mean": round(float(y.mean()), 4),
    "target_std": round(float(y.std()), 4),
}

# %% [markdown]
# ## 2 — Fit DummyRegressor

# %%
dummy = DummyRegressor(strategy="mean")

# %% [markdown]
# ## 3 — Evaluate with skore (random 20 % holdout)
#
# skore.evaluate fits the estimator on the 80 % training split and
# evaluates it on the 20 % holdout.  The default splitter is 0.2.
# RMSE should equal the std of y (~16.5) because a constant predictor
# has zero variance and its error is exactly the residual std.

# %%
report = skore.evaluate(dummy, X, y)
report.metrics.rmse()

# %% [markdown]
# ## 4 — Generate test predictions and write submission CSV

# %%
# Fit on the full training set before predicting on test.
dummy.fit(X, y)
pred_mean = float(dummy.predict(X[:1])[0])  # constant = training mean

X_test_feats = X_test_raw[FEATURE_COLS]
y_pred_test = dummy.predict(X_test_feats)

submission = pd.DataFrame({
    "Index": X_test_raw["Index"],
    "target": y_pred_test,
})

(PROJECT_ROOT / "submissions").mkdir(exist_ok=True)
sub_path = PROJECT_ROOT / "submissions" / "01_dummy.csv"
submission.to_csv(sub_path, index=False)

{  # noqa: B018
    "submission_path": str(sub_path),
    "submission_shape": submission.shape,
    "predicted_value": round(pred_mean, 4),
}

# %% [markdown]
# ## 5 — Push EstimatorReport to Skore Hub

# %%
cfg = load_skore_credentials()
skore.login(mode="hub")
project = skore.Project(
    name="bobathon-esilv",
    mode="hub",
    workspace=cfg["workspace"],
)
project.put("01_dummy", report)

