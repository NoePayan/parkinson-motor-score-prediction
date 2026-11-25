# %% [markdown]
# # EDA: Parkinson's Disease Motor Score (true OFF)
#
# Exploratory data analysis of X_train / y_train / X_test, run before
# designing a model.
#
# - **Raw data** is read-only — `RAW_*` variables point at the CSVs.
#   This file never cleans or modifies them.
# - **Outputs** go under `EDA_DIR` (the repo's `data/`): one
#   `eda_<table>.html` report per table, summarised in `eda.md`.

# %%
import json
from pathlib import Path

import pandas as pd
import skrub

# This file lives in data/, so parents[1] is the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# EDA outputs always land here (created if missing).
EDA_DIR = PROJECT_ROOT / "data"
EDA_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Load the raw data
#
# Three tables: X_train (features), y_train (target), X_test (hold-out).
# Merge X_train + y_train on Index to get the labelled visits table.

# %%
RAW_X_TRAIN = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv")
RAW_Y_TRAIN = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv")
RAW_X_TEST = pd.read_csv(PROJECT_ROOT / "data" / "X_test.csv")

# Labelled visits (train + target)
RAW_TRAIN = RAW_X_TRAIN.merge(RAW_Y_TRAIN, on="Index")

{
    "X_train": RAW_X_TRAIN.shape,
    "y_train": RAW_Y_TRAIN.shape,
    "train_merged": RAW_TRAIN.shape,
    "X_test": RAW_X_TEST.shape,
}  # noqa: B018

# %% [markdown]
# ## Table overview — training visits
#
# Per-column summary: dtype, fraction missing, unique counts.

# %%
report_train = skrub.TableReport(RAW_TRAIN, title="train_visits", verbose=0)
report_train.write_html(EDA_DIR / "eda_train.html")

summary_train = json.loads(report_train.json())
n_rows_train = summary_train.get("n_rows")
overview_train = [
    {
        "column": col.get("name"),
        "dtype": col.get("dtype"),
        "null_pct": col.get("null_proportion"),
        "n_unique": col.get("n_unique"),
    }
    for col in summary_train.get("columns", [])
]
{"n_rows": n_rows_train, "n_columns": len(overview_train), "columns": overview_train}  # noqa: B018

# %% [markdown]
# ## Table overview — test visits
#
# Same profile on X_test so we can compare missingness and ranges.

# %%
report_test = skrub.TableReport(RAW_X_TEST, title="test_visits", verbose=0)
report_test.write_html(EDA_DIR / "eda_test.html")

summary_test = json.loads(report_test.json())
n_rows_test = summary_test.get("n_rows")
overview_test = [
    {
        "column": col.get("name"),
        "dtype": col.get("dtype"),
        "null_pct": col.get("null_proportion"),
        "n_unique": col.get("n_unique"),
    }
    for col in summary_test.get("columns", [])
]
{"n_rows": n_rows_test, "n_columns": len(overview_test), "columns": overview_test}  # noqa: B018

# %% [markdown]
# ## Target distribution
#
# Spread and skew of the true OFF score — shapes the metric choice.

# %%
TARGET = "target"
target_col = next(
    (col for col in summary_train.get("columns", []) if col.get("name") == TARGET),
    None,
)
target_col  # noqa: B018

# %% [markdown]
# ## Patient-level structure
#
# How many patients / visits are in train vs test; visits per patient.
# `patient_id` is the group variable for GroupKFold.

# %%
train_patient_stats = {
    "n_patients_train": int(RAW_TRAIN["patient_id"].nunique()),
    "n_visits_train": len(RAW_TRAIN),
    "visits_per_patient_mean": round(
        len(RAW_TRAIN) / RAW_TRAIN["patient_id"].nunique(), 2
    ),
    "visits_per_patient_max": int(RAW_TRAIN.groupby("patient_id").size().max()),
    "n_patients_test": int(RAW_X_TEST["patient_id"].nunique()),
    "n_visits_test": len(RAW_X_TEST),
    "patients_overlap_train_test": int(
        len(
            set(RAW_TRAIN["patient_id"]) & set(RAW_X_TEST["patient_id"])
        )
    ),
}
train_patient_stats  # noqa: B018

# %% [markdown]
# ## Structure signals
#
# High-cardinality columns (id / group candidates) and datetime columns.

# %%
datetime_cols = [
    col.get("name")
    for col in summary_train.get("columns", [])
    if "date" in str(col.get("dtype", "")).lower()
]
unique_ratio = sorted(
    (
        {
            "column": col.get("name"),
            "unique_ratio": (col.get("n_unique") or 0) / n_rows_train
            if n_rows_train
            else None,
        }
        for col in summary_train.get("columns", [])
    ),
    key=lambda r: (r["unique_ratio"] is not None, r["unique_ratio"]),
    reverse=True,
)
{"datetime_cols": datetime_cols, "top_unique_ratio": unique_ratio[:10]}  # noqa: B018

# %% [markdown]
# ## Missingness detail
#
# Which columns have missing data and by how much — both in train and test.

# %%
def _miss(overview):
    return sorted(
        [
            {"column": c["column"], "null_pct": c["null_pct"]}
            for c in overview
            if c["null_pct"] and c["null_pct"] > 0
        ],
        key=lambda r: r["null_pct"],
        reverse=True,
    )

missing_train = _miss(overview_train)
missing_test = _miss(overview_test)

# Visits where both on and off are missing (train)
both_missing_train = int(
    (RAW_TRAIN["on"].isna() & RAW_TRAIN["off"].isna()).sum()
)

{
    "missing_train": missing_train,
    "missing_test": missing_test,
    "visits_both_on_off_missing_train": both_missing_train,
}  # noqa: B018

# %% [markdown]
# ## Associations
#
# Strongest pairwise associations — flags predictors and leakage risks.

# %%
assoc = skrub.column_associations(RAW_TRAIN)
rows = assoc.to_dicts() if hasattr(assoc, "to_dicts") else assoc.to_dict(orient="records")
target_links = [
    row
    for row in rows
    if row["left_column_name"] == TARGET or row["right_column_name"] == TARGET
]
{"with_target": target_links[:15], "strongest": rows[:10]}  # noqa: B018

# %% [markdown]
# ## Summary
#
# Findings and modelling implications are written up in `data/eda.md`.
