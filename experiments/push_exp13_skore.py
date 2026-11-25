# push_exp13_skore.py — push EXP 13 EstimatorReport to Skore Hub
#
# Standalone script: rebuilds the EXP 13 feature set on a small sample
# (train[:2000] / test[:500]) and pushes an EstimatorReport to Skore Hub
# under the key "13_patient_context" (same pattern as EXP 06-11).
#
# Run: python experiments/push_exp13_skore.py

import io
import sys

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import skore
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split

from parkinson import PROJECT_ROOT
from parkinson.hub import load_skore_credentials

# ── EXP 11 column sets ────────────────────────────────────────────────────────
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
CAT_COLS   = ["cohort", "gene", "rater_id"]
EXP11_COLS = NUM_COLS + DOMAIN_COLS + TEMPORAL_COLS + CAT_COLS

CONTEXT_VARS = ["on", "off", "ledd", "time_since_intake_on", "time_since_intake_off"]

# EXP 11 best hyperparameters
BEST_PARAMS = {
    "learning_rate":     0.04885694382473504,
    "num_leaves":        50,
    "max_depth":         15,
    "min_child_samples": 40,
    "reg_lambda":        0.9117967494611006,
    "reg_alpha":         0.08968227868317444,
}


# ── LightGBM wrapper (same as EXP 06-13) ─────────────────────────────────────
class LGBMCatRegressor(BaseEstimator, RegressorMixin):
    """LightGBM with categorical_feature baked into fit()."""

    def __init__(self, cat_cols, n_estimators=500, **params):
        self.cat_cols     = cat_cols
        self.n_estimators = n_estimators
        self._params      = params
        for k, v in params.items():
            setattr(self, k, v)

    def fit(self, X, y):
        self.model_ = LGBMRegressor(
            n_estimators=self.n_estimators,
            random_state=42,
            deterministic=True,
            verbose=-1,
            **self._params,
        )
        self.model_.fit(X, y, categorical_feature=self.cat_cols)
        return self

    def predict(self, X):
        return self.model_.predict(X)


# ── Feature helpers ───────────────────────────────────────────────────────────
def _impute_column(df, col, patient_id, fallback_medians=None):
    df = df.copy()
    if df[col].isna().sum() == 0:
        return df
    if fallback_medians is None:
        pat_med = df.groupby(patient_id)[col].transform("median")
        mask    = df[col].isna()
        df.loc[mask, col] = pat_med[mask]
        if df[col].isna().sum() > 0:
            rat_med = df.groupby("rater_id")[col].transform("median")
            mask    = df[col].isna()
            df.loc[mask, col] = rat_med[mask]
        df[col] = df[col].fillna(df[col].median())
    else:
        pm = fallback_medians.get(f"{col}_patient", {})
        rm = fallback_medians.get(f"{col}_rater",   {})
        gm = fallback_medians.get(f"{col}_global",  0.0)
        mask = df[col].isna()
        df.loc[mask, col] = patient_id[mask].map(pm)
        mask = df[col].isna()
        df.loc[mask, col] = df.loc[mask, "rater_id"].map(rm)
        df[col] = df[col].fillna(gm)
    return df


def build_exp11_features(df, patient_id, rater_bias_dict,
                          velocity_dict, accel_dict, rater_consistency_dict,
                          impute_medians=None):
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
    return out[EXP11_COLS]


def build_context_features(df_all: pd.DataFrame) -> pd.DataFrame:
    """Build all patient-context features (train+test concat, no label)."""
    assert "target" not in df_all.columns
    out = pd.DataFrame(index=df_all.index)
    pid = df_all["patient_id"]

    for v in CONTEXT_VARS:
        col = df_all[v].copy()
        grp = col.groupby(pid)
        out[f"{v}_pat_mean"]   = grp.transform("mean")
        out[f"{v}_pat_std"]    = grp.transform("std").fillna(0.0)
        out[f"{v}_pat_min"]    = grp.transform("min")
        out[f"{v}_pat_max"]    = grp.transform("max")
        out[f"{v}_pat_median"] = grp.transform("median")
        out[f"{v}_pat_count"]  = grp.transform("count")
        out[f"{v}_dev"]        = col - out[f"{v}_pat_mean"]
        for shift in [1, 2]:
            lag = col.groupby(pid).shift(shift)
            out[f"{v}_lag{shift}"]      = lag
            out[f"{v}_lag{shift}_diff"] = col - lag
            lead = col.groupby(pid).shift(-shift)
            out[f"{v}_lead{shift}"]      = lead
            out[f"{v}_lead{shift}_diff"] = col - lead
        out[f"{v}_roll3_mean"] = col.groupby(pid).transform(
            lambda s: s.rolling(3, min_periods=1, center=True).mean())
        out[f"{v}_roll3_std"]  = col.groupby(pid).transform(
            lambda s: s.rolling(3, min_periods=1, center=True).std().fillna(0))

    for v in ["on", "off"]:
        col    = df_all[v].copy()
        ages   = df_all["age"]
        interp = np.full(len(df_all), np.nan)
        slope  = np.full(len(df_all), np.nan)
        resid  = np.full(len(df_all), np.nan)
        for pid_val, grp_idx in df_all.groupby("patient_id").groups.items():
            grp_ages = ages.loc[grp_idx].values.astype(float)
            grp_vals = col.loc[grp_idx].values.astype(float)
            valid    = ~np.isnan(grp_vals)
            if valid.sum() >= 2:
                interp_vals = np.interp(grp_ages, grp_ages[valid], grp_vals[valid])
                interp[df_all.index.get_indexer(grp_idx)] = interp_vals
                x  = grp_ages[valid]
                yy = grp_vals[valid]
                sl = float(np.polyfit(x - x.mean(), yy, 1)[0])
                slope[df_all.index.get_indexer(grp_idx)] = sl
                pred_line = sl * (grp_ages - x.mean()) + yy.mean()
                resid[df_all.index.get_indexer(grp_idx)] = interp_vals - pred_line
            elif valid.sum() == 1:
                interp[df_all.index.get_indexer(grp_idx)] = grp_vals[valid][0]
        out[f"{v}_interp"]      = interp
        out[f"{v}_slope"]       = slope
        out[f"{v}_slope_resid"] = resid

    out["on_off_gap"]            = df_all["off"] - df_all["on"]
    out["on_off_gap_interp"]     = out["off_interp"] - out["on_interp"]
    out["visit_rank"]            = df_all.groupby("patient_id").cumcount()
    out["n_visits"]              = df_all.groupby("patient_id")["age"].transform("count")
    age_min = df_all.groupby("patient_id")["age"].transform("min")
    age_max = df_all.groupby("patient_id")["age"].transform("max")
    out["followup_years"]        = df_all["age"] - age_min
    out["total_followup_yrs"]    = age_max - age_min

    ledd_slope = np.full(len(df_all), np.nan)
    for pid_val, grp_idx in df_all.groupby("patient_id").groups.items():
        ledd_v = df_all.loc[grp_idx, "ledd"].values.astype(float)
        ages_v = df_all.loc[grp_idx, "age"].values.astype(float)
        valid  = ~np.isnan(ledd_v)
        if valid.sum() >= 2:
            sl = float(np.polyfit(ages_v[valid] - ages_v[valid].mean(), ledd_v[valid], 1)[0])
            ledd_slope[df_all.index.get_indexer(grp_idx)] = sl
    out["ledd_per_year"]   = ledd_slope
    out["off_missing_rate"] = (df_all["off"].isna().astype(float)
                               .groupby(pid).transform("mean"))
    out["rater_n_visits"]   = df_all.groupby("rater_id")["age"].transform("count")
    return out


# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Step 1: loading data...")
sys.stdout.flush()

X_train_raw = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv")
y_df        = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv")
X_test_raw  = pd.read_csv(PROJECT_ROOT / "data" / "X_test.csv")

train_raw = X_train_raw.merge(y_df, on="Index")

print(f"  train shape: {train_raw.shape}")
sys.stdout.flush()

# ── 2. Compute EXP 11 domain statistics ───────────────────────────────────────
print("Step 2: computing domain + temporal statistics...")
sys.stdout.flush()

velocity_dict = {}
accel_dict    = {}
for pid_val, grp in train_raw.groupby("patient_id"):
    vals = grp.sort_values("time_since_intake_on")["off"].dropna().values
    if len(vals) >= 3:
        x = np.arange(len(vals))
        velocity_dict[pid_val] = float(np.polyfit(x, vals, 1)[0]) if vals.std() > 0 else 0.0
        accel_dict[pid_val]    = float(np.polyfit(x, vals, 2)[0]) if len(vals) >= 4 and vals.std() > 0 else 0.0
    else:
        velocity_dict[pid_val] = 0.0
        accel_dict[pid_val]    = 0.0

rater_cons_dict = train_raw.groupby("rater_id")["off"].std().fillna(0).to_dict()
rater_bias_dict = train_raw.groupby("rater_id")["target"].mean().to_dict()

impute_medians: dict = {}
for col in ["off", "on", "time_since_intake_off"]:
    impute_medians[f"{col}_patient"] = train_raw.groupby("patient_id")[col].median().to_dict()
    impute_medians[f"{col}_rater"]   = train_raw.groupby("rater_id")[col].median().to_dict()
    impute_medians[f"{col}_global"]  = train_raw[col].median()

print("  Done.")
sys.stdout.flush()

# ── 3. Build context features on train+test concat ────────────────────────────
print("Step 3: building patient-context features (train+test concat) ...")
sys.stdout.flush()

train_ctx = train_raw.drop(columns=["target"]).copy()
train_ctx["_is_test"] = 0
test_ctx              = X_test_raw.copy()
test_ctx["_is_test"]  = 1

all_df = (pd.concat([train_ctx, test_ctx], ignore_index=True)
          .sort_values(["patient_id", "age"])
          .reset_index(drop=True))

ctx_feats   = build_context_features(all_df)
all_df["_ctx_idx"] = np.arange(len(all_df))

ctx_cols_d = list(ctx_feats.columns)   # full variant D (best)

# Re-align to train rows
train_merged = train_raw.merge(all_df[["Index", "_ctx_idx"]], on="Index").sort_values("Index").reset_index(drop=True)
ctx_train    = ctx_feats.iloc[train_merged["_ctx_idx"].values].reset_index(drop=True)

print(f"  context features: {ctx_feats.shape[1]} cols")
sys.stdout.flush()

# ── 4. Build sample feature matrices for EstimatorReport ─────────────────────
print("Step 4: building sample feature matrices (train[:2000] / test[:500]) ...")
sys.stdout.flush()

y_full = train_raw["target"].values

# Hold-out split from labelled train data (same seed as EXP 11)
idx_tr, idx_te = train_test_split(np.arange(len(train_raw)), test_size=0.2, random_state=42)
tr_s_idx = idx_tr[:2000]
te_s_idx = idx_te[:500]

# EXP 11 features
X_tr_11 = build_exp11_features(
    train_raw.iloc[tr_s_idx], train_raw.iloc[tr_s_idx]["patient_id"],
    rater_bias_dict, velocity_dict, accel_dict, rater_cons_dict)
X_te_11 = build_exp11_features(
    train_raw.iloc[te_s_idx], train_raw.iloc[te_s_idx]["patient_id"],
    rater_bias_dict, velocity_dict, accel_dict, rater_cons_dict)

# Context features
ctx_tr_s = ctx_train.iloc[tr_s_idx][ctx_cols_d].reset_index(drop=True)
ctx_te_s = ctx_train.iloc[te_s_idx][ctx_cols_d].reset_index(drop=True)

X_tr_s = pd.concat([X_tr_11.reset_index(drop=True), ctx_tr_s], axis=1)
X_te_s = pd.concat([X_te_11.reset_index(drop=True), ctx_te_s], axis=1)

for c in ctx_cols_d:
    X_tr_s[c] = X_tr_s[c].astype(float)
    X_te_s[c] = X_te_s[c].astype(float)

y_tr_s = y_full[tr_s_idx]
y_te_s = y_full[te_s_idx]

all_feature_cols = EXP11_COLS + ctx_cols_d
print(f"  sample shapes: X_tr={X_tr_s.shape}, X_te={X_te_s.shape}")
print(f"  total features: {len(all_feature_cols)}")
sys.stdout.flush()

# ── 5. Push EstimatorReport to Skore Hub ──────────────────────────────────────
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

    report = skore.EstimatorReport(
        estimator=LGBMCatRegressor(
            cat_cols=CAT_COLS,
            n_estimators=500,
            **BEST_PARAMS,
        ),
        X_train=X_tr_s,
        y_train=y_tr_s,
        X_test=X_te_s,
        y_test=y_te_s,
    )

    project.put("13_patient_context", report)

    hub_url = "https://skore.probabl.ai/projects/bobathon-esilv/items/13_patient_context"
    print("[OK] Pushed EstimatorReport to Skore Hub")
    sys.stdout.flush()
    print(f"[REPORT] {hub_url}")
    sys.stdout.flush()

    # Save URL to logs
    url_path = PROJECT_ROOT / "logs" / "13_skore_hub_url.txt"
    url_path.write_text(hub_url + "\n", encoding="utf-8")
    print(f"[OK] URL saved -> {url_path}")
    sys.stdout.flush()

except Exception as exc:
    print(f"[WARN] Skore Hub push failed: {exc}")
    sys.stdout.flush()

print("\n=== Done ===")
print(f"  EstimatorReport key : 13_patient_context")
print(f"  Feature count       : {len(all_feature_cols)}")
print(f"  Sample sizes        : train={len(X_tr_s)}, test={len(X_te_s)}")
sys.stdout.flush()
