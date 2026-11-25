# diag_data.py — full diagnostic of the Parkinson dataset
#
# Prints everything to stdout (and saves to logs/diag.txt).
# No heavy output files — pure text.
#
# Sections
# --------
# 1. train.info() and train.describe() (rounded to 3 dp)
# 2. test.info()
# 3. First 10 rows of train and test
# 4. Target distribution: describe, skew, quantiles, n_unique
# 5. Patient overlap: train / test / both; avg visits per patient
# 6. Missing values (% train vs % test)
# 7. Spearman correlation of every feature with target (top 15)
# 8. OOF RMSE by cohort / gene / rater + 20 worst errors
# 9. First 5 rows of the best submission file

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold

# --- rebuild EXP 11 feature engineering inline (no heavy import) ---
from parkinson import PROJECT_ROOT

# ── constants ──────────────────────────────────────────────────────────────
LOG_PATH   = PROJECT_ROOT / "logs" / "diag.txt"
SUB_DIR    = PROJECT_ROOT / "submissions"
BEST_SUB   = "11_optuna_exp10_featureset.csv"   # best known CV RMSE

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
CAT_COLS     = ["cohort", "gene", "rater_id"]
FEATURE_COLS = NUM_COLS + DOMAIN_COLS + TEMPORAL_COLS + CAT_COLS  # 22 cols

# ── output helper ──────────────────────────────────────────────────────────
pd.set_option("display.max_columns",  None)
pd.set_option("display.max_rows",     None)
pd.set_option("display.width",        200)
pd.set_option("display.float_format", lambda x: f"{x:.4f}")

lines: list[str] = []

def pr(*args, **kw) -> None:
    """Print to stdout AND accumulate for the log file."""
    text = " ".join(str(a) for a in args)
    print(text, **kw)
    lines.append(text)

def section(title: str) -> None:
    bar = "=" * 72
    pr(f"\n{bar}")
    pr(f"  {title}")
    pr(bar)

# ── 0. Load raw data ────────────────────────────────────────────────────────
print("Loading data...", flush=True)
X_train_raw = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv")
y_df        = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv")
X_test_raw  = pd.read_csv(PROJECT_ROOT / "data" / "X_test.csv")

train = X_train_raw.merge(y_df, on="Index")
test  = X_test_raw.copy()

# ─────────────────────────────────────────────────────────────────────────────
# 1. train.info() and train.describe()
# ─────────────────────────────────────────────────────────────────────────────
section("1 — train.info() and train.describe()")

buf = io.StringIO()
train.info(buf=buf)
pr(buf.getvalue())

pr("\ntrain.describe().T (rounded to 3 dp):")
pr(train.describe().T.round(3).to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 2. test.info()
# ─────────────────────────────────────────────────────────────────────────────
section("2 — test.info()")

buf = io.StringIO()
test.info(buf=buf)
pr(buf.getvalue())

# ─────────────────────────────────────────────────────────────────────────────
# 3. First 10 rows
# ─────────────────────────────────────────────────────────────────────────────
section("3 — First 10 rows of train")
pr(train.head(10).to_string())

section("3 — First 10 rows of test")
pr(test.head(10).to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 4. Target distribution
# ─────────────────────────────────────────────────────────────────────────────
section("4 — Target distribution")
y = train["target"]

pr(f"describe:\n{y.describe().round(3).to_string()}")
pr(f"\nskewness : {y.skew():.4f}")
pr(f"kurtosis : {y.kurtosis():.4f}")
pr(f"n_unique : {y.nunique()}")

quantiles = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
pr("\nquantiles:")
for q in quantiles:
    pr(f"  {int(q*100):3d}%  {y.quantile(q):.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Patient overlap
# ─────────────────────────────────────────────────────────────────────────────
section("5 — Patient overlap train / test")

train_pids = set(train["patient_id"].unique())
test_pids  = set(test["patient_id"].unique())
common     = train_pids & test_pids

pr(f"Patients in train          : {len(train_pids):,}")
pr(f"Patients in test           : {len(test_pids):,}")
pr(f"Patients in both           : {len(common):,}")
pr(f"Train-only patients        : {len(train_pids - test_pids):,}")
pr(f"Test-only  patients        : {len(test_pids - train_pids):,}")

if common:
    avg_tr = train[train["patient_id"].isin(common)].groupby("patient_id").size().mean()
    avg_te = test[test["patient_id"].isin(common)].groupby("patient_id").size().mean()
    pr(f"\nAmong common patients:")
    pr(f"  avg visits in train : {avg_tr:.2f}")
    pr(f"  avg visits in test  : {avg_te:.2f}")

pr(f"\nOverall avg visits per patient (train): "
   f"{train.groupby('patient_id').size().mean():.2f}")
pr(f"Overall avg visits per patient (test) : "
   f"{test.groupby('patient_id').size().mean():.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Missing values
# ─────────────────────────────────────────────────────────────────────────────
section("6 — Missing values (% train vs % test)")

miss_train = (train.isnull().mean() * 100).round(2).rename("train_%")
miss_test  = (test.isnull().mean()  * 100).round(2).rename("test_%")
miss = pd.concat([miss_train, miss_test], axis=1)
miss = miss[miss["train_%"] + miss["test_%"] > 0].sort_values("train_%", ascending=False)

if miss.empty:
    pr("No missing values found.")
else:
    pr(miss.to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 7. Spearman correlations with target (top 15)
# ─────────────────────────────────────────────────────────────────────────────
section("7 — Spearman correlation with target (top 15)")

corr_rows = []
for col in train.columns:
    if col in ("Index", "patient_id", "target"):
        continue
    s = train[col].copy()
    # For categoricals: encode as integer codes, drop unknowns
    if s.dtype == object or str(s.dtype) == "category":
        codes = s.astype("category").cat.codes.astype(float)
        codes[codes < 0] = np.nan
        s = codes
    mask = s.notna() & train["target"].notna()
    if mask.sum() < 10:
        continue
    # Use pandas rank (handles ties/NaN) then np.corrcoef for reliability
    x = s[mask].rank().values.astype(float)
    y_r = train["target"][mask].rank().values.astype(float)
    r = float(np.corrcoef(x, y_r)[0, 1])
    # p-value via t-distribution approximation
    n = len(x)
    t = r * np.sqrt((n - 2) / max(1 - r**2, 1e-15))
    from scipy.stats import t as t_dist
    p = float(2 * t_dist.sf(abs(t), df=n - 2))
    corr_rows.append({"feature": col, "spearman_r": round(r, 4), "p_value": round(p, 6)})

corr_df = (pd.DataFrame(corr_rows)
           .sort_values("spearman_r", key=abs, ascending=False)
           .head(15)
           .reset_index(drop=True))
pr(corr_df.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 8. OOF RMSE by cohort / gene / rater + 20 worst errors
#    We rebuild features and run a single GroupKFold pass (EXP 11 best params)
# ─────────────────────────────────────────────────────────────────────────────
section("8 — OOF diagnostics (EXP 11 LightGBM best params, GroupKFold 5)")

try:
    from lightgbm import LGBMRegressor
    from sklearn.base import BaseEstimator, RegressorMixin

    BEST_PARAMS_LGB = {
        "learning_rate":     0.04885694382473504,
        "num_leaves":        50,
        "max_depth":         15,
        "min_child_samples": 40,
        "reg_lambda":        0.9117967494611006,
        "reg_alpha":         0.08968227868317444,
    }

    class LGBMCatRegressor(BaseEstimator, RegressorMixin):
        def __init__(self, cat_cols, **params):
            self.cat_cols = cat_cols
            for k, v in params.items():
                setattr(self, k, v)
            self._params = params
        def fit(self, X, y):
            self.model_ = LGBMRegressor(n_estimators=150, random_state=42, verbose=-1,
                                        **self._params)
            self.model_.fit(X, y, categorical_feature=self.cat_cols)
            return self
        def predict(self, X):
            return self.model_.predict(X)

    def _impute(df, col, pid, fallback=None):
        df = df.copy()
        if df[col].isna().sum() == 0:
            return df
        if fallback is None:
            pat = df.groupby(pid)[col].transform("median")
            m = df[col].isna(); df.loc[m, col] = pat[m]
            if df[col].isna().sum() > 0:
                rat = df.groupby("rater_id")[col].transform("median")
                m = df[col].isna(); df.loc[m, col] = rat[m]
            df[col] = df[col].fillna(df[col].median())
        else:
            pm = fallback.get(f"{col}_patient", {}); rm = fallback.get(f"{col}_rater", {})
            gm = fallback.get(f"{col}_global", 0.0)
            m = df[col].isna(); df.loc[m, col] = pid[m].map(pm)
            m = df[col].isna(); df.loc[m, col] = df.loc[m, "rater_id"].map(rm)
            df[col] = df[col].fillna(gm)
        return df

    def build_feats(df, pid, rater_bias, vel, acc, rater_cons):
        out = df[NUM_COLS + CAT_COLS].copy()
        for c in ["off", "on", "time_since_intake_off"]:
            out = _impute(out, c, pid)
        out["is_off_missing"] = df["off"].isna().astype(np.float32)
        out["is_on_missing"]  = df["on"].isna().astype(np.float32)
        out["rater_bias"]     = out["rater_id"].map(rater_bias).fillna(0.0).clip(-2, 2)
        out["off_x_age"]      = out["off"] * out["age"]
        out["off_x_time"]     = out["off"] * out["time_since_intake_on"]
        out["ledd_x_age"]     = out["ledd"] * out["age"]
        out["on_x_age"]       = out["on"]  * out["age"]
        out["patient_velocity_off"] = pid.map(vel).fillna(0.0)
        out["patient_accel_off"]    = pid.map(acc).fillna(0.0)
        out["rater_consistency"]    = out["rater_id"].map(rater_cons).fillna(0.0)
        out["visit_month"]          = (out["time_since_intake_on"].fillna(0) / 30.0).astype(int) % 12
        for c in CAT_COLS:
            out[c] = out[c].astype("category")
        return out[FEATURE_COLS]

    # Pre-compute temporal stats from full train
    vel, acc = {}, {}
    for pid_val, grp in train.groupby("patient_id"):
        vals = grp.sort_values("time_since_intake_on")["off"].dropna().values
        if len(vals) >= 3:
            x = np.arange(len(vals))
            vel[pid_val] = float(np.polyfit(x, vals, 1)[0]) if vals.std() > 0 else 0.0
            acc[pid_val] = float(np.polyfit(x, vals, 2)[0]) if len(vals) >= 4 and vals.std() > 0 else 0.0
        else:
            vel[pid_val] = 0.0; acc[pid_val] = 0.0

    rater_cons = train.groupby("rater_id")["off"].std().fillna(0).to_dict()

    y_full  = train["target"].values
    groups  = train["patient_id"].values
    cv      = GroupKFold(n_splits=5)
    oof     = np.full(len(train), np.nan)

    pr("  Building OOF predictions (5 folds) — takes ~2 min ...")
    sys.stdout.flush()

    for fold_i, (tr_idx, va_idx) in enumerate(cv.split(train, y_full, groups=groups)):
        tr_v = train.iloc[tr_idx]; va_v = train.iloc[va_idx]
        rb   = tr_v.groupby("rater_id")["target"].mean().to_dict()
        X_tr = build_feats(tr_v, tr_v["patient_id"], rb, vel, acc, rater_cons)
        X_va = build_feats(va_v, va_v["patient_id"], rb, vel, acc, rater_cons)
        est  = LGBMCatRegressor(cat_cols=CAT_COLS, **BEST_PARAMS_LGB)
        est.fit(X_tr, y_full[tr_idx])
        oof[va_idx] = est.predict(X_va)
        rmse_f = np.sqrt(np.mean((y_full[va_idx] - oof[va_idx])**2))
        pr(f"  fold {fold_i+1}/5 RMSE={rmse_f:.4f}"); sys.stdout.flush()

    oof_rmse = np.sqrt(np.mean((y_full - oof)**2))
    pr(f"\n  OOF RMSE (all folds): {oof_rmse:.4f}")

    # Attach OOF to train slice
    diag = train[["Index", "patient_id", "cohort", "gene", "rater_id", "target"]].copy()
    diag["oof_pred"] = oof
    diag["abs_err"]  = np.abs(diag["target"] - diag["oof_pred"])

    def rmse_group(df, col):
        rows = []
        for val, grp in df.groupby(col):
            r = np.sqrt(np.mean((grp["target"] - grp["oof_pred"])**2))
            rows.append({col: val, "n": len(grp), "RMSE": round(r, 4)})
        return pd.DataFrame(rows).sort_values("RMSE", ascending=False)

    pr("\n  RMSE by cohort:")
    pr(rmse_group(diag, "cohort").to_string(index=False))

    pr("\n  RMSE by gene:")
    pr(rmse_group(diag, "gene").to_string(index=False))

    pr("\n  RMSE by rater (top 15 worst):")
    pr(rmse_group(diag, "rater_id").head(15).to_string(index=False))

    pr("\n  20 worst OOF errors:")
    worst = (diag.nlargest(20, "abs_err")
             [["Index", "patient_id", "cohort", "gene", "rater_id",
               "target", "oof_pred", "abs_err"]]
             .round(3))
    pr(worst.to_string(index=False))

except Exception as exc:
    pr(f"  [WARN] OOF section failed: {exc}")

# ─────────────────────────────────────────────────────────────────────────────
# 9. First 5 rows of best submission
# ─────────────────────────────────────────────────────────────────────────────
section(f"9 — First 5 rows of best submission ({BEST_SUB})")

sub_path = SUB_DIR / BEST_SUB
if sub_path.exists():
    sub = pd.read_csv(sub_path)
    pr(f"  shape: {sub.shape}")
    pr(sub.head(5).to_string(index=False))
    pr(f"\n  y stats: mean={sub.iloc[:,1].mean():.3f}  std={sub.iloc[:,1].std():.3f}  "
       f"min={sub.iloc[:,1].min():.3f}  max={sub.iloc[:,1].max():.3f}")
else:
    pr(f"  [WARN] submission file not found: {sub_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Save to logs/diag.txt
# ─────────────────────────────────────────────────────────────────────────────
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
LOG_PATH.write_text("\n".join(lines), encoding="utf-8")
print(f"\n[OK] Saved to {LOG_PATH}", flush=True)
