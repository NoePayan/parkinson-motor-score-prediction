# 12 — XGBoost (device='cuda') + Optuna on the full 22-col feature set
#
# EXP 11 found the best LightGBM params on the same 22-col feature set.
# This experiment swaps LightGBM for XGBoost (device='cuda') and
# re-tunes 30 Optuna trials on the same GroupKFold(5) splitter.
# Benchmark to beat: EXP 10 CV RMSE = 6.5791 ± 0.0859.
#
# Feature set (22 cols — identical to EXP 10/11):
#   8 numeric + 7 domain + 4 temporal + 3 cats (cohort, gene, rater_id)
#
# XGBoost 3.x note: enable_categorical=True requires pandas category dtype;
# device='cuda' replaces the deprecated tree_method='gpu_hist' + gpu_id.
#
# Steps
# -----
# 1. Load data; compute domain + temporal statistics from full train.
# 2. Build 22-col feature matrix (cats as pandas category dtype).
# 3. Run Optuna study (30 trials, GroupKFold 5-fold, fold-safe rater_bias).
# 4. Re-run best params on 5 folds to compute CV RMSE ± std.
# 5. Fit final model on 100 % of training data; print top-10 importances.
# 6. Predict on X_test and write submissions/12_xgboost_gpu.csv.
# 7. Save logs: best_params.json, cv_rmse.txt.
# 8. Push EstimatorReport (train[:2000] / test[:500]) to Skore Hub.

import io
import json
import sys

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import optuna
import pandas as pd
import skore
from sklearn.model_selection import GroupKFold, train_test_split
from xgboost import XGBRegressor

from parkinson import PROJECT_ROOT
from parkinson.hub import load_skore_credentials

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Column definitions (same as EXP 10/11)
# ---------------------------------------------------------------------------
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
CAT_COLS = ["cohort", "gene", "rater_id"]
FEATURE_COLS = NUM_COLS + DOMAIN_COLS + TEMPORAL_COLS + CAT_COLS  # 22 cols


# ---------------------------------------------------------------------------
# Feature engineering helpers (identical to EXP 11)
# ---------------------------------------------------------------------------
def _impute_column(df, col, patient_id, fallback_medians=None):
    df = df.copy()
    if df[col].isna().sum() == 0:
        return df
    if fallback_medians is None:
        pat_med = df.groupby(patient_id)[col].transform("median")
        mask = df[col].isna()
        df.loc[mask, col] = pat_med[mask]
        if df[col].isna().sum() > 0:
            rater_med = df.groupby("rater_id")[col].transform("median")
            mask = df[col].isna()
            df.loc[mask, col] = rater_med[mask]
        df[col] = df[col].fillna(df[col].median())
    else:
        pat_med_map   = fallback_medians.get(f"{col}_patient", {})
        rater_med_map = fallback_medians.get(f"{col}_rater", {})
        global_med    = fallback_medians.get(f"{col}_global", 0.0)
        mask = df[col].isna()
        df.loc[mask, col] = patient_id[mask].map(pat_med_map)
        mask = df[col].isna()
        df.loc[mask, col] = df.loc[mask, "rater_id"].map(rater_med_map)
        df[col] = df[col].fillna(global_med)
    return df


def build_features(df, patient_id, rater_bias_dict,
                   velocity_dict, accel_dict, rater_consistency_dict,
                   impute_medians=None):
    """Build 22-col feature matrix (EXP 10 set).

    XGBoost note: categorical columns are kept as plain str (object dtype) so
    that XGBRegressor(enable_categorical=True) can handle them natively via the
    GPU hist method without requiring any manual label encoding.
    """
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

    # XGBoost 3.x requires pandas category dtype for enable_categorical=True.
    for col in CAT_COLS:
        out[col] = out[col].astype("category")

    return out[FEATURE_COLS]


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("Step 1: loading data...")
sys.stdout.flush()

X_train_raw = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv")
y_df        = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv")
X_test_raw  = pd.read_csv(PROJECT_ROOT / "data" / "X_test.csv")

visits = X_train_raw.merge(y_df, on="Index")
patient_id_train = visits["patient_id"]
y_full = visits["target"].values

print(f"  visits shape: {visits.shape}")
sys.stdout.flush()

# ---------------------------------------------------------------------------
# 2. Compute per-patient / per-rater statistics (from full train)
# ---------------------------------------------------------------------------
print("Step 2: computing domain + temporal feature statistics...")
sys.stdout.flush()

velocity_dict = {}
accel_dict = {}
for pid, grp in visits.groupby("patient_id"):
    vals = grp.sort_values("time_since_intake_on")["off"].dropna().values
    if len(vals) >= 3:
        x = np.arange(len(vals))
        velocity_dict[pid] = float(np.polyfit(x, vals, 1)[0]) if vals.std() > 0 else 0.0
        accel_dict[pid]    = float(np.polyfit(x, vals, 2)[0]) if len(vals) >= 4 and vals.std() > 0 else 0.0
    else:
        velocity_dict[pid] = 0.0
        accel_dict[pid]    = 0.0

rater_consistency_dict = visits.groupby("rater_id")["off"].std().fillna(0).to_dict()
rater_bias_dict        = visits.groupby("rater_id")["target"].mean().to_dict()

impute_medians = {}
for col in ["off", "on", "time_since_intake_off"]:
    impute_medians[f"{col}_patient"] = visits.groupby("patient_id")[col].median().to_dict()
    impute_medians[f"{col}_rater"]   = visits.groupby("rater_id")[col].median().to_dict()
    impute_medians[f"{col}_global"]  = visits[col].median()

# ---------------------------------------------------------------------------
# 3. Build full feature matrix
# ---------------------------------------------------------------------------
print("Step 3: building feature matrix...")
sys.stdout.flush()

X_full = build_features(visits, patient_id_train, rater_bias_dict,
                         velocity_dict, accel_dict, rater_consistency_dict)
groups = patient_id_train.values

print(f"  X shape: {X_full.shape}, columns: {FEATURE_COLS}")
sys.stdout.flush()

# ---------------------------------------------------------------------------
# 4. Optuna objective — fold-safe rater_bias, GroupKFold(5)
# ---------------------------------------------------------------------------
cv = GroupKFold(n_splits=5)


import time as _time

_study_start = _time.time()
N_TRIALS = 30


def objective(trial: optuna.Trial) -> float:
    t0 = _time.time()
    params = {
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "max_depth":        trial.suggest_int("max_depth", 4, 15),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-3, 2.0, log=True),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
    }
    print(f"  [Trial {trial.number+1:02d}/{N_TRIALS}] params: lr={params['learning_rate']:.4f} "
          f"depth={params['max_depth']} sub={params['subsample']:.2f} "
          f"col={params['colsample_bytree']:.2f}", flush=True)

    fold_rmses = []
    for fold_i, (tr_idx, va_idx) in enumerate(cv.split(X_full, y_full, groups=groups)):
        fold_visits_tr = visits.iloc[tr_idx]
        fold_visits_va = visits.iloc[va_idx]
        fold_rater_bias = fold_visits_tr.groupby("rater_id")["target"].mean().to_dict()

        X_tr = build_features(fold_visits_tr, fold_visits_tr["patient_id"],
                               fold_rater_bias, velocity_dict, accel_dict,
                               rater_consistency_dict)
        X_va = build_features(fold_visits_va, fold_visits_va["patient_id"],
                               fold_rater_bias, velocity_dict, accel_dict,
                               rater_consistency_dict)

        est = XGBRegressor(
            device="cuda",
            n_estimators=150,
            enable_categorical=True,
            random_state=42,
            n_jobs=1,
            verbosity=0,
            **params,
        )
        est.fit(X_tr, y_full[tr_idx])
        preds = est.predict(X_va)
        rmse = float(np.sqrt(np.mean((y_full[va_idx] - preds) ** 2)))
        fold_rmses.append(rmse)
        print(f"    fold {fold_i+1}/5 RMSE={rmse:.4f}", flush=True)

    mean_rmse = float(np.mean(fold_rmses))
    elapsed = _time.time() - t0
    total_elapsed = _time.time() - _study_start
    done = trial.number + 1
    eta = (total_elapsed / done) * (N_TRIALS - done)
    print(f"  [Trial {trial.number+1:02d}/{N_TRIALS}] CV RMSE={mean_rmse:.4f}  "
          f"({elapsed:.0f}s this trial, ETA ~{eta/60:.1f}min remaining)", flush=True)
    return mean_rmse


# ---------------------------------------------------------------------------
# Step 4: Run Optuna study (30 trials)
# ---------------------------------------------------------------------------
print("Step 4: running Optuna study (30 trials, XGBoost GPU hist)...")
sys.stdout.flush()

study = optuna.create_study(
    direction="minimize",
    pruner=optuna.pruners.MedianPruner(),
)
study.optimize(objective, n_trials=N_TRIALS, n_jobs=1, show_progress_bar=False)

best_params  = study.best_params
best_cv_rmse = study.best_value

# Compute per-trial RMSE to get std across best trial's folds (re-run best params once)
print(f"  Best CV RMSE : {best_cv_rmse:.4f}")
print(f"  Best params  : {best_params}")
sys.stdout.flush()

# Re-run the best params over folds to capture per-fold RMSE for ± std reporting
print("  Re-running best params on 5 folds to compute ± std...")
sys.stdout.flush()
fold_rmses_best = []
for fold_i, (tr_idx, va_idx) in enumerate(cv.split(X_full, y_full, groups=groups)):
    print(f"    re-run fold {fold_i+1}/5 ...", flush=True)
    fold_visits_tr = visits.iloc[tr_idx]
    fold_visits_va = visits.iloc[va_idx]
    fold_rater_bias = fold_visits_tr.groupby("rater_id")["target"].mean().to_dict()

    X_tr = build_features(fold_visits_tr, fold_visits_tr["patient_id"],
                           fold_rater_bias, velocity_dict, accel_dict,
                           rater_consistency_dict)
    X_va = build_features(fold_visits_va, fold_visits_va["patient_id"],
                           fold_rater_bias, velocity_dict, accel_dict,
                           rater_consistency_dict)

    est = XGBRegressor(
        device="cuda",
        n_estimators=150,
        enable_categorical=True,
        random_state=42,
        n_jobs=1,
        verbosity=0,
        **best_params,
    )
    est.fit(X_tr, y_full[tr_idx])
    preds = est.predict(X_va)
    rmse = float(np.sqrt(np.mean((y_full[va_idx] - preds) ** 2)))
    fold_rmses_best.append(rmse)
    print(f"    re-run fold {fold_i+1}/5 RMSE={rmse:.4f}", flush=True)

cv_mean = float(np.mean(fold_rmses_best))
cv_std  = float(np.std(fold_rmses_best))
print(f"  Best CV RMSE (re-run): {cv_mean:.4f} ± {cv_std:.4f}")
sys.stdout.flush()

# ---------------------------------------------------------------------------
# Step 5: Fit final model with best params on 100 % of training data
# ---------------------------------------------------------------------------
print("Step 5: fitting final model on 100 % training data...")
sys.stdout.flush()

best_model = XGBRegressor(
    device="cuda",
    n_estimators=150,
    enable_categorical=True,
    random_state=42,
    n_jobs=1,
    verbosity=0,
    **best_params,
)
best_model.fit(X_full, y_full)
print("  Done.")
sys.stdout.flush()

# Feature importances (weight-based)
feat_imp = pd.Series(
    best_model.feature_importances_, index=FEATURE_COLS
).sort_values(ascending=False)
print("  Top-10 feature importances:")
print(feat_imp.head(10).to_string())
sys.stdout.flush()

# ---------------------------------------------------------------------------
# Step 6: Predict on X_test and save submission
# ---------------------------------------------------------------------------
print("Step 6: predicting on X_test and saving submission CSV...")
sys.stdout.flush()

X_test_feats = build_features(
    X_test_raw, X_test_raw["patient_id"],
    rater_bias_dict, velocity_dict, accel_dict,
    rater_consistency_dict, impute_medians,
)
y_pred_test = np.array(best_model.predict(X_test_feats))

sub_dir = PROJECT_ROOT / "submissions"
sub_dir.mkdir(parents=True, exist_ok=True)
sub_path = sub_dir / "12_xgboost_gpu.csv"
submission = pd.DataFrame({"Id": X_test_raw["Index"], "y": y_pred_test})
submission.to_csv(sub_path, index=False)

print(f"  Saved {len(submission)} predictions -> {sub_path}")
print(f"  Prediction sample (first 3): {y_pred_test[:3].round(4).tolist()}")
sys.stdout.flush()

# ---------------------------------------------------------------------------
# Step 7: Save logs
# ---------------------------------------------------------------------------
print("Step 7: saving logs...")
sys.stdout.flush()

logs_dir = PROJECT_ROOT / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)

# best_params.json
params_path = logs_dir / "12_xgboost_best_params.json"
with open(params_path, "w", encoding="utf-8") as fh:
    json.dump(best_params, fh, indent=2)
print(f"  Saved best params -> {params_path}")

# cv_rmse.txt
rmse_path = logs_dir / "12_xgboost_cv_rmse.txt"
with open(rmse_path, "w", encoding="utf-8") as fh:
    fh.write(f"Best CV RMSE: {cv_mean:.4f} ± {cv_std:.4f}\n")
    fh.write(f"Per-fold RMSE: {[round(r, 4) for r in fold_rmses_best]}\n")
    fh.write(f"EXP 10 benchmark: 6.5791 ± 0.0859\n")
    delta = cv_mean - 6.5791
    verdict = "BETTER" if delta < 0 else "WORSE"
    fh.write(f"Delta vs EXP 10: {delta:+.4f} ({verdict})\n")
print(f"  Saved CV RMSE -> {rmse_path}")
sys.stdout.flush()

# ---------------------------------------------------------------------------
# Step 8: Push EstimatorReport to Skore Hub
# ---------------------------------------------------------------------------
print("Step 8: pushing EstimatorReport to Skore Hub...")
sys.stdout.flush()

hub_url = "N/A"
try:
    cfg = load_skore_credentials()
    skore.login(mode="hub")

    project = skore.Project(
        name="bobathon-esilv",
        mode="hub",
        workspace=cfg["workspace"],
    )

    # Hold-out split from labelled train data (same pattern as EXP 11)
    idx_tr, idx_te = train_test_split(
        np.arange(len(X_full)), test_size=0.2, random_state=42
    )
    X_tr_s = X_full.iloc[idx_tr[:2000]]
    y_tr_s = y_full[idx_tr[:2000]]
    X_te_s = X_full.iloc[idx_te[:500]]
    y_te_s = y_full[idx_te[:500]]

    report = skore.EstimatorReport(
        estimator=XGBRegressor(
            device="cuda",
            n_estimators=150,
            enable_categorical=True,
            random_state=42,
            n_jobs=1,
            verbosity=0,
            **best_params,
        ),
        X_train=X_tr_s,
        y_train=y_tr_s,
        X_test=X_te_s,
        y_test=y_te_s,
    )

    project.put("12_xgboost_gpu", report)
    hub_url = "https://skore.probabl.ai/projects/bobathon-esilv/items/12_xgboost_gpu"

    url_path = logs_dir / "12_skore_hub_url.txt"
    with open(url_path, "w", encoding="utf-8") as fh:
        fh.write(hub_url + "\n")

    print(f"[OK] Pushed EstimatorReport to Skore Hub")
    print(f"[REPORT] {hub_url}")
    sys.stdout.flush()
except Exception as exc:
    print(f"[WARN] Skore Hub push failed: {exc}")
    sys.stdout.flush()

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
delta = cv_mean - 6.5791
verdict = "BETTER than EXP 10" if delta < 0 else "WORSE than EXP 10"
print("\n=== Final results ===")
print(f"  Best CV RMSE : {cv_mean:.4f} ± {cv_std:.4f}")
print(f"  EXP 10 bench : 6.5791 ± 0.0859")
print(f"  Delta        : {delta:+.4f}  →  {verdict}")
print(f"  Best params  : {best_params}")
print(f"  Submission   : {sub_path}")
print(f"  Skore Hub    : {hub_url}")
sys.stdout.flush()
