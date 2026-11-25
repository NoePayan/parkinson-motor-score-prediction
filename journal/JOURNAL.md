# ML Project Journal — Parkinson's Disease Motor Score

## Status

- **Goal:** Predict the debiased true OFF MDS-UPDRS motor score (`target`) for every patient visit.
- **Metric:** RMSE
- **Splitter:** GroupKFold(n_splits=5) on `patient_id`
- **Tabular library:** pandas
- **Best model:** EXP 13 — LGB + patient-context features — CV RMSE **4.1991 ± 0.0628**
- **Last updated:** 2026-10-12

## Workspace decisions

| Decision | Value | Date |
|---|---|---|
| Tabular library | pandas | 2025-10-10 |
| Cross-validator | GroupKFold(n_splits=5, groups=patient_id) | — |
| Primary metric | RMSE | — |
| Skore Hub report type | EstimatorReport on sample (train[:2000] / test[:500]) | 2026-10-11 |

> **Skore Hub sizing rule:** CrossValidationReport on 44K rows serialises to ~130 MB → Hub upload timeout.
> Always use `EstimatorReport` on a small sample (2 000 train / 500 test) → ~1.2 MB → push in < 30 s.

---

## Data understanding (EDA)

- **Status:** done - 2025-10-10
- **Summary:** 44 590 training visits across 5 576 patients (mean 8 visits each, max 12). Holdout is patient-level (zero overlap — 1 395 test patients distinct from all train patients). Target is continuous 0–110, mean 37.5, std 16.5, nearly symmetric (skew 0.05). Top predictors: `off` (Spearman r=0.87), `on` (r=0.67). Heavy missingness in `time_since_intake_off` (79 %), `time_since_intake_on` (46 %), `off` (42 %), `ledd` (37 %). GroupKFold on `patient_id` is mandatory.
- **Report:** [data/eda.md](../data/eda.md)

---

## Experiments

| # | Name | CV RMSE | Kaggle rank | Skore Hub | Notes |
|---|---|---|---|---|---|
| 01 | dummy_mean | 16.50 | — | — | Predicts global mean; establishes RMSE floor |
| 02 | histgb | 7.7501 ± 0.1415 | — | ✅ 2026-10-11 | HistGBR, 8 numeric features, GroupKFold |
| 03 | ridge | 10.3653 ± 0.0919 | — | ✅ 2026-10-11 | Ridge + OHE cats, KFold (no group — biased) |
| 04 | histgb_cats | 7.5521 ± 0.1301 | — | ✅ 2026-10-11 | HistGBR + native categoricals, GroupKFold |
| 05 | ridge_groupkfold | 10.4470 ± 0.2090 | — | ✅ 2026-10-11 | Ridge + OHE + GroupKFold; confirms Ridge < HistGBR |
| 06 | lightgbm_cats | 7.5439 ± 0.1335 | — | ✅ 2026-10-11 | LightGBM + native cats, GroupKFold; ≈ 04 |
| 07 | lightgbm_trajectory | 7.4793 ± 0.1312 | — | ✅ 2026-10-11 | LGB + visit trajectory (visit_count, time_span, off/on slope) |
| 08 | lightgbm_optuna | 7.5369 | — | ✅ 2026-10-11 | LGB + Optuna 30 trials on 11-col set; tuned < 06 default |
| 09 | lightgbm_domain_features | 6.7292 ± 0.0759 | — | ✅ 2026-10-11 | LGB + rater_bias + interactions + domain imputation |
| 10 | stacking_temporal | 6.5791 ± 0.0859 | — | ✅ 2026-10-11 | Stacking (LGB+HGB+Ridge) + temporal features |
| 11 | optuna_exp10_featureset | 6.5695 ± 0.0945 | — | ✅ 2026-10-11 | LGB Optuna v2 on 22-col EXP 10 feature set |
| 12 | xgboost_gpu | 6.5814 ± 0.0827 | — | ✅ 2026-10-11 | XGBoost device='cuda' + Optuna 30 trials, same 22-col set |
| **13** | **patient_context** | **4.1991 ± 0.0628** | — | ✅ 2026-10-12 | **LGB + patient-context features (agg/interp/lag/roll/traj) — best model** |

---

## Experiment details

### 01 — dummy_mean

- **Status:** done
- **Splitter:** GroupKFold(n_splits=5)
- **Model:** `DummyRegressor(strategy="mean")`
- **CV RMSE:** 16.50 (= target std, expected)

---

### 02 — histgb

- **Status:** done
- **Splitter:** GroupKFold(n_splits=5, groups=patient_id)
- **Per-fold RMSE:** [7.8272, 7.9277, 7.5694, 7.5964, 7.8299]
- **CV RMSE:** 7.7501 ± 0.1415
- **Features (8 numeric):** sexM, age_at_diagnosis, age, ledd, time_since_intake_on, time_since_intake_off, on, off
- **Hyperparameters:** `HistGradientBoostingRegressor(max_iter=100, learning_rate=0.1, random_state=42)`

---

### 03 — ridge

- **Status:** done
- **Splitter:** KFold(n_splits=5, shuffle=True, random_state=42) — no group split, CV is biased
- **Per-fold RMSE:** [10.5108, 10.3418, 10.4217, 10.2493, 10.3028]
- **CV RMSE:** 10.3653 ± 0.0919
- **Features:** 8 numeric + OHE(cohort, gene, rater_id) = 61 cols; patient_id dropped
- **Hyperparameters:** `Pipeline([SimpleImputer(median), StandardScaler, Ridge(alpha=1.0)])`

---

### 04 — histgb_cats

- **Status:** done
- **Splitter:** GroupKFold(n_splits=5, groups=patient_id)
- **Per-fold RMSE:** [7.6085, 7.6809, 7.3592, 7.4384, 7.6735]
- **CV RMSE:** 7.5521 ± 0.1301 (−0.20 vs 02)
- **Features:** 8 numeric + 3 native categorical (cohort, gene, rater_id) = 11 cols
- **Hyperparameters:** `HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, categorical_features=[...])`

---

### 05 — ridge_groupkfold

- **Status:** done
- **Splitter:** GroupKFold(n_splits=5, groups=patient_id)
- **CV RMSE:** 10.4470 ± 0.2090 — slightly worse than 03 despite honest CV; confirms Ridge is far below HistGBR
- **Features:** 8 numeric + OHE(cohort, gene, rater_id) = 61 cols

---

### 06 — lightgbm_cats

- **Status:** done
- **Splitter:** GroupKFold(n_splits=5, groups=patient_id)
- **CV RMSE:** 7.5439 ± 0.1335 — essentially identical to 04 HistGBR
- **Features:** 8 numeric + 3 native categorical = 11 cols

---

### 07 — lightgbm_trajectory

- **Status:** done
- **Splitter:** GroupKFold(n_splits=5, groups=patient_id)
- **CV RMSE:** 7.4793 ± 0.1312 (−0.06 vs 06) — modest gain from trajectory features
- **Features:** 12 numeric (+ visit_count, time_span, off_slope, on_slope) + 3 cats = 15 cols

---

### 08 — lightgbm_optuna

- **Status:** done
- **Splitter:** GroupKFold(n_splits=5, groups=patient_id)
- **CV RMSE:** 7.5369 — Optuna on 11-col set doesn't beat 06 default params
- **Best params:** lr=0.04886, num_leaves=50, max_depth=15, min_child_samples=40, reg_lambda=0.9118, reg_alpha=0.0897 (reused in EXP 09/10/11)

---

### 09 — lightgbm_domain_features

- **Status:** done
- **Splitter:** GroupKFold(n_splits=5, groups=patient_id) — rater_bias refitted per fold
- **Per-fold RMSE:** [6.7706, 6.8273, 6.6163, 6.6691, 6.7628]
- **CV RMSE:** 6.7292 ± 0.0759 (−0.75 vs 07 — biggest single jump in the project)
- **Features (18 cols):** 8 numeric + 7 domain (rater_bias, missingness flags, interactions) + 3 cats
- **Key insight:** rater_bias (fold-safe per-rater mean target) is the single strongest new signal

---

### 10 — stacking_temporal

- **Status:** done
- **Splitter:** manual GroupKFold(n_splits=5) — OOF stacking, meta-Ridge refitted per fold
- **CV RMSE:** 6.5791 ± 0.0859 (−0.15 vs 09)
- **Features:** EXP 09 18 cols + 4 temporal (patient_velocity_off, patient_accel_off, rater_consistency, visit_month) = 22 cols
- **Base learner CV RMSE:** LGB 6.5906 ± 0.0885 · HistGBR 6.6384 ± 0.0815 · Ridge 7.2625 ± 0.1120
- **Meta coefs:** LGB=0.7744, HGB=0.1423, Ridge=0.0984

---

### 11 — optuna_exp10_featureset

- **Status:** done
- **Splitter:** GroupKFold(n_splits=5, groups=patient_id) — rater_bias refitted per fold
- **Per-fold RMSE:** [6.6228, 6.6897, 6.4635, 6.4453, 6.5980]
- **CV RMSE:** 6.5695 ± 0.0945 (−0.010 vs 10 stacking; best single model at the time)
- **Features (22 cols):** identical to EXP 10 — 8 numeric + 7 domain + 4 temporal + 3 cats
- **Optuna:** 30 trials, MedianPruner; best params: lr=0.04886, num_leaves=50, max_depth=15, min_child_samples=40, reg_lambda=0.9118, reg_alpha=0.0897

---

### 12 — xgboost_gpu

- **Status:** done
- **Splitter:** GroupKFold(n_splits=5, groups=patient_id) — rater_bias refitted per fold
- **Per-fold RMSE:** [6.6396, 6.6845, 6.4837, 6.4839, 6.6155]
- **CV RMSE:** 6.5814 ± 0.0827 (+0.0023 vs EXP 11 — marginally worse)
- **Model:** `XGBRegressor(device='cuda', n_estimators=150, enable_categorical=True)`
- **Features (22 cols):** identical to EXP 10/11
- **Optuna best params:** lr=0.09837, max_depth=5, subsample=0.967, colsample_bytree=0.794, reg_lambda=1.481, reg_alpha=0.043
- **Verdict:** XGBoost GPU does not beat LightGBM on this dataset at 150 trees

---

### 13 — patient_context

- **Status:** done
- **Splitter:** GroupKFold(n_splits=5, groups=patient_id) — rater_bias refitted per fold
- **Model:** `LGBMCatRegressor` (EXP 11 best params + early stopping, n_estimators=500)
- **Features (122 cols):** 22 EXP11 cols + 100 patient-context cols (aggregates, lag/lead, rolling, interpolation, slopes, trajectory)
- **Test predictions:** fold-average of 5 models (more stable than single full-train refit)

**Ablation (same folds, early stopping):**

| Variant | Context cols added | OOF RMSE | vs A |
|---|---|---|---|
| A — EXP 11 base | none | 6.5645 ± 0.0945 | — |
| B — +agg+interp | +38 (patient mean/std/min/max/median + interpolated on/off) | 5.1147 ± 0.1000 | −1.45 |
| C — +lag/lead/roll | +88 (B + lag±1/±2 + rolling mean/std window=3) | 4.3602 ± 0.0839 | −2.20 |
| **D — full set** | **+100 (C + slopes + trajectory + ledd_per_year + off_missing_rate)** | **4.1991 ± 0.0628** | **−2.37** |

**Per-fold RMSE (variant D):** [4.2281, 4.2896, 4.1706, 4.0999, 4.2049]

**OOF diagnostics (variant D):**

| Condition | n | RMSE |
|---|---|---|
| `off` missing (42 %) | 18 913 | 4.8560 |
| `off` present (58 %) | 25 677 | 3.6401 |
| target < 20 | 6 802 | 2.1033 |
| target 20–50 | 27 141 | 3.8795 |
| target 50–70 | 9 705 | 5.4067 |
| target > 70 | 942 | 8.2397 |

**Top-10 feature importances (gain):**

| Feature | Gain |
|---|---|
| rater_id | 3928 |
| off_slope | 743 |
| age_at_diagnosis | 707 |
| on_slope | 597 |
| off_pat_mean | 568 |
| followup_years | 514 |
| total_followup_yrs | 501 |
| off_pat_std | 480 |
| off_pat_max | 475 |
| off_roll3_mean | 475 |

- **OOF std:** 15.916 vs target std 16.500 — predictions well-calibrated

---

## Backlog — next models to try

| ID | Item | Rationale | Priority |
|---|---|---|---|
| B01 | ~~HistGBR + categorical features~~ | Done as 04 — RMSE 7.5521 | ✅ Done |
| B02 | ~~GroupKFold for Ridge~~ | Done as 05 — RMSE 10.4470 | ✅ Done |
| B03 | ~~Feature engineering: visit trajectory~~ | Done as 07 — RMSE 7.4793 | ✅ Done |
| B04 | ~~LightGBM~~ | Done as 06/08 — RMSE 7.54/7.54 | ✅ Done |
| B05 | ~~Stacking~~ | Done as 10 — RMSE 6.5791 | ✅ Done |
| B07 | ~~Optuna + trajectory~~ | Absorbed into 09+10 | ✅ Done |
| B08 | ~~Optuna on EXP 10 feature set~~ | Done as 11 — RMSE 6.5695 | ✅ Done |
| B12 | ~~XGBoost GPU~~ | Done as 12 — RMSE 6.5814 (worse than LGB) | ✅ Done |
| B13 | ~~Patient-context features~~ | Done as 13 — RMSE **4.1991** (new best, −2.37) | ✅ Done |
| B14 | **Optuna on EXP 13 full feature set (122 cols)** | EXP 11 params were tuned on 22 cols; 100 new features likely warrant re-tuning. Expected gain: 0.2–0.5 RMSE. | 🔴 High |
| B15 | **CAUSAL=True variant of EXP 13** | Use only past visits (no lead/future agg) to check if the RMSE drop is genuine or due to future-visit leakage. | 🔴 High |
| B06 | **Target encoding for rater_id** | Replace rater_bias mean with leave-one-out encoder within fold — more nuanced rater correction. | 🟡 Medium |
| B11 | **Stacking v2 with EXP 13 features** | Build stacking ensemble on top of EXP 13 feature set. | 🟡 Medium |
| B09 | **Neural network (MLP) base learner** | Small MLP for non-linear rater×patient interactions. | 🟡 Medium |
| B10 | **Per-patient random intercept (mixed model)** | Linear mixed model with patient random effects. | 🟡 Medium |
