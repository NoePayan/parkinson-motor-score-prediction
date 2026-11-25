# EDA: Parkinson's Disease Motor Score (true OFF)

---

## Dataset at a glance

| | Train | Test |
|---|---|---|
| Visits (rows) | 44 590 | 11 013 |
| Patients | 5 576 | 1 395 |
| Mean visits / patient | 8.0 | 7.9 |
| Max visits / patient | 12 | — |
| Patient overlap train/test | **0** | — |
| Columns | 14 (X + target) | 13 (X only) |

The holdout is **by patient** — zero patients appear in both splits. This is the
dominant constraint for cross-validation: folds must respect `patient_id`.

---

## Per-column findings

| Column | dtype | Null % (train) | Null % (test) | n\_unique (train) |
|---|---|---|---|---|
| `Index` | Int64 | 0 % | 0 % | 44 590 (row key) |
| `patient_id` | string | 0 % | 0 % | 5 576 |
| `cohort` | string | 0 % | 0 % | 2 |
| `sexM` | Int64 | 0 % | 0 % | 2 |
| `gene` | string | **32 %** | 32 % | 4 |
| `age_at_diagnosis` | Float64 | 5 % | 5 % | 563 |
| `age` | Float64 | 0 % | 0 % | 1 082 |
| `ledd` | Float64 | **37 %** | 39 % | 1 320 |
| `time_since_intake_on` | Float64 | **46 %** | 48 % | 64 |
| `time_since_intake_off` | Float64 | **79 %** | 79 % | 178 |
| `rater_id` | string | 0 % | 0 % | 50 |
| `on` | Float64 | **30 %** | 31 % | 83 |
| `off` | Float64 | **42 %** | 41 % | 101 |
| `target` | Float64 | 0 % | — | 867 |

Key points:
- `time_since_intake_off` is missing ~79 % of the time; OFF exams are rare.
- `off` is missing ~42 % of visits; `on` ~30 %. **No visit has both `on` and
  `off` missing simultaneously** (0 such rows in train) — every visit has at
  least one measured score.
- Missingness rates are **nearly identical in train and test** — the model can
  rely on the same pattern.
- `Index` is a unique row identifier: must be excluded from features.
- `patient_id` has cardinality ≈ 5 576 (12.5 % unique-ratio) — not an ID leak
  in the classical sense, but it encodes patient history; use it for grouping,
  not as a raw feature.

---

## Target distribution

| Statistic | Value |
|---|---|
| Mean | 37.5 |
| Std | 16.5 |
| Min | 0.0 |
| Q25 | 25.6 |
| Median | 37.3 |
| Q75 | 49.3 |
| Max | 109.5 |

The true OFF score is **roughly symmetric and unimodal** (bell-shaped
histogram, bin peak ≈ 33–44). It uses only a fraction of the theoretical 0–132
range, with the mass between 10 and 65. The tail beyond 75 is sparse (< 1 %).
RMSE is a natural metric; the std ≈ 16.5 sets the dummy-mean floor.

---

## Associations with target

| Feature | Pearson r | Cramér V |
|---|---|---|
| `off` | **0.871** | 0.41 |
| `on` | **0.669** | 0.37 |
| `ledd` | 0.298 | 0.24 |
| `age` | 0.310 | 0.11 |
| `time_since_intake_on` | ≈ 0 | 0.18 |
| `time_since_intake_off` | ≈ 0 | 0.09 |
| `age_at_diagnosis` | 0.133 | 0.05 |

Notable findings:

- **`off` → target** Pearson r = 0.87, Cramér V = 0.41. Very high — the
  measured OFF score is a near-proxy of the true OFF. **This is the biggest
  leakage risk**: feeding `off` directly may dominate the model and mask
  generalisation failures. It should be used deliberately, not blindly.
- **`on` → target** r = 0.67. Meaningful but weaker — the ON score captures
  the drug effect; its gap from `off` encodes wearing-off information.
- **`off` ↔ `on`** r = 0.87 — the two measured scores are also highly
  correlated with each other.
- **`age_at_diagnosis` ↔ `age`** r = 0.94 — near-collinear. Prefer
  `age - age_at_diagnosis` (years since diagnosis / disease duration) as a
  derived feature; it is the clinically meaningful quantity and avoids
  redundancy.
- `time_since_intake_on` and `time_since_intake_off` have near-zero Pearson
  but non-negligible Cramér V — non-linear or threshold effects expected from
  the pharmacokinetic curve described in CONTEXT.md.
- `rater_id` has low association (Cramér V ≈ 0.02) but is available in both
  splits — could encode scorer bias as a categorical feature.

---

## Structure signals

- **No datetime column** detected by skrub. Temporal ordering is implicit via
  `age` (visit age of the patient) or derived disease duration. There is no
  explicit `visit_date` column.
- **Group structure**: `patient_id` repeats across rows (mean 8 visits each).
  This is the grouping column for `GroupKFold`.
- **Cohort**: two distinct cohorts — may introduce distribution shift; include
  as a covariate.

---

## Modelling implications

1. **Cross-validator → `GroupKFold` on `patient_id`.**  
   The Kaggle holdout is by patient. Using random row splits would leak patient
   history across folds and produce over-optimistic CV scores. Use
   `GroupKFold(n_splits=5)` with `groups=patient_id`.

2. **`off` is the strongest predictor and a leakage risk.**  
   r = 0.87 means a model that just copies `off` would score well in-sample.
   `off` is missing 42 % of the time, so imputation becomes load-bearing for
   these rows. Consider the gap `off - on` as a feature (wearing-off magnitude).

3. **High missingness requires deliberate imputation.**  
   `time_since_intake_off` (79 %), `off` (42 %), `ledd` (37 %),
   `time_since_intake_on` (46 %) must all be imputed. Missingness itself is
   signal (a missing OFF exam means an OFF state was not assessed — not at
   random clinically).

4. **Derive disease duration from `age` and `age_at_diagnosis`.**  
   These are nearly collinear (r = 0.94). Replace both with
   `time_since_diagnosis = age - age_at_diagnosis` to avoid multicollinearity
   and inject the clinically relevant quantity.

5. **Drop `Index` from features.**  
   It is a unique row key with no predictive information.

6. **`patient_id` — group only, not a raw feature.**  
   Its unique ratio is 12.5 %; encoding it naively risks overfitting. Use it
   purely as the `groups` argument to the splitter.

7. **Metric → RMSE.**  
   The target is continuous and roughly symmetric. RMSE is standard; the
   dummy floor (predict mean) gives RMSE ≈ 16.5 (= std of target).

8. **`rater_id` — potential scorer-bias covariate.**  
   50 unique raters in both train and test. Low association with target, but
   it captures systematic scoring differences. Safe to include as a categorical.

---

## Open questions

- Does the dataset record visit order explicitly? `age` increases per patient
  over visits and can serve as a proxy visit timestamp.
- Is `ledd` missing because the patient is not yet on levodopa (early stage)
  or because the dose was not recorded? The clinical meaning differs.
- The two cohorts — do they differ in disease duration or score distributions?
  Worth checking in a follow-up EDA slice per cohort.
