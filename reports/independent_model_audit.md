# Independent model audit — Stage 1

Run `stage1_audit_001`. This report separates verification of historical saved test results, recomputation of earlier-study development results, and newly fitted exploratory development evidence. All months and full-year EDA have previously been seen. **No new November–December model predictions were made. No period is a newly untouched test set.** Production models, original findings, lock/selection files and the prior study remain unchanged.

## Provenance and exact commands

Interpreter: `/Users/animusoyan/PycharmProjects/PythonProject4/.venv/bin/python`. Python: `3.13.13 (v3.13.13:01104ce1beb, Apr  7 2026, 14:43:30) [Clang 16.0.0 (clang-1600.0.26.6)]`. Git commit: `44aeaf4d64ac7eda02152540c9b6c5561227120e`. Initial status (including pre-existing untracked work):

```text
?? experiments/
?? notebooks/modeling_improvement_study.ipynb
?? reports/modeling_improvement_study.md
?? tests/test_independent_audit.py
?? tests/test_modeling_study.py
```

| package | version |
| --- | --- |
| numpy | 2.5.5 |
| scipy | 1.18.1 |
| pandas | 2.3.1 |
| scikit-learn | 1.7.2 |
| matplotlib | 3.10.7 |
| joblib | 1.6.0 |
| threadpoolctl | 3.8.0 |
| nbclient | 0.11.0 |
| nbformat | 5.11.1 |
| ipykernel | 7.1.0 |
| notebook | 7.4.7 |

```text
/Users/animusoyan/PycharmProjects/PythonProject4/.venv/bin/python -B -m experiments.independent_audit.run start --run-id stage1_audit_001
/Users/animusoyan/PycharmProjects/PythonProject4/.venv/bin/python -B -m experiments.independent_audit.run historical --run-id stage1_audit_001
/Users/animusoyan/PycharmProjects/PythonProject4/.venv/bin/python -B -m experiments.independent_audit.run experiment --run-id stage1_audit_001
/Users/animusoyan/PycharmProjects/PythonProject4/.venv/bin/python -B -m experiments.independent_audit.run uncertainty --run-id stage1_audit_001
/Users/animusoyan/PycharmProjects/PythonProject4/.venv/bin/python -B -m experiments.independent_audit.run render --run-id stage1_audit_001
/Users/animusoyan/PycharmProjects/PythonProject4/.venv/bin/python -B -m experiments.independent_audit.run check --run-id stage1_audit_001
```

No packages were upgraded. NumPy, joblib, SciPy and threadpoolctl are directly imported by this audit but transitively supplied by the current requirements; nbformat/nbclient support notebook tooling. These should be explicitly declared in a future dependency update, with the verified versions above. Existing requirements are preserved. No applicable AGENTS.md was found in repository/ancestor inspection.

protocol.json was written before verification/fitting and contains exact candidate parameters, folds, threshold grids, seed 42, bootstrap design, versions, Git status and SHA-256 checksums of audit source and protected data/artifacts. New run creation refuses an existing directory. Permanent phase locks prevent accidental duplicate phases; a completed phase is skipped. No lock is deleted. All outputs are under outputs/experiments/independent_audit/stage1_audit_001/.

Preservation: **83 existing files byte-identical** before/after. The protected set includes cached raw/processed data, source, original reports/notebooks/tests, production artifacts and prior-study files, including untracked work. Manifests are in protocol.json and preservation_final.json (created by the final check command).

## 1. Independent reconstruction and data integrity

| check | result |
| --- | --- |
| raw_sha256_matches_manifest | True |
| raw_sha256 | b791a2590c9167aa0bd3cb45af3f64101180878cf088c183eae4011e3a654f42 |
| all_processed_comparisons | exact by flight_id including missing masks |
| rows | 336776 |
| classification_rows | 328521 |
| unknown_labels | 8255 |
| duplicate_source_content_rows | 0 |
| negative_departure_delays | 183575 |
| departure_delay_min | -43.000000 |
| departure_delay_max | 1301.000000 |
| departure_delay_over_1440 | 0 |
| missing_departure_clock_with_observed_delay | 0 |

The cached raw snapshot was checksum-verified against source.json. Target, dates, calendar features, schedules and split were independently reconstructed in memory without invoking prepare() or main.py. Exact column/key coverage and every value/missing mask match descriptive, classifier and missing-label exports by flight_id. No approved classifier input is missing; no new imputer was introduced. The original pipelines retain their already-fitted imputers.

| dep_delay | n | label_min | label_max |
| --- | --- | --- | --- |
| 14.000000 | 2256 | 0 | 0 |
| 15.000000 | 2140 | 1 | 1 |
| 16.000000 | 2085 | 1 | 1 |

| key | duplicate_rows | duplicate_groups |
| --- | --- | --- |
| date + origin + carrier + flight | 0 | 0 |
| date + origin + carrier + flight + dest + sched_dep_time | 0 | 0 |

| clock | comparable_n | inconsistent_modulo_1440 |
| --- | --- | --- |
| dep_time | 328521 | 0 |
| arr_time | 327346 | 0 |

The independent clock checker accepts 2400 as midnight and rejects malformed HHMM. Actual and scheduled departure clocks use origin local time, and arrival clocks use destination local time. Delay consistency is checked within each same-local-clock pair modulo 1440; arrival is never subtracted from departure to infer flight duration. Negative delays and observed extreme delays are retained. Modulo checks cannot identify an incorrect whole-day offset. Missing labels remain unknown, never zero or an assumed cancellation.

| month | n | known | missing_count | missing_share |
| --- | --- | --- | --- | --- |
| 1 | 27004 | 26483 | 521 | 0.019293 |
| 2 | 24951 | 23690 | 1261 | 0.050539 |
| 3 | 28834 | 27973 | 861 | 0.029861 |
| 4 | 28330 | 27662 | 668 | 0.023579 |
| 5 | 28796 | 28233 | 563 | 0.019551 |
| 6 | 28243 | 27234 | 1009 | 0.035726 |
| 7 | 29425 | 28485 | 940 | 0.031946 |
| 8 | 29327 | 28841 | 486 | 0.016572 |
| 9 | 27574 | 27122 | 452 | 0.016392 |
| 10 | 28889 | 28653 | 236 | 0.008169 |
| 11 | 27268 | 27035 | 233 | 0.008545 |
| 12 | 28135 | 27110 | 1025 | 0.036431 |

All reconstructed-column missing counts are in data_missingness.csv, including outcomes not eligible for prediction.

## 2. Historical saved-score verification — including old test results

{'new_Nov_Dec_inference': False, 'missing_artifacts': [], 'saved_metrics_verified': True, 'pipelines_checked': 4}

Saved row probabilities were aligned to exact classifier IDs/dates/splits/targets; stored predictions must equal probability >= their stored threshold. Confusion counts and metrics were independently recalculated. No model was called on November–December. Missing artifacts, if any, are listed above rather than fabricated or recreated.

| model | split | threshold | n | mean_probability | prevalence | accuracy | precision | recall | f1 | roc_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| decision_tree | test | 0.175000 | 54145 | 0.264730 | 0.218210 | 0.621904 | 0.303205 | 0.564452 | 0.394499 | 0.615798 |
| decision_tree | validation | 0.175000 | 55775 | 0.268646 | 0.149422 | 0.625442 | 0.221858 | 0.600912 | 0.324069 | 0.630173 |
| logistic_regression | test | 0.325000 | 54145 | 0.315081 | 0.218210 | 0.609733 | 0.305268 | 0.618028 | 0.408675 | 0.653322 |
| logistic_regression | validation | 0.325000 | 55775 | 0.292701 | 0.149422 | 0.639695 | 0.233022 | 0.615911 | 0.338120 | 0.675439 |
| majority_baseline | test | 0.500000 | 54145 | 0.000000 | 0.218210 | 0.781790 | undefined | 0.000000 | 0.000000 | 0.500000 |
| majority_baseline | validation | 0.500000 | 55775 | 0.000000 | 0.149422 | 0.850578 | undefined | 0.000000 | 0.000000 | 0.500000 |
| random_forest | test | 0.250000 | 54145 | 0.248664 | 0.218210 | 0.582455 | 0.295889 | 0.662124 | 0.409003 | 0.645101 |
| random_forest | validation | 0.250000 | 55775 | 0.249014 | 0.149422 | 0.577373 | 0.219188 | 0.713583 | 0.335363 | 0.675547 |

Historical monthly probability/calibration diagnostics (November–December rows below use saved probabilities only):

| model | month | prevalence | mean_probability | precision | recall | f1 | roc_auc | average_precision | tpr | fpr | f1_q20 | brier | log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| decision_tree | 9 | 0.144090 | 0.264578 | 0.211105 | 0.584698 | 0.310209 | 0.635657 | 0.199472 | 0.584698 | 0.367838 | 0.382649 | 0.178272 | 0.543258 |
| decision_tree | 10 | 0.154469 | 0.272496 | 0.231764 | 0.615228 | 0.336692 | 0.624664 | 0.199147 | 0.615228 | 0.372560 | 0.396223 | 0.191758 | 0.572001 |
| decision_tree | 11 | 0.150472 | 0.261878 | 0.212233 | 0.564651 | 0.308508 | 0.609526 | 0.195365 | 0.564651 | 0.371228 | 0.370316 | 0.183312 | 0.553938 |
| decision_tree | 12 | 0.285762 | 0.267575 | 0.391335 | 0.564347 | 0.462181 | 0.622632 | 0.360243 | 0.564347 | 0.351185 | 0.380149 | 0.234776 | 0.689360 |
| logistic_regression | 9 | 0.144090 | 0.283932 | 0.220379 | 0.574463 | 0.318553 | 0.663322 | 0.223430 | 0.574463 | 0.342121 | 0.390400 | 0.146392 | 0.458118 |
| logistic_regression | 10 | 0.154469 | 0.301001 | 0.243898 | 0.652508 | 0.355075 | 0.685994 | 0.266057 | 0.652508 | 0.369546 | 0.416846 | 0.151133 | 0.470235 |
| logistic_regression | 11 | 0.150472 | 0.309715 | 0.218467 | 0.629302 | 0.324338 | 0.657033 | 0.234228 | 0.629302 | 0.398746 | 0.390351 | 0.157721 | 0.485678 |
| logistic_regression | 12 | 0.285762 | 0.320431 | 0.388625 | 0.612108 | 0.475412 | 0.652283 | 0.400530 | 0.612108 | 0.385271 | 0.388247 | 0.198169 | 0.581267 |
| majority_baseline | 9 | 0.144090 | 0.000000 | undefined | 0.000000 | 0.000000 | 0.500000 | 0.144090 | 0.000000 | 0.000000 | 0.000000 | 0.144090 | 4.976681 |
| majority_baseline | 10 | 0.154469 | 0.000000 | undefined | 0.000000 | 0.000000 | 0.500000 | 0.154469 | 0.000000 | 0.000000 | 0.000000 | 0.154469 | 5.335170 |
| majority_baseline | 11 | 0.150472 | 0.000000 | undefined | 0.000000 | 0.000000 | 0.500000 | 0.150472 | 0.000000 | 0.000000 | 0.000000 | 0.150472 | 5.197105 |
| majority_baseline | 12 | 0.285762 | 0.000000 | undefined | 0.000000 | 0.000000 | 0.500000 | 0.285762 | 0.000000 | 0.000000 | 0.000000 | 0.285762 | 9.869860 |
| random_forest | 9 | 0.144090 | 0.248415 | 0.207445 | 0.697288 | 0.319761 | 0.661570 | 0.222182 | 0.697288 | 0.448479 | 0.399454 | 0.129942 | 0.431869 |
| random_forest | 10 | 0.154469 | 0.249580 | 0.230209 | 0.727971 | 0.349799 | 0.688012 | 0.273422 | 0.727971 | 0.444710 | 0.415175 | 0.133662 | 0.439767 |
| random_forest | 11 | 0.150472 | 0.248856 | 0.208112 | 0.674779 | 0.318113 | 0.645218 | 0.224092 | 0.674779 | 0.454783 | 0.386260 | 0.133660 | 0.440432 |
| random_forest | 12 | 0.285762 | 0.248472 | 0.383274 | 0.655480 | 0.483711 | 0.652387 | 0.401956 | 0.655480 | 0.421990 | 0.392099 | 0.197575 | 0.581070 |

## 3. Historical pipeline and preprocessing audit

Compatible saved pipelines were loaded and called only on January–October. Their September–October probabilities were compared by flight_id to saved probabilities. Scaler means/sample counts and category vocabularies were checked against January–August only; unseen-category probes must produce finite probabilities. Numeric roles are the original approved numeric list, including month/day_of_year/weekday and schedule clocks; carrier/flight/origin/dest are one-hot categories. Full feature names, dimensions and original n_iter_ are saved in historical_pipeline_checks.csv.

| model | max_abs_probability_difference | mean_abs_probability_difference | input_names | encoded_dimension | iterations |
| --- | --- | --- | --- | --- | --- |
| majority_baseline | 0.000000 | 0.000000 | ['month', 'day_of_week', 'day_of_year', 'is_weekend', 'sched_dep_minutes', 'sched_arr_minutes', 'carrier', 'flight', 'origin', 'dest', 'distance'] | 3388 | undefined |
| logistic_regression | 0.000000 | 0.000000 | ['month', 'day_of_week', 'day_of_year', 'is_weekend', 'sched_dep_minutes', 'sched_arr_minutes', 'carrier', 'flight', 'origin', 'dest', 'distance'] | 3388 | [19] |
| decision_tree | 0.000000 | 0.000000 | ['month', 'day_of_week', 'day_of_year', 'is_weekend', 'sched_dep_minutes', 'sched_arr_minutes', 'carrier', 'flight', 'origin', 'dest', 'distance'] | 3388 | undefined |
| random_forest | 0.000000 | 0.000000 | ['month', 'day_of_week', 'day_of_year', 'is_weekend', 'sched_dep_minutes', 'sched_arr_minutes', 'carrier', 'flight', 'origin', 'dest', 'distance'] | 3388 | undefined |

Fresh development-only original LR comparison (same C/solver/seed, iteration ceiling raised to 1000; no demand for bit-identical floats):

```json
{
  "warnings": [],
  "convergence_warnings": 0,
  "iterations": [
    19
  ],
  "encoded_dimension": 3388,
  "n_fit": 218601,
  "max_abs_probability_difference": 0.0,
  "mean_abs_probability_difference": 0.0,
  "auc_difference": 0.0
}
```

Matched train and development ranking: training scores below are in-sample and optimistic; they are not future performance estimates. A large train/development gap is diagnostic, not proof that one feature alone causes overfitting.

| model | period | n | prevalence | mean_probability | roc_auc | average_precision | f1 | brier | log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| majority_baseline | train_optimistic | 218601 | 0.241376 | 0.000000 | 0.500000 | 0.241376 | 0.000000 | 0.241376 | 8.336826 |
| majority_baseline | validation_reused | 55775 | 0.149422 | 0.000000 | 0.500000 | 0.149422 | 0.000000 | 0.149422 | 5.160846 |
| logistic_regression | train_optimistic | 218601 | 0.241376 | 0.241381 | 0.730846 | 0.452503 | 0.470731 | 0.160412 | 0.489763 |
| logistic_regression | validation_reused | 55775 | 0.149422 | 0.292701 | 0.675439 | 0.246266 | 0.338120 | 0.148828 | 0.464342 |
| decision_tree | train_optimistic | 218601 | 0.241376 | 0.241376 | 0.763754 | 0.538036 | 0.499766 | 0.149567 | 0.463439 |
| decision_tree | validation_reused | 55775 | 0.149422 | 0.268646 | 0.630173 | 0.199055 | 0.324069 | 0.185200 | 0.558024 |
| random_forest | train_optimistic | 218601 | 0.241376 | 0.241554 | 0.715886 | 0.436747 | 0.484723 | 0.171361 | 0.520921 |
| random_forest | validation_reused | 55775 | 0.149422 | 0.249014 | 0.675547 | 0.247750 | 0.335363 | 0.131853 | 0.435926 |

Monthly train/development probability means and all metrics are in historical_development_monthly.csv; original training uses in-sample predictions. Category sizes and per-month unknown counts/shares are in historical_category_coverage.csv. Original January–August controls (majority, always positive, constant training prior) are evaluated per development month in historical_development_controls.csv.

## 4. Temporal covariates and prevalence decomposition

| feature | distance_type | distance | train_min | train_max | validation_min | validation_max |
| --- | --- | --- | --- | --- | --- | --- |
| month | KS | 1.000000 | 1.000000 | 8.000000 | 9.000000 | 10.000000 |
| day_of_week | TV | 0.029150 | undefined | undefined | undefined | undefined |
| day_of_year | KS | 1.000000 | 1.000000 | 243.000000 | 244.000000 | 304.000000 |
| is_weekend | TV | 0.013929 | undefined | undefined | undefined | undefined |
| sched_dep_minutes | KS | 0.018015 | 300.000000 | 1439.000000 | 300.000000 | 1439.000000 |
| sched_arr_minutes | KS | 0.018186 | 1.000000 | 1439.000000 | 1.000000 | 1439.000000 |
| carrier | TV | 0.024582 | undefined | undefined | undefined | undefined |
| flight | TV | 0.463227 | undefined | undefined | undefined | undefined |
| origin | TV | 0.030198 | undefined | undefined | undefined | undefined |
| dest | TV | 0.054563 | undefined | undefined | undefined | undefined |
| distance | KS | 0.019971 | 80.000000 | 4983.000000 | 94.000000 | 4983.000000 |

TV is half the summed absolute difference of categorical proportions over their union; KS is the maximum numerical empirical-CDF gap. Month/day-of-year ranges are necessarily disjoint by design. A KS distance of 1 for calendar variables is not itself evidence of harmful concept drift or causality.

| quantity | value |
| --- | --- |
| strata | 153.000000 |
| common_strata | 147.000000 |
| train_support_coverage | 0.999973 |
| validation_support_coverage | 0.998870 |
| all_training_prevalence | 0.241376 |
| all_validation_prevalence | 0.149422 |
| common_training_prevalence | 0.241373 |
| expected_validation_mix_training_rates | 0.238741 |
| common_observed_validation_prevalence | 0.149429 |
| mix_component_common_support | -0.002633 |
| residual_within_strata | -0.089311 |
| common_total_change | -0.091944 |

Common strata are origin × carrier × four-hour scheduled-departure bands with observations in both training and validation; no outcome-based support cutoffs are selected. On that common support, training prevalence uses training weights; expected validation rate applies validation weights to training stratum rates; the residual is observed validation prevalence minus that expected rate. Coverage is reported separately so unsupported schedules are not silently included. The common-support change equals mix component plus residual, not necessarily the all-data change. Coarse observational strata do not establish pure label shift, concept drift mechanism, airline causality, or weather effects.

## 5. Prior modeling study — reproduced saved results, not a rerun

{'available': True, 'saved_validation_metrics_recomputed': True, 'internal_folds_available': True, 'prior_models_retrained': False, 'interpretation': 'Internal fold tables inspected as saved; only saved validation metrics independently recomputed'}

| candidate | f1 | roc_auc | average_precision | mean_probability | prevalence |
| --- | --- | --- | --- | --- | --- |
| decision_tree_original | 0.324069 | 0.630173 | 0.199055 | 0.268646 | 0.149422 |
| forest_interactions_d14 | 0.339214 | 0.679429 | 0.251076 | 0.238225 | 0.149422 |
| logistic_regression_original | 0.338120 | 0.675439 | 0.246266 | 0.292701 | 0.149422 |
| lr_cyclic_C1 | 0.343544 | 0.683829 | 0.255234 | 0.217284 | 0.149422 |
| lr_interactions_C1 | 0.342908 | 0.680481 | 0.250381 | 0.212085 | 0.149422 |
| lr_keyed_C1 | 0.336755 | 0.675184 | 0.243595 | 0.286218 | 0.149422 |
| random_forest_original | 0.335363 | 0.675547 | 0.247750 | 0.249014 | 0.149422 |

Previous internal fold tables were inspected as saved; their models were not refit. Earlier published validation metrics were recomputed from prior saved row probabilities aligned to development IDs. Reproduction of those arithmetic results does not validate prior feature/model selection as unbiased.

## 6. Exactly seven fixed candidates; no search

| candidate | change |
| --- | --- |
| A | Original LR; C=1, liblinear, max_iter=1000, random_state=42 |
| B | A minus month and day_of_year |
| C | A minus flight number |
| D | A with categorical weekday; weekend retained |
| E | B with departure spline, five uniform knots, degree 3, no bias, linear extrapolation |
| F | Compact LR: no month/day_of_year/flight; categorical carrier/origin/dest/weekday |
| G | Compact native-categorical HGB on exactly F inputs; fixed parameters |

F/G numerical predictors are is_weekend, sched_dep_minutes, sched_arr_minutes and distance. LR numerics and the E spline basis are scaled on fit data; categories are train-fitted one-hot with unknowns ignored. G uses train-fitted OrdinalEncoder with unknowns -> NaN, explicit categorical column indices [0,1,2,3], and raw numerics. Codes are not treated as ordered continuous values. HGB uses max_iter=100, max_leaf_nodes=15, min_samples_leaf=100, learning_rate=0.1, l2_regularization=10, early_stopping=False, random_state=42; no random validation is performed. Train cardinalities are checked against the installed max_bins=255 and recorded in convergence.json. There is no class weighting, SMOTE, weather, target encoding, tail number, rolling outcome variable or extra fitted algorithm.

| fold | fit | threshold_month | assessment_months |
| --- | --- | --- | --- |
| a | Jan–month 3 | 4 | [5] |
| b | Jan–month 4 | 5 | [6] |
| c | Jan–month 5 | 6 | [7, 8] |
| d | Jan–month 7 | 8 | [9, 10] |

All are reused exploratory development periods. Each full preprocessing pipeline is freshly fit on fit months only, with no shared estimator instances. Thresholds use only their designated later month. The primary threshold grid is 0.10–0.50 step 0.025; ties favor closest to 0.5 then higher. A separate 0–1 step .01 diagnostic grid includes always positive and 1.01 for all negative. Neither assessment labels nor a comparison of assessment grids chooses which result to keep: the primary grid remains primary. Both sets of fixed thresholds/curves are retained.

All following assessment metrics are calculated by reading development_predictions.csv, not by trusting metrics from a training loop. Row scores contain flight_id/date/fold/candidate/grid/probability/threshold/target. Tune-only probabilities and threshold curves are saved separately.

### Fold a — primary grid

| candidate | threshold | n | prevalence | mean_probability | accuracy | balanced_accuracy | precision | recall | f1 | roc_auc | average_precision | f1_q20 | brier | log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 0.250000 | 28233 | 0.235363 | 0.268677 | 0.607481 | 0.634267 | 0.336140 | 0.684876 | 0.450951 | 0.688579 | 0.388419 | 0.408851 | 0.168002 | 0.509584 |
| B | 0.200000 | 28233 | 0.235363 | 0.208888 | 0.627316 | 0.635101 | 0.345081 | 0.649812 | 0.450778 | 0.687708 | 0.387062 | 0.410203 | 0.167109 | 0.509067 |
| C | 0.225000 | 28233 | 0.235363 | 0.271081 | 0.561612 | 0.641148 | 0.323631 | 0.791422 | 0.459402 | 0.700836 | 0.404151 | 0.413499 | 0.165084 | 0.503578 |
| D | 0.225000 | 28233 | 0.235363 | 0.268005 | 0.574009 | 0.634463 | 0.324485 | 0.748683 | 0.452746 | 0.689200 | 0.388155 | 0.408256 | 0.167958 | 0.509333 |
| E | 0.200000 | 28233 | 0.235363 | 0.209003 | 0.624269 | 0.636599 | 0.344381 | 0.659895 | 0.452575 | 0.689402 | 0.386903 | 0.411575 | 0.166873 | 0.508071 |
| F | 0.175000 | 28233 | 0.235363 | 0.211812 | 0.575886 | 0.644232 | 0.329275 | 0.773363 | 0.461891 | 0.700397 | 0.401113 | 0.416575 | 0.164922 | 0.502157 |
| G | 0.200000 | 28233 | 0.235363 | 0.212435 | 0.620550 | 0.649115 | 0.348345 | 0.703085 | 0.465872 | 0.705653 | 0.410629 | 0.423226 | 0.163444 | 0.498090 |
| always_positive | 0.500000 | 28233 | 0.235363 | 1.000000 | 0.235363 | 0.500000 | 0.235363 | 1.000000 | 0.381042 | 0.500000 | 0.235363 | 0.333333 | 0.764637 | 26.410242 |
| majority | 0.500000 | 28233 | 0.235363 | 0.000000 | 0.764637 | 0.500000 | undefined | 0.000000 | 0.000000 | 0.500000 | 0.235363 | 0.000000 | 0.235363 | 8.129146 |
| training_prior | 0.500000 | 28233 | 0.235363 | 0.210734 | 0.764637 | 0.500000 | undefined | 0.000000 | 0.000000 | 0.500000 | 0.235363 | 0.000000 | 0.180574 | 0.547450 |

### Fold b — primary grid

| candidate | threshold | n | prevalence | mean_probability | accuracy | balanced_accuracy | precision | recall | f1 | roc_auc | average_precision | f1_q20 | brier | log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 0.225000 | 27234 | 0.294338 | 0.262254 | 0.605603 | 0.640488 | 0.405072 | 0.725299 | 0.519827 | 0.688785 | 0.449802 | 0.414150 | 0.191464 | 0.565065 |
| B | 0.200000 | 27234 | 0.294338 | 0.209231 | 0.635015 | 0.634787 | 0.420443 | 0.634232 | 0.505669 | 0.684884 | 0.443742 | 0.410126 | 0.198727 | 0.587091 |
| C | 0.250000 | 27234 | 0.294338 | 0.263910 | 0.633216 | 0.651218 | 0.424781 | 0.694985 | 0.527282 | 0.707317 | 0.466178 | 0.425695 | 0.187581 | 0.553921 |
| D | 0.225000 | 27234 | 0.294338 | 0.261840 | 0.606668 | 0.640806 | 0.405734 | 0.723802 | 0.519986 | 0.689188 | 0.450828 | 0.414482 | 0.191380 | 0.564839 |
| E | 0.200000 | 27234 | 0.294338 | 0.209136 | 0.631967 | 0.634881 | 0.418408 | 0.641966 | 0.506621 | 0.685953 | 0.442064 | 0.410100 | 0.198615 | 0.586284 |
| F | 0.225000 | 27234 | 0.294338 | 0.210377 | 0.662187 | 0.638696 | 0.443662 | 0.581587 | 0.503347 | 0.703182 | 0.460994 | 0.415662 | 0.195618 | 0.574708 |
| G | 0.225000 | 27234 | 0.294338 | 0.211331 | 0.660388 | 0.645675 | 0.444011 | 0.609905 | 0.513901 | 0.706147 | 0.466282 | 0.422939 | 0.194210 | 0.571492 |
| always_positive | 0.500000 | 27234 | 0.294338 | 1.000000 | 0.294338 | 0.500000 | 0.294338 | 1.000000 | 0.454809 | 0.500000 | 0.294338 | 0.333333 | 0.705662 | 24.373268 |
| majority | 0.500000 | 27234 | 0.294338 | 0.000000 | 0.705662 | 0.500000 | undefined | 0.000000 | 0.000000 | 0.500000 | 0.294338 | 0.000000 | 0.294338 | 10.166073 |
| training_prior | 0.500000 | 27234 | 0.294338 | 0.216175 | 0.705662 | 0.500000 | undefined | 0.000000 | 0.000000 | 0.500000 | 0.294338 | 0.000000 | 0.213813 | 0.622706 |

### Fold c — primary grid

| candidate | threshold | n | prevalence | mean_probability | accuracy | balanced_accuracy | precision | recall | f1 | roc_auc | average_precision | f1_q20 | brier | log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 0.200000 | 57326 | 0.265691 | 0.264408 | 0.547064 | 0.621052 | 0.344263 | 0.778938 | 0.477492 | 0.669346 | 0.393049 | 0.396782 | 0.184263 | 0.549127 |
| B | 0.150000 | 57326 | 0.265691 | 0.215917 | 0.539999 | 0.623993 | 0.343584 | 0.803230 | 0.481294 | 0.675309 | 0.400938 | 0.399199 | 0.184765 | 0.553233 |
| C | 0.200000 | 57326 | 0.265691 | 0.269763 | 0.539773 | 0.630543 | 0.346222 | 0.824240 | 0.487619 | 0.686309 | 0.406379 | 0.404351 | 0.180255 | 0.538434 |
| D | 0.175000 | 57326 | 0.265691 | 0.264004 | 0.510135 | 0.611766 | 0.331321 | 0.828639 | 0.473370 | 0.668138 | 0.392101 | 0.390033 | 0.184551 | 0.549866 |
| E | 0.150000 | 57326 | 0.265691 | 0.215101 | 0.539145 | 0.624857 | 0.343717 | 0.807760 | 0.482234 | 0.675989 | 0.399122 | 0.399886 | 0.184639 | 0.552601 |
| F | 0.175000 | 57326 | 0.265691 | 0.217679 | 0.578394 | 0.641463 | 0.362822 | 0.776049 | 0.494468 | 0.692442 | 0.415535 | 0.414054 | 0.181488 | 0.542129 |
| G | 0.175000 | 57326 | 0.265691 | 0.216848 | 0.581917 | 0.641851 | 0.364280 | 0.769746 | 0.494527 | 0.696508 | 0.422682 | 0.414519 | 0.180523 | 0.539732 |
| always_positive | 0.500000 | 57326 | 0.265691 | 1.000000 | 0.265691 | 0.500000 | 0.265691 | 1.000000 | 0.419835 | 0.500000 | 0.265691 | 0.333333 | 0.734309 | 25.362723 |
| majority | 0.500000 | 57326 | 0.265691 | 0.000000 | 0.734309 | 0.500000 | undefined | 0.000000 | 0.000000 | 0.500000 | 0.265691 | 0.000000 | 0.265691 | 9.176641 |
| training_prior | 0.500000 | 57326 | 0.265691 | 0.220216 | 0.734309 | 0.500000 | undefined | 0.000000 | 0.000000 | 0.500000 | 0.265691 | 0.000000 | 0.197167 | 0.584680 |

### Fold d — primary grid

| candidate | threshold | n | prevalence | mean_probability | accuracy | balanced_accuracy | precision | recall | f1 | roc_auc | average_precision | f1_q20 | brier | log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 0.300000 | 55775 | 0.149422 | 0.354884 | 0.517149 | 0.622543 | 0.204613 | 0.772858 | 0.323563 | 0.670348 | 0.242072 | 0.397976 | 0.175112 | 0.525196 |
| B | 0.225000 | 55775 | 0.149422 | 0.257174 | 0.561488 | 0.627440 | 0.213606 | 0.721502 | 0.329624 | 0.670067 | 0.240174 | 0.402176 | 0.139916 | 0.442411 |
| C | 0.325000 | 55775 | 0.149422 | 0.358538 | 0.540708 | 0.630062 | 0.211074 | 0.757499 | 0.330152 | 0.676435 | 0.247467 | 0.404323 | 0.171859 | 0.520466 |
| D | 0.300000 | 55775 | 0.149422 | 0.355189 | 0.519659 | 0.626393 | 0.206426 | 0.778618 | 0.326335 | 0.675433 | 0.248610 | 0.401148 | 0.174411 | 0.523623 |
| E | 0.225000 | 55775 | 0.149422 | 0.257376 | 0.555715 | 0.627657 | 0.212663 | 0.730262 | 0.329400 | 0.671496 | 0.241898 | 0.402342 | 0.139520 | 0.441396 |
| F | 0.225000 | 55775 | 0.149422 | 0.243331 | 0.579901 | 0.637869 | 0.221529 | 0.720542 | 0.338873 | 0.684959 | 0.256398 | 0.411766 | 0.131590 | 0.424707 |
| G | 0.250000 | 55775 | 0.149422 | 0.244553 | 0.609269 | 0.640048 | 0.229293 | 0.683945 | 0.343446 | 0.691480 | 0.269726 | 0.414595 | 0.131128 | 0.422513 |
| always_positive | 0.500000 | 55775 | 0.149422 | 1.000000 | 0.149422 | 0.500000 | 0.149422 | 1.000000 | 0.259995 | 0.500000 | 0.149422 | 0.333333 | 0.850578 | 29.378611 |
| majority | 0.500000 | 55775 | 0.149422 | 0.000000 | 0.850578 | 0.500000 | undefined | 0.000000 | 0.000000 | 0.500000 | 0.149422 | 0.000000 | 0.149422 | 5.160846 |
| training_prior | 0.500000 | 55775 | 0.149422 | 0.243323 | 0.850578 | 0.500000 | undefined | 0.000000 | 0.000000 | 0.500000 | 0.149422 | 0.000000 | 0.135912 | 0.448345 |

### Equal-weight summary of (a)–(c) only

| candidate | f1 | roc_auc | average_precision | accuracy | balanced_accuracy | precision | recall | brier | log_loss | f1_q20 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 0.482757 | 0.682237 | 0.410423 | 0.586716 | 0.631936 | 0.361825 | 0.729704 | 0.181243 | 0.541259 | 0.406595 |
| B | 0.479247 | 0.682634 | 0.410580 | 0.600777 | 0.631294 | 0.369703 | 0.695758 | 0.183534 | 0.549797 | 0.406509 |
| C | 0.491434 | 0.698154 | 0.425569 | 0.578200 | 0.640970 | 0.364878 | 0.770216 | 0.177640 | 0.531978 | 0.414515 |
| D | 0.482034 | 0.682175 | 0.410361 | 0.563604 | 0.629012 | 0.353847 | 0.767042 | 0.181296 | 0.541346 | 0.404257 |
| E | 0.480477 | 0.683781 | 0.409363 | 0.598460 | 0.632112 | 0.368835 | 0.703207 | 0.183375 | 0.548986 | 0.407187 |
| F | 0.486569 | 0.698673 | 0.425880 | 0.605489 | 0.641464 | 0.378586 | 0.710333 | 0.180676 | 0.539665 | 0.415430 |
| G | 0.491434 | 0.702769 | 0.433198 | 0.620952 | 0.645547 | 0.385545 | 0.694245 | 0.179392 | 0.536438 | 0.420228 |
| always_positive | 0.418562 | 0.500000 | 0.265131 | 0.265131 | 0.500000 | 0.265131 | 1.000000 | 0.734869 | 25.382078 | 0.333333 |
| majority | 0.000000 | 0.500000 | 0.265131 | 0.734869 | 0.500000 | undefined | 0.000000 | 0.265131 | 9.157286 | 0.000000 |
| training_prior | 0.000000 | 0.500000 | 0.265131 | 0.734869 | 0.500000 | undefined | 0.000000 | 0.197185 | 0.584945 | 0.000000 |

The first three folds receive equal weight despite the longer July–August assessment. The fourth fold is shown separately; assessments overlap earlier threshold/fit history and are not pooled as an independent test estimate.

### Matched ablation changes

| fold | candidate | reference | f1 | roc_auc | average_precision | brier | log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- |
| a | B | A | -0.000174 | -0.000871 | -0.001357 | -0.000893 | -0.000517 |
| a | C | A | 0.008450 | 0.012257 | 0.015732 | -0.002918 | -0.006005 |
| a | D | A | 0.001795 | 0.000621 | -0.000264 | -0.000044 | -0.000250 |
| a | E | B | 0.001797 | 0.001694 | -0.000159 | -0.000237 | -0.000995 |
| a | F | A | 0.010940 | 0.011818 | 0.012694 | -0.003080 | -0.007426 |
| a | G | F | 0.003981 | 0.005256 | 0.009517 | -0.001478 | -0.004067 |
| b | B | A | -0.014157 | -0.003901 | -0.006061 | 0.007263 | 0.022026 |
| b | C | A | 0.007456 | 0.018531 | 0.016376 | -0.003883 | -0.011144 |
| b | D | A | 0.000159 | 0.000402 | 0.001026 | -0.000084 | -0.000226 |
| b | E | B | 0.000951 | 0.001069 | -0.001677 | -0.000112 | -0.000807 |
| b | F | A | -0.016480 | 0.014396 | 0.011192 | 0.004154 | 0.009643 |
| b | G | F | 0.010554 | 0.002965 | 0.005288 | -0.001409 | -0.003216 |
| c | B | A | 0.003802 | 0.005962 | 0.007888 | 0.000501 | 0.004105 |
| c | C | A | 0.010127 | 0.016963 | 0.013329 | -0.004008 | -0.010693 |
| c | D | A | -0.004121 | -0.001209 | -0.000949 | 0.000288 | 0.000739 |
| c | E | B | 0.000941 | 0.000680 | -0.001816 | -0.000126 | -0.000632 |
| c | F | A | 0.016976 | 0.023095 | 0.022486 | -0.002776 | -0.006999 |
| c | G | F | 0.000059 | 0.004066 | 0.007147 | -0.000965 | -0.002397 |
| d | B | A | 0.006061 | -0.000282 | -0.001898 | -0.035196 | -0.082785 |
| d | C | A | 0.006589 | 0.006087 | 0.005395 | -0.003253 | -0.004730 |
| d | D | A | 0.002772 | 0.005085 | 0.006537 | -0.000701 | -0.001573 |
| d | E | B | -0.000224 | 0.001430 | 0.001724 | -0.000396 | -0.001015 |
| d | F | A | 0.015310 | 0.014610 | 0.014326 | -0.043522 | -0.100489 |
| d | G | F | 0.004573 | 0.006521 | 0.013328 | -0.000462 | -0.002194 |

Compare B/C/D to A, E to B, and G to F. F versus A is a feature bundle and cannot identify the effect of any single removal/encoding. Results that improve only one period are unstable rather than general improvement.

### Per-month assessment metrics

| fold | candidate | month | n | positive_count | prevalence | mean_probability | precision | recall | f1 | roc_auc | average_precision | tpr | fpr | f1_q20 | brier | log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a | A | 5 | 28233 | 6645 | 0.235363 | 0.268677 | 0.336140 | 0.684876 | 0.450951 | 0.688579 | 0.388419 | 0.684876 | 0.416342 | 0.408851 | 0.168002 | 0.509584 |
| a | B | 5 | 28233 | 6645 | 0.235363 | 0.208888 | 0.345081 | 0.649812 | 0.450778 | 0.687708 | 0.387062 | 0.649812 | 0.379609 | 0.410203 | 0.167109 | 0.509067 |
| a | C | 5 | 28233 | 6645 | 0.235363 | 0.271081 | 0.323631 | 0.791422 | 0.459402 | 0.700836 | 0.404151 | 0.791422 | 0.509125 | 0.413499 | 0.165084 | 0.503578 |
| a | D | 5 | 28233 | 6645 | 0.235363 | 0.268005 | 0.324485 | 0.748683 | 0.452746 | 0.689200 | 0.388155 | 0.748683 | 0.479757 | 0.408256 | 0.167958 | 0.509333 |
| a | E | 5 | 28233 | 6645 | 0.235363 | 0.209003 | 0.344381 | 0.659895 | 0.452575 | 0.689402 | 0.386903 | 0.659895 | 0.386696 | 0.411575 | 0.166873 | 0.508071 |
| a | F | 5 | 28233 | 6645 | 0.235363 | 0.211812 | 0.329275 | 0.773363 | 0.461891 | 0.700397 | 0.401113 | 0.773363 | 0.484899 | 0.416575 | 0.164922 | 0.502157 |
| a | G | 5 | 28233 | 6645 | 0.235363 | 0.212435 | 0.348345 | 0.703085 | 0.465872 | 0.705653 | 0.410629 | 0.703085 | 0.404855 | 0.423226 | 0.163444 | 0.498090 |
| a | always_positive | 5 | 28233 | 6645 | 0.235363 | 1.000000 | 0.235363 | 1.000000 | 0.381042 | 0.500000 | 0.235363 | 1.000000 | 1.000000 | 0.333333 | 0.764637 | 26.410242 |
| a | majority | 5 | 28233 | 6645 | 0.235363 | 0.000000 | undefined | 0.000000 | 0.000000 | 0.500000 | 0.235363 | 0.000000 | 0.000000 | 0.000000 | 0.235363 | 8.129146 |
| a | training_prior | 5 | 28233 | 6645 | 0.235363 | 0.210734 | undefined | 0.000000 | 0.000000 | 0.500000 | 0.235363 | 0.000000 | 0.000000 | 0.000000 | 0.180574 | 0.547450 |
| b | A | 6 | 27234 | 8016 | 0.294338 | 0.262254 | 0.405072 | 0.725299 | 0.519827 | 0.688785 | 0.449802 | 0.725299 | 0.444323 | 0.414150 | 0.191464 | 0.565065 |
| b | B | 6 | 27234 | 8016 | 0.294338 | 0.209231 | 0.420443 | 0.634232 | 0.505669 | 0.684884 | 0.443742 | 0.634232 | 0.364658 | 0.410126 | 0.198727 | 0.587091 |
| b | C | 6 | 27234 | 8016 | 0.294338 | 0.263910 | 0.424781 | 0.694985 | 0.527282 | 0.707317 | 0.466178 | 0.694985 | 0.392549 | 0.425695 | 0.187581 | 0.553921 |
| b | D | 6 | 27234 | 8016 | 0.294338 | 0.261840 | 0.405734 | 0.723802 | 0.519986 | 0.689188 | 0.450828 | 0.723802 | 0.442190 | 0.414482 | 0.191380 | 0.564839 |
| b | E | 6 | 27234 | 8016 | 0.294338 | 0.209136 | 0.418408 | 0.641966 | 0.506621 | 0.685953 | 0.442064 | 0.641966 | 0.372203 | 0.410100 | 0.198615 | 0.586284 |
| b | F | 6 | 27234 | 8016 | 0.294338 | 0.210377 | 0.443662 | 0.581587 | 0.503347 | 0.703182 | 0.460994 | 0.581587 | 0.304194 | 0.415662 | 0.195618 | 0.574708 |
| b | G | 6 | 27234 | 8016 | 0.294338 | 0.211331 | 0.444011 | 0.609905 | 0.513901 | 0.706147 | 0.466282 | 0.609905 | 0.318556 | 0.422939 | 0.194210 | 0.571492 |
| b | always_positive | 6 | 27234 | 8016 | 0.294338 | 1.000000 | 0.294338 | 1.000000 | 0.454809 | 0.500000 | 0.294338 | 1.000000 | 1.000000 | 0.333333 | 0.705662 | 24.373268 |
| b | majority | 6 | 27234 | 8016 | 0.294338 | 0.000000 | undefined | 0.000000 | 0.000000 | 0.500000 | 0.294338 | 0.000000 | 0.000000 | 0.000000 | 0.294338 | 10.166073 |
| b | training_prior | 6 | 27234 | 8016 | 0.294338 | 0.216175 | undefined | 0.000000 | 0.000000 | 0.500000 | 0.294338 | 0.000000 | 0.000000 | 0.000000 | 0.213813 | 0.622706 |
| c | A | 7 | 28485 | 8639 | 0.303282 | 0.256969 | 0.394749 | 0.762357 | 0.520160 | 0.676332 | 0.448669 | 0.762357 | 0.508818 | 0.401491 | 0.198006 | 0.581946 |
| c | A | 8 | 28841 | 6592 | 0.228564 | 0.271756 | 0.296884 | 0.800667 | 0.433156 | 0.669945 | 0.345746 | 0.800667 | 0.561823 | 0.395591 | 0.170690 | 0.516713 |
| c | B | 7 | 28485 | 8639 | 0.303282 | 0.214826 | 0.390660 | 0.799861 | 0.524936 | 0.680568 | 0.453263 | 0.799861 | 0.543082 | 0.402731 | 0.202751 | 0.597198 |
| c | B | 8 | 28841 | 6592 | 0.228564 | 0.216994 | 0.297115 | 0.807646 | 0.434417 | 0.672593 | 0.350778 | 0.807646 | 0.566093 | 0.396681 | 0.167000 | 0.509810 |
| c | C | 7 | 28485 | 8639 | 0.303282 | 0.262540 | 0.397256 | 0.807848 | 0.532606 | 0.692071 | 0.461839 | 0.807848 | 0.533558 | 0.409859 | 0.194061 | 0.570308 |
| c | C | 8 | 28841 | 6592 | 0.228564 | 0.276896 | 0.298256 | 0.845722 | 0.440990 | 0.689265 | 0.360361 | 0.845722 | 0.589555 | 0.402347 | 0.166620 | 0.506954 |
| c | D | 7 | 28485 | 8639 | 0.303282 | 0.256263 | 0.381600 | 0.815719 | 0.519959 | 0.675153 | 0.446773 | 0.815719 | 0.575431 | 0.396226 | 0.198443 | 0.583003 |
| c | D | 8 | 28841 | 6592 | 0.228564 | 0.271649 | 0.284011 | 0.845570 | 0.425204 | 0.669016 | 0.345869 | 0.845570 | 0.631579 | 0.386822 | 0.170830 | 0.517137 |
| c | E | 7 | 28485 | 8639 | 0.303282 | 0.213988 | 0.391133 | 0.803681 | 0.526184 | 0.679588 | 0.449003 | 0.803681 | 0.544593 | 0.403651 | 0.203128 | 0.597794 |
| c | E | 8 | 28841 | 6592 | 0.228564 | 0.216200 | 0.297068 | 0.813107 | 0.435153 | 0.675308 | 0.351589 | 0.813107 | 0.570048 | 0.397287 | 0.166377 | 0.507965 |
| c | F | 7 | 28485 | 8639 | 0.303282 | 0.216688 | 0.412344 | 0.770228 | 0.537133 | 0.696094 | 0.465780 | 0.770228 | 0.477829 | 0.418427 | 0.199860 | 0.585990 |
| c | F | 8 | 28841 | 6592 | 0.228564 | 0.218657 | 0.314214 | 0.783677 | 0.448574 | 0.692035 | 0.367753 | 0.783677 | 0.506764 | 0.411300 | 0.163342 | 0.498809 |
| c | G | 7 | 28485 | 8639 | 0.303282 | 0.216001 | 0.412281 | 0.760852 | 0.534782 | 0.698114 | 0.468325 | 0.760852 | 0.472135 | 0.416974 | 0.199315 | 0.584910 |
| c | G | 8 | 28841 | 6592 | 0.228564 | 0.217684 | 0.317160 | 0.781402 | 0.451189 | 0.698506 | 0.379762 | 0.781402 | 0.498449 | 0.413966 | 0.161963 | 0.495112 |
| c | always_positive | 7 | 28485 | 8639 | 0.303282 | 1.000000 | 0.303282 | 1.000000 | 0.465413 | 0.500000 | 0.303282 | 1.000000 | 1.000000 | 0.333333 | 0.696718 | 24.064329 |
| c | always_positive | 8 | 28841 | 6592 | 0.228564 | 1.000000 | 0.228564 | 1.000000 | 0.372083 | 0.500000 | 0.228564 | 1.000000 | 1.000000 | 0.333333 | 0.771436 | 26.645089 |
| c | majority | 7 | 28485 | 8639 | 0.303282 | 0.000000 | undefined | 0.000000 | 0.000000 | 0.500000 | 0.303282 | 0.000000 | 0.000000 | 0.000000 | 0.303282 | 10.475004 |
| c | majority | 8 | 28841 | 6592 | 0.228564 | 0.000000 | undefined | 0.000000 | 0.000000 | 0.500000 | 0.228564 | 0.000000 | 0.000000 | 0.000000 | 0.228564 | 7.894304 |
| c | training_prior | 7 | 28485 | 8639 | 0.303282 | 0.220216 | undefined | 0.000000 | 0.000000 | 0.500000 | 0.303282 | 0.000000 | 0.000000 | 0.000000 | 0.218202 | 0.632211 |
| c | training_prior | 8 | 28841 | 6592 | 0.228564 | 0.220216 | undefined | 0.000000 | 0.000000 | 0.500000 | 0.228564 | 0.000000 | 0.000000 | 0.000000 | 0.176392 | 0.537736 |
| d | A | 9 | 27122 | 3908 | 0.144090 | 0.341507 | 0.196593 | 0.738229 | 0.310499 | 0.656903 | 0.218116 | 0.738229 | 0.507883 | 0.391658 | 0.170622 | 0.514311 |
| d | A | 10 | 28653 | 4426 | 0.154469 | 0.367546 | 0.211616 | 0.803434 | 0.334998 | 0.682225 | 0.262535 | 0.803434 | 0.546828 | 0.402649 | 0.179363 | 0.535500 |
| d | B | 9 | 27122 | 3908 | 0.144090 | 0.255429 | 0.201516 | 0.700870 | 0.313029 | 0.657444 | 0.219156 | 0.700870 | 0.467520 | 0.392540 | 0.139576 | 0.441309 |
| d | B | 10 | 28653 | 4426 | 0.154469 | 0.258827 | 0.224894 | 0.739720 | 0.344922 | 0.681278 | 0.260425 | 0.739720 | 0.465761 | 0.410640 | 0.140238 | 0.443455 |
| d | C | 9 | 27122 | 3908 | 0.144090 | 0.344181 | 0.203181 | 0.712641 | 0.316208 | 0.661463 | 0.221454 | 0.712641 | 0.470492 | 0.396505 | 0.167023 | 0.509001 |
| d | C | 10 | 28653 | 4426 | 0.154469 | 0.372128 | 0.217751 | 0.797108 | 0.342059 | 0.690077 | 0.269186 | 0.797108 | 0.523135 | 0.409861 | 0.176436 | 0.531319 |
| d | D | 9 | 27122 | 3908 | 0.144090 | 0.342994 | 0.198068 | 0.745138 | 0.312950 | 0.660684 | 0.222011 | 0.745138 | 0.507883 | 0.394601 | 0.170796 | 0.514692 |
| d | D | 10 | 28653 | 4426 | 0.154469 | 0.366732 | 0.213769 | 0.808179 | 0.338107 | 0.688558 | 0.271888 | 0.808179 | 0.543031 | 0.406089 | 0.177832 | 0.532077 |
| d | E | 9 | 27122 | 3908 | 0.144090 | 0.255762 | 0.200695 | 0.709570 | 0.312891 | 0.658445 | 0.220353 | 0.709570 | 0.475747 | 0.392835 | 0.139262 | 0.440493 |
| d | E | 10 | 28653 | 4426 | 0.154469 | 0.258903 | 0.223836 | 0.748531 | 0.344620 | 0.683069 | 0.262408 | 0.748531 | 0.474182 | 0.410688 | 0.139764 | 0.442251 |
| d | F | 9 | 27122 | 3908 | 0.144090 | 0.242433 | 0.208854 | 0.701382 | 0.321865 | 0.668372 | 0.229390 | 0.701382 | 0.447273 | 0.401883 | 0.131650 | 0.424840 |
| d | F | 10 | 28653 | 4426 | 0.154469 | 0.244180 | 0.233426 | 0.737460 | 0.354609 | 0.699763 | 0.283118 | 0.737460 | 0.442440 | 0.420538 | 0.131533 | 0.424581 |
| d | G | 9 | 27122 | 3908 | 0.144090 | 0.244005 | 0.215193 | 0.663255 | 0.324955 | 0.673603 | 0.235594 | 0.663255 | 0.407211 | 0.402937 | 0.131630 | 0.423849 |
| d | G | 10 | 28653 | 4426 | 0.154469 | 0.245071 | 0.242547 | 0.702214 | 0.360557 | 0.707466 | 0.305724 | 0.702214 | 0.400627 | 0.424976 | 0.130654 | 0.421248 |
| d | always_positive | 9 | 27122 | 3908 | 0.144090 | 1.000000 | 0.144090 | 1.000000 | 0.251885 | 0.500000 | 0.144090 | 1.000000 | 1.000000 | 0.333333 | 0.855910 | 29.562780 |
| d | always_positive | 10 | 28653 | 4426 | 0.154469 | 1.000000 | 0.154469 | 1.000000 | 0.267602 | 0.500000 | 0.154469 | 1.000000 | 1.000000 | 0.333333 | 0.845531 | 29.204283 |
| d | majority | 9 | 27122 | 3908 | 0.144090 | 0.000000 | undefined | 0.000000 | 0.000000 | 0.500000 | 0.144090 | 0.000000 | 0.000000 | 0.000000 | 0.144090 | 4.976681 |
| d | majority | 10 | 28653 | 4426 | 0.154469 | 0.000000 | undefined | 0.000000 | 0.000000 | 0.500000 | 0.154469 | 0.000000 | 0.000000 | 0.000000 | 0.154469 | 5.335170 |
| d | training_prior | 9 | 27122 | 3908 | 0.144090 | 0.243323 | undefined | 0.000000 | 0.000000 | 0.500000 | 0.144090 | 0.000000 | 0.000000 | 0.000000 | 0.133175 | 0.442295 |
| d | training_prior | 10 | 28653 | 4426 | 0.154469 | 0.243323 | undefined | 0.000000 | 0.000000 | 0.500000 | 0.154469 | 0.000000 | 0.000000 | 0.000000 | 0.138503 | 0.454071 |

Full numeric diagnostics including accuracy/confusion counts are in fold_monthly_metrics.csv. Mean probability versus observed prevalence exposes over/underprediction; Brier/log loss and reliability bins measure related aspects of probability quality. They combine calibration and discrimination and do not alone prove calibration. No calibrator is fit in Stage 1.

AP is non-interpolated average precision, not trapezoidal PR area; constant-score AP equals positive prevalence, so compare AP to each period’s prevalence. ROC-AUC measures ranking at all thresholds. Precision/recall/F1 and accuracy depend on the selected operating threshold. An F1-selected low threshold can flag many on-time flights and lower accuracy versus majority prediction while improving delay recall; accuracy optimizes a different error trade-off. Undefined precision remains NaN/undefined.

Fixed-prevalence F1 is 2qTPR/[q(1+TPR)+(1-q)FPR], q=0.20. It descriptively standardizes prevalence while holding observed TPR/FPR fixed. It is not a causal adjustment, probability recalibration, forecast or evidence that rates transport to another population.

### Diagnostic wider-grid results (not a replacement operating point)

| fold | candidate | threshold | f1 | roc_auc | average_precision | mean_probability |
| --- | --- | --- | --- | --- | --- | --- |
| a | A | 0.250000 | 0.450951 | 0.688579 | 0.388419 | 0.268677 |
| a | B | 0.200000 | 0.450778 | 0.687708 | 0.387062 | 0.208888 |
| a | C | 0.230000 | 0.460560 | 0.700836 | 0.404151 | 0.271081 |
| a | D | 0.240000 | 0.453166 | 0.689200 | 0.388155 | 0.268005 |
| a | E | 0.210000 | 0.451176 | 0.689402 | 0.386903 | 0.209003 |
| a | F | 0.180000 | 0.462663 | 0.700397 | 0.401113 | 0.211812 |
| a | G | 0.190000 | 0.465593 | 0.705653 | 0.410629 | 0.212435 |
| b | A | 0.230000 | 0.518061 | 0.688785 | 0.449802 | 0.262254 |
| b | B | 0.190000 | 0.511096 | 0.684884 | 0.443742 | 0.209231 |
| b | C | 0.260000 | 0.523978 | 0.707317 | 0.466178 | 0.263910 |
| b | D | 0.230000 | 0.518903 | 0.689188 | 0.450828 | 0.261840 |
| b | E | 0.200000 | 0.506621 | 0.685953 | 0.442064 | 0.209136 |
| b | F | 0.220000 | 0.506746 | 0.703182 | 0.460994 | 0.210377 |
| b | G | 0.230000 | 0.511423 | 0.706147 | 0.466282 | 0.211331 |
| c | A | 0.200000 | 0.477492 | 0.669346 | 0.393049 | 0.264408 |
| c | B | 0.170000 | 0.481537 | 0.675309 | 0.400938 | 0.215917 |
| c | C | 0.200000 | 0.487619 | 0.686309 | 0.406379 | 0.269763 |
| c | D | 0.190000 | 0.475591 | 0.668138 | 0.392101 | 0.264004 |
| c | E | 0.160000 | 0.482519 | 0.675989 | 0.399122 | 0.215101 |
| c | F | 0.170000 | 0.493961 | 0.692442 | 0.415535 | 0.217679 |
| c | G | 0.170000 | 0.494681 | 0.696508 | 0.422682 | 0.216848 |
| d | A | 0.320000 | 0.327329 | 0.670348 | 0.242072 | 0.354884 |
| d | B | 0.230000 | 0.330477 | 0.670067 | 0.240174 | 0.257174 |
| d | C | 0.320000 | 0.328711 | 0.676435 | 0.247467 | 0.358538 |
| d | D | 0.290000 | 0.323057 | 0.675433 | 0.248610 | 0.355189 |
| d | E | 0.240000 | 0.332180 | 0.671496 | 0.241898 | 0.257376 |
| d | F | 0.230000 | 0.340042 | 0.684959 | 0.256398 | 0.243331 |
| d | G | 0.240000 | 0.340442 | 0.691480 | 0.269726 | 0.244553 |

### Convergence and dimensions

| fold | candidate | fit_n | dimension | iterations | convergence_warnings | warnings | category_counts |
| --- | --- | --- | --- | --- | --- | --- | --- |
| a | A | 78146 | 2479 | [16] | 0 | [] | n/a |
| a | B | 78146 | 2477 | [16] | 0 | [] | n/a |
| a | C | 78146 | 122 | [7] | 0 | [] | n/a |
| a | D | 78146 | 2485 | [16] | 0 | [] | n/a |
| a | E | 78146 | 2482 | [18] | 0 | [] | n/a |
| a | F | 78146 | 126 | [7] | 0 | [] | n/a |
| a | G | 78146 | 8 | 100 | 0 | [] | [16, 3, 96, 7] |
| b | A | 105808 | 2703 | [16] | 0 | [] | n/a |
| b | B | 105808 | 2701 | [19] | 0 | [] | n/a |
| b | C | 105808 | 123 | [7] | 0 | [] | n/a |
| b | D | 105808 | 2709 | [18] | 0 | [] | n/a |
| b | E | 105808 | 2706 | [18] | 0 | [] | n/a |
| b | F | 105808 | 127 | [8] | 0 | [] | n/a |
| b | G | 105808 | 8 | 100 | 0 | [] | [16, 3, 97, 7] |
| c | A | 134041 | 2877 | [20] | 0 | [] | n/a |
| c | B | 134041 | 2875 | [17] | 0 | [] | n/a |
| c | C | 134041 | 125 | [8] | 0 | [] | n/a |
| c | D | 134041 | 2883 | [19] | 0 | [] | n/a |
| c | E | 134041 | 2880 | [19] | 0 | [] | n/a |
| c | F | 134041 | 129 | [8] | 0 | [] | n/a |
| c | G | 134041 | 8 | 100 | 0 | [] | [16, 3, 99, 7] |
| d | A | 189760 | 3233 | [19] | 0 | [] | n/a |
| d | B | 189760 | 3231 | [18] | 0 | [] | n/a |
| d | C | 189760 | 127 | [8] | 0 | [] | n/a |
| d | D | 189760 | 3239 | [18] | 0 | [] | n/a |
| d | E | 189760 | 3236 | [19] | 0 | [] | n/a |
| d | F | 189760 | 131 | [8] | 0 | [] | n/a |
| d | G | 189760 | 8 | 100 | 0 | [] | [16, 3, 101, 7] |

## 7. Paired uncertainty with predictions and thresholds fixed

Historical comparisons resample whole September–October calendar days, using identical sampled days for each pair. These are existing saved scores, some with historically validation-tuned thresholds. No historical model is refit for this bootstrap.

| candidate | reference | metric | point_difference | ci_low | ci_high | scheme | replicates | seed | valid_replicates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| decision_tree | logistic_regression | f1 | -0.014051 | -0.028149 | -0.002825 | whole_day | 500 | 42 | 500 |
| random_forest | logistic_regression | f1 | -0.002757 | -0.009499 | 0.004221 | whole_day | 500 | 42 | 500 |
| lr_cyclic_C1 | logistic_regression | f1 | 0.005423 | -0.001529 | 0.012554 | whole_day | 500 | 42 | 500 |
| forest_interactions_d14 | logistic_regression | f1 | 0.001094 | -0.004988 | 0.006730 | whole_day | 500 | 42 | 500 |

Fold (d): 500 fixed-seed replicates use moving blocks of seven consecutive calendar days. Blocks are sampled with replacement from valid starts within the September–October calendar and concatenated/truncated to its original length. No artificial wrap joins October to September. All A/C/F/G pairs share each replicate’s day weights. Point differences and 95% percentile intervals:

| candidate | reference | metric | point_difference | ci_low | ci_high | scheme | replicates | seed | valid_replicates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C | A | f1 | 0.006589 | 0.004356 | 0.010242 | moving_7_day | 500 | 42 | 500 |
| C | A | roc_auc | 0.006087 | 0.003101 | 0.010130 | moving_7_day | 500 | 42 | 500 |
| C | A | average_precision | 0.005395 | 0.003281 | 0.008944 | moving_7_day | 500 | 42 | 500 |
| F | A | f1 | 0.015310 | 0.009076 | 0.020213 | moving_7_day | 500 | 42 | 500 |
| F | A | roc_auc | 0.014610 | 0.006699 | 0.016965 | moving_7_day | 500 | 42 | 500 |
| F | A | average_precision | 0.014326 | 0.003889 | 0.020120 | moving_7_day | 500 | 42 | 500 |
| G | A | f1 | 0.019883 | 0.012289 | 0.026115 | moving_7_day | 500 | 42 | 500 |
| G | A | roc_auc | 0.021131 | 0.013595 | 0.023650 | moving_7_day | 500 | 42 | 500 |
| G | A | average_precision | 0.027654 | 0.013216 | 0.038057 | moving_7_day | 500 | 42 | 500 |
| F | C | f1 | 0.008721 | 0.003473 | 0.011346 | moving_7_day | 500 | 42 | 500 |
| F | C | roc_auc | 0.008524 | 0.000373 | 0.009865 | moving_7_day | 500 | 42 | 500 |
| F | C | average_precision | 0.008931 | -0.001181 | 0.013038 | moving_7_day | 500 | 42 | 500 |
| G | C | f1 | 0.013294 | 0.006877 | 0.018126 | moving_7_day | 500 | 42 | 500 |
| G | C | roc_auc | 0.015045 | 0.007356 | 0.016869 | moving_7_day | 500 | 42 | 500 |
| G | C | average_precision | 0.022259 | 0.008377 | 0.030884 | moving_7_day | 500 | 42 | 500 |
| G | F | f1 | 0.004573 | 0.000983 | 0.007378 | moving_7_day | 500 | 42 | 500 |
| G | F | roc_auc | 0.006521 | 0.004204 | 0.008752 | moving_7_day | 500 | 42 | 500 |
| G | F | average_precision | 0.013328 | 0.007738 | 0.021388 | moving_7_day | 500 | 42 | 500 |

These intervals condition on the fitted scores/thresholds. They do not correct for previous selection, repeatedly examined validation, overlapping folds, nonstationarity, or all seasonal dependence. Whole-day resampling addresses within-day flight dependence; seven-day blocks retain some short-range dependence, not every temporal mechanism. Flight count alone is not a reason to call a small gain significant. Endpoint dates have fewer possible moving-block memberships; interpret intervals as limited diagnostics.

## 8. Independent findings and proposed next decision

Historical LR validation mean probability 0.292701, observed prevalence 0.149422, AUC 0.675439, F1 0.338120. This is the original Jan–Aug fit with Sep–Oct-tuned threshold 0.325, not fold (d).

| candidate | threshold | prevalence | mean_probability | f1 | roc_auc | average_precision | brier | log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 0.300000 | 0.149422 | 0.354884 | 0.323563 | 0.670348 | 0.242072 | 0.175112 | 0.525196 |
| C | 0.325000 | 0.149422 | 0.358538 | 0.330152 | 0.676435 | 0.247467 | 0.171859 | 0.520466 |
| F | 0.225000 | 0.149422 | 0.243331 | 0.338873 | 0.684959 | 0.256398 | 0.131590 | 0.424707 |
| G | 0.250000 | 0.149422 | 0.244553 | 0.343446 | 0.691480 | 0.269726 | 0.131128 | 0.422513 |

Independent numeric checks are comparisons, never optimization targets. Expected approximate fold (d) reference values and actual differences:

| candidate | actual_auc | reference_auc | auc_difference | actual_f1 | reference_f1 | f1_difference |
| --- | --- | --- | --- | --- | --- | --- |
| A | 0.670348 | 0.670350 | -0.000002 | 0.323563 | 0.323550 | 0.000013 |
| F | 0.684959 | 0.684960 | -0.000001 | 0.338873 | 0.338840 | 0.000033 |
| G | 0.691480 | 0.691480 | -0.000000 | 0.343446 | 0.343450 | -0.000004 |

Interpret changes across all four fold tables and paired intervals, not just aggregate September–October scores. Flight removal is isolated by C versus A; compact capacity by G versus F. If the compact candidates improve ranking/F1 but still overpredict average risk, feature simplification has not solved temporal calibration drift.

Proposed next decision: review the single-feature C ablation and the matched F/G results with their temporal stability and block intervals. Retain the historical production LR until an explicit integration decision. Any further calibration study needs a separately agreed chronological design; no calibration, extra search, old-test reassessment, dashboard, commit or push is performed here.

## 9. Tests and limitations

Focused tests cover independent mismatch detection, unknown-label preservation, threshold/boundary cases, Nov–Dec sentinel exclusion before conversion, strict fold chronology, LR and native-HGB unknown categories, isolated estimator instances, outcome/metadata isolation, weighted rank metrics versus explicit row replication, phase duplicate refusal and protected-file mutation detection. Existing tests run unchanged. Exact test command: `/Users/animusoyan/PycharmProjects/PythonProject4/.venv/bin/python -B -m unittest discover -s tests -v`.

Only one year of NYC departures is represented. Missing departure labels may be nonrandom. Calendar gaps are structural and the prevalence decomposition is coarse and observational. Original and earlier-study outputs were already reviewed. The dataset has no newly untouched future period for this team. No feature is approved for production merely because it wins one reused development comparison.

Execution note: Mac sleep interrupted code construction before any independent-audit run existed; process inspection confirmed no active experiment. The completed run starts from that partial code, not a duplicated experiment. The prior study is preserved rather than reconstructed. Notebook charts load saved audit tables only and never retrain or call a model.

## 10. Reviewed interpretation and execution handoff

The independent findings substantially reproduce locally. Target/export construction, exact saved confusion metrics, clock consistency and original LR mean validation probability match. The fresh Jan–Aug original LR is identical here (19 solver iterations, zero warnings); historical forest probability differences are at most 1.11e-16. Prior saved development metrics also recalculate correctly without retraining that study.

Removing flight number is the clearest single-change diagnostic: C improves F1, ROC-AUC and AP over A in every fold, and in both months of fold (d). Equal-weight (a)–(c) F1 increases 0.482757 → 0.491434 and AUC 0.682237 → 0.698154. It also reduces the optimistic training/development ranking gap, consistent with the high-cardinality flight predictor absorbing training-specific variation. This does not prove a unique overfitting mechanism or guarantee future improvement. C still overpredicts mean risk (0.358538 against 0.149422), so dropping flight alone does not resolve calibration drift.

Removing linear month/day_of_year (B) reduces fold (d) mean probability from 0.354884 to 0.257174 and improves F1, but its equal-weight earlier F1 declines; categorical weekday (D) has mixed earlier results. The spline (E versus B) has small earlier gains and slightly lower fold (d) F1. None supplies decisive evidence for integration on its own.

Compact LR F improves fold (d) F1 to 0.338873 and AUC to 0.684959. Its F1 is lower than A in June, so its bundled changes are not uniformly better. G versus matched F improves ROC-AUC/AP and F1 in all four folds, though the July–August F1 improvement is tiny. G has fold (d) F1 0.343446 and AUC 0.691480. Both compact models still overpredict mean risk (~0.243–0.245 versus 0.149422); neither is ready to be described as a calibrated numerical risk estimate.

Fold (d) matched in-sample and later assessment ranking (training scores remain optimistic):

| candidate | roc_auc_train_optimistic | average_precision_train_optimistic | roc_auc_assessment | average_precision_assessment |
| --- | --- | --- | --- | --- |
| A | 0.736558 | 0.466299 | 0.670348 | 0.242072 |
| B | 0.731088 | 0.458346 | 0.670067 | 0.240174 |
| C | 0.710792 | 0.425632 | 0.676435 | 0.247467 |
| D | 0.737178 | 0.468244 | 0.675433 | 0.248610 |
| E | 0.731603 | 0.460408 | 0.671496 | 0.241898 |
| F | 0.702778 | 0.409997 | 0.684959 | 0.256398 |
| G | 0.725330 | 0.447645 | 0.691480 | 0.269726 |

The review’s approximate fold (d) A/F/G AUC/F1 values are reproduced to within roughly 0.00004. We did not tune toward them. The iteration ceiling is 1000 as requested; all LR fits converge in 7–20 iterations, with zero convergence or other captured fit warnings. HGB completes the fixed 100 iterations; categorical counts are at most [16,3,101,7], within max_bins=255.

The prevalence decomposition has 99.997% training and 99.887% validation common-support coverage. Within that support, training prevalence is 24.137%, the expected rate under validation schedule weights is 23.874%, and observed validation prevalence is 14.943%. The mix component is −0.263 percentage points and the residual within-stratum change −8.931 points. This supports substantial temporal change not accounted for by these coarse schedule strata, but does not establish pure label shift, a causal explanation, or weather effects.

With fixed predictions and thresholds, seven-day block F1 difference intervals are C−A +0.006589 [0.004356,0.010242], F−A +0.015310 [0.009076,0.020213], G−A +0.019883 [0.012289,0.026115], and G−F +0.004573 [0.000983,0.007378]. AUC/AP intervals and every pair are in the full bootstrap table. These are conditional exploratory intervals, not selection-corrected evidence of a generalizable gain. Historical whole-day intervals include zero for prior cyclic-LR and enhanced-forest F1 gains over original LR, cautioning against their earlier point-score advantage.

The next decision is whether to retain C as the simplest candidate for a separately reviewed integration and F/G as matched research alternatives. Keep production unchanged now. Calibration or additional evaluation would need its own approved design; do not repeatedly reuse the old test months or describe a new untouched evaluation in this dataset.

### Exact execution environment and validation

Every start/historical/experiment/uncertainty/render phase used `LOKY_MAX_CPU_COUNT=2` in addition to the recorded interpreter. threadpoolctl limited native compute to two threads. The environment variable bounds CPU discovery without changing model hyperparameters or silently upgrading packages.

```text
LOKY_MAX_CPU_COUNT=2 /Users/animusoyan/PycharmProjects/PythonProject4/.venv/bin/python -B -m experiments.independent_audit.run start --run-id stage1_audit_001
LOKY_MAX_CPU_COUNT=2 /Users/animusoyan/PycharmProjects/PythonProject4/.venv/bin/python -B -m experiments.independent_audit.run historical --run-id stage1_audit_001
LOKY_MAX_CPU_COUNT=2 /Users/animusoyan/PycharmProjects/PythonProject4/.venv/bin/python -B -m experiments.independent_audit.run experiment --run-id stage1_audit_001
LOKY_MAX_CPU_COUNT=2 /Users/animusoyan/PycharmProjects/PythonProject4/.venv/bin/python -B -m experiments.independent_audit.run uncertainty --run-id stage1_audit_001
LOKY_MAX_CPU_COUNT=2 /Users/animusoyan/PycharmProjects/PythonProject4/.venv/bin/python -B -m experiments.independent_audit.run render --run-id stage1_audit_001
LOKY_MAX_CPU_COUNT=2 /Users/animusoyan/PycharmProjects/PythonProject4/.venv/bin/python -B -m experiments.independent_audit.run check --run-id stage1_audit_001
```

The completed test command was `LOKY_MAX_CPU_COUNT=2 /Users/animusoyan/PycharmProjects/PythonProject4/.venv/bin/python -B -m unittest discover -s tests -v`: 25 tests passed, no failures or skips. `python -B -m pip check` in that same interpreter found no broken requirements; it emitted only a non-writable pip-cache notice. The notebook was executed with that interpreter through nbclient, using localhost kernel socket permission; every cell reads saved tables only. The Jupyter parent-shutdown notice after completion is kernel cleanup, not a cell failure.

The notebook execution command was `/Users/animusoyan/PycharmProjects/PythonProject4/.venv/bin/python -B /private/tmp/nyc_execute_independent_audit.py`. Its executor is copied to this run as notebook_execution.py for provenance. The cells themselves contain no file writes, training or inference. This handoff was generated by `LOKY_MAX_CPU_COUNT=2 /Users/animusoyan/PycharmProjects/PythonProject4/.venv/bin/python -B outputs/experiments/independent_audit/stage1_audit_001/handoff_from_saved.py` with PYTHONPATH set to the repository root, using saved tables only.

No pre-existing study output was replaced and no duplicate run launched. Persistent phase locks and done markers remain in the run. No commit, push, production integration or dashboard action was taken.
