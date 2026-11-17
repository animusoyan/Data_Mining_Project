# Controlled modeling improvement study

This is an exploratory development study, not a replacement of the historical model. No November–December predictions or metrics were computed. Original models, selection metadata, tables, source and findings were verified unchanged by SHA-256 (see the preservation manifest).

## Encoding audit

- The original logistic regression treats month and day_of_year as linear numerical predictors. They are redundant seasonal proxies; with only January–August for fitting, later months require linear extrapolation. Trees cannot extrapolate a smooth seasonal trend beyond their training range. A single partial annual cycle cannot establish seasonality.
- Numerical weekday imposes a linear ordering. Weekday categories remove that assumption; is_weekend is retained despite redundancy.
- Linear local-clock minutes capture the strong later-departure association but have a discontinuity at midnight. Sine/cosine clock components are investigated alongside the raw ramps. Scheduled arrival clock is destination-local and is never subtracted from departure clock.
- One-hot flight number is preferable to a numeric magnitude, but numbers can be shared across carriers. Carrier-qualified flight IDs are investigated. Their higher cardinality increases variance and leaves more future categories unseen.
- Scaling for tree models is unnecessary but harmless; it is retained for comparability. Original preprocessing remains training-only and no target encodings are introduced.

Actual category coverage: original flight encoding has 3,260 training categories and 6.28% of validation flights have unseen numbers. Carrier-qualified IDs have 4,527 categories and 14.46% unseen validation IDs. This is a material variance/coverage trade-off; safe unknown handling does not supply an estimated effect for an unseen service.

| mode | feature | training_categories | validation_unseen_n | validation_unseen_share |
| --- | --- | --- | --- | --- |
| original | carrier | 16 | 0 | 0.0000 |
| original | flight | 3260 | 3503 | 0.0628 |
| original | origin | 3 | 0 | 0.0000 |
| original | dest | 102 | 51 | 0.0009 |
| keyed | carrier | 16 | 0 | 0.0000 |
| keyed | flight | 4527 | 8065 | 0.1446 |
| keyed | origin | 3 | 0 | 0.0000 |
| keyed | dest | 102 | 51 | 0.0009 |
| cyclic | carrier | 16 | 0 | 0.0000 |
| cyclic | flight | 3260 | 3503 | 0.0628 |
| cyclic | origin | 3 | 0 | 0.0000 |
| cyclic | dest | 102 | 51 | 0.0009 |
| cyclic | day_of_week | 7 | 0 | 0.0000 |
| interactions | carrier | 16 | 0 | 0.0000 |
| interactions | flight | 4527 | 8065 | 0.1446 |
| interactions | origin | 3 | 0 | 0.0000 |
| interactions | dest | 102 | 51 | 0.0009 |
| interactions | day_of_week | 7 | 0 | 0.0000 |
| interactions | carrier_origin | 35 | 0 | 0.0000 |
| interactions | origin_band | 15 | 0 | 0.0000 |
| interactions | carrier_band | 68 | 8 | 0.0001 |

## Predeclared chronology and search

| fold | fit | threshold | assessment |
| --- | --- | --- | --- |
| 1 | Jan-Mar | April | May |
| 2 | Jan-Apr | May | June |
| 3 | Jan-May | June | July-August |

Each fold uses earlier data to fit the full pipeline, a later disjoint month to select F1 threshold, and still-later months to score it. No shuffling is used for training or threshold selection. The three assessment scores are averaged with equal fold weight (the third covers two months). Folds share training history and are not independent replications. Their standard deviations are temporal variation, not confidence intervals.

Fourteen configurations, 42 chronological fits: original controls plus modest logistic C, tree depth/leaf-size, and forest depth/leaf-size/feature-sampling alternatives. Random states are fixed at 42; each forest has 100 trees and two workers. Threshold grid is 0.10–0.50 in steps of 0.025, with the original deterministic tie rule. C=0.1 and C=10 test stronger and weaker logistic regularization; bounded tree depth and leaf size control variance, while forest feature sampling changes tree diversity. The shortlist and internal champion were written before validation scoring. There was no expansion of the search after seeing validation results.

| candidate | mode | parameters |
| --- | --- | --- |
| logistic_regression_original | original | {'C': 1.0} |
| decision_tree_original | original | {'max_depth': 12, 'max_features': None, 'min_samples_leaf': 100} |
| random_forest_original | original | {'max_depth': 18, 'max_features': 'sqrt', 'min_samples_leaf': 30, 'n_estimators': 100} |
| lr_cyclic_C1 | cyclic | {'C': 1} |
| lr_keyed_C1 | keyed | {'C': 1} |
| lr_interactions_C1 | interactions | {'C': 1} |
| lr_interactions_C0.1 | interactions | {'C': 0.1} |
| lr_interactions_C10 | interactions | {'C': 10} |
| tree_interactions_d6 | interactions | {'max_depth': 6, 'max_features': None, 'min_samples_leaf': 200} |
| tree_interactions_d12 | interactions | {'max_depth': 12, 'max_features': None, 'min_samples_leaf': 100} |
| tree_interactions_d16 | interactions | {'max_depth': 16, 'max_features': None, 'min_samples_leaf': 50} |
| forest_interactions_d14 | interactions | {'max_depth': 14, 'max_features': 'sqrt', 'min_samples_leaf': 30, 'n_estimators': 100} |
| forest_interactions_d20 | interactions | {'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 10, 'n_estimators': 100} |
| forest_interactions_d18 | interactions | {'max_depth': 18, 'max_features': 0.1, 'min_samples_leaf': 30, 'n_estimators': 100} |

Encoding modes: original uses the historical features unchanged. keyed only changes flight to carrier+flight. cyclic replaces linear month/day_of_year with annual sine/cosine, treats weekday categorically, and adds sine/cosine departure and arrival clocks while retaining raw clocks. interactions extends cyclic with carrier-qualified flight plus carrier×origin, origin×4-hour departure band, and carrier×4-hour departure band categories. No route delay averages, outcomes, anomaly indicators or weather are added. These bundles do not identify the separate causal contribution of each engineering change.

## Internal assessment stability

| candidate | mean_f1 | min_f1 | mean_auc | mean_delta_f1 | better_f1_folds | better_auc_folds |
| --- | --- | --- | --- | --- | --- | --- |
| forest_interactions_d14 | 0.5029 | 0.4690 | 0.7089 | 0.0124 | 3 | 3 |
| forest_interactions_d20 | 0.5016 | 0.4680 | 0.7092 | 0.0111 | 3 | 3 |
| forest_interactions_d18 | 0.4998 | 0.4705 | 0.7043 | 0.0093 | 3 | 3 |
| random_forest_original | 0.4905 | 0.4666 | 0.6908 | 0.0000 | 0 | 0 |
| lr_keyed_C1 | 0.4835 | 0.4543 | 0.6868 | 0.0008 | 2 | 3 |
| logistic_regression_original | 0.4828 | 0.4510 | 0.6822 | 0.0000 | 0 | 0 |
| decision_tree_original | 0.4486 | 0.3937 | 0.6241 | 0.0000 | 0 | 0 |
| tree_interactions_d6 | 0.4440 | 0.3326 | 0.6479 | -0.0046 | 2 | 2 |
| tree_interactions_d16 | 0.4437 | 0.4106 | 0.6166 | -0.0049 | 1 | 1 |
| lr_interactions_C1 | 0.4419 | 0.3567 | 0.6827 | -0.0408 | 1 | 1 |
| lr_cyclic_C1 | 0.4394 | 0.3508 | 0.6790 | -0.0433 | 1 | 1 |
| tree_interactions_d12 | 0.4220 | 0.3301 | 0.5940 | -0.0266 | 0 | 1 |
| lr_interactions_C10 | 0.4159 | 0.3194 | 0.6637 | -0.0669 | 1 | 1 |
| lr_interactions_C0.1 | 0.4030 | 0.2663 | 0.6935 | -0.0798 | 1 | 3 |

The depth-14 enhanced forest wins internal mean F1 (0.5029 versus original forest 0.4905), with F1 changes +0.0025, +0.0208 and +0.0140 across the three periods. ROC-AUC also improves in all three periods (mean 0.7089 versus 0.6908). Depth 20 and the larger feature-sampling setting offer no F1 advantage over the depth-14 choice; retain the simpler depth-14 candidate for review.

The carrier-qualified logistic model has only a small mean F1 gain (+0.0008) and loses F1 in June, although ROC-AUC improves in all three folds. Cyclic logistic regression has lower internal mean F1 (0.4394 versus 0.4828) and loses in two periods. It is retained as a predeclared encoding diagnostic, not because it won internal selection. Interaction logistic regression is similarly unstable. C=0.1 improves internal ranking but has poor threshold-transfer F1; C=10 worsens ranking. Engineered decision trees do not beat the original tree on mean F1. Each candidate and every fold is available in internal_folds.csv.

## September–October comparison

Every final candidate is trained on January–August only; the original controls load their preserved fitted pipelines and original thresholds. The best internal candidate in each family plus the three C=1 logistic encoding ablations form the fixed shortlist. Learned new candidates tune their final thresholds on September–October F1. Thus aggregate and monthly F1 below are development scores after threshold selection, not unbiased future estimates. Ranking metrics do not depend on those thresholds, but candidate selection across validation still introduces optimism. No original test metrics were parsed or used for modeling; original test-containing files were only byte-hashed for preservation.

Threshold-dependent metrics:

| candidate | threshold | accuracy | balanced_accuracy | precision | recall | f1 |
| --- | --- | --- | --- | --- | --- | --- |
| logistic_regression_original | 0.3250 | 0.6397 | 0.6299 | 0.2330 | 0.6159 | 0.3381 |
| decision_tree_original | 0.1750 | 0.6254 | 0.6153 | 0.2219 | 0.6009 | 0.3241 |
| random_forest_original | 0.2500 | 0.5774 | 0.6335 | 0.2192 | 0.7136 | 0.3354 |
| lr_cyclic_C1 | 0.2500 | 0.6580 | 0.6336 | 0.2409 | 0.5989 | 0.3435 |
| lr_keyed_C1 | 0.3000 | 0.6143 | 0.6312 | 0.2266 | 0.6554 | 0.3368 |
| lr_interactions_C1 | 0.2250 | 0.6321 | 0.6364 | 0.2339 | 0.6424 | 0.3429 |
| forest_interactions_d14 | 0.2750 | 0.6542 | 0.6294 | 0.2374 | 0.5941 | 0.3392 |
| majority | 0.5000 | 0.8506 | 0.5000 | undefined | 0.0000 | 0.0000 |
| always_positive | 0.5000 | 0.1494 | 0.5000 | 0.1494 | 1.0000 | 0.2600 |

Threshold-independent ranking metrics and probability scores:

| candidate | n | positive_count | prevalence | roc_auc | average_precision | brier | log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression_original | 55775 | 8334 | 0.1494 | 0.6754 | 0.2463 | 0.1488 | 0.4643 |
| decision_tree_original | 55775 | 8334 | 0.1494 | 0.6302 | 0.1991 | 0.1852 | 0.5577 |
| random_forest_original | 55775 | 8334 | 0.1494 | 0.6755 | 0.2477 | 0.1319 | 0.4359 |
| lr_cyclic_C1 | 55775 | 8334 | 0.1494 | 0.6838 | 0.2552 | 0.1280 | 0.4137 |
| lr_keyed_C1 | 55775 | 8334 | 0.1494 | 0.6752 | 0.2436 | 0.1473 | 0.4600 |
| lr_interactions_C1 | 55775 | 8334 | 0.1494 | 0.6805 | 0.2504 | 0.1279 | 0.4134 |
| forest_interactions_d14 | 55775 | 8334 | 0.1494 | 0.6794 | 0.2511 | 0.1286 | 0.4236 |
| majority | 55775 | 8334 | 0.1494 | 0.5000 | 0.1494 | 0.1494 | 4.1287 |
| always_positive | 55775 | 8334 | 0.1494 | 0.5000 | 0.1494 | 0.8506 | 23.5024 |

Average precision (AP) is the non-interpolated precision–recall summary implemented by average_precision_score, often called PR-AUC; it is not trapezoidal PR-curve area. Its constant-score reference equals positive prevalence. Accuracy is dominated by on-time flights; balanced accuracy gives both actual classes equal weight. Undefined denominators or single-class ranking cases remain NaN/undefined. All probabilities, not hard labels, feed ranking metrics. Constant dummy probability scores are extreme and should not be mistaken for useful probability estimates.

Cyclic logistic regression improves validation F1 from 0.3381 to 0.3435 and ROC-AUC from 0.6754 to 0.6838; F1 improves in both September and October. This is a small, exploratory improvement that conflicts with its internal F1 instability. The enhanced forest improves the original forest from F1 0.3354 to 0.3392 and ROC-AUC 0.6755 to 0.6794, also in both validation months. It only narrowly exceeds original logistic F1 and does not beat cyclic logistic on this validation set. Carrier-qualified logistic regression does not improve aggregate validation F1 or ROC-AUC; interactions do not improve on the simpler cyclic encoding.

| candidate | month | n | positive_count | prevalence | f1 | roc_auc | average_precision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression_original | 9 | 27122 | 3908 | 0.1441 | 0.3186 | 0.6633 | 0.2234 |
| logistic_regression_original | 10 | 28653 | 4426 | 0.1545 | 0.3551 | 0.6860 | 0.2661 |
| decision_tree_original | 9 | 27122 | 3908 | 0.1441 | 0.3102 | 0.6357 | 0.1995 |
| decision_tree_original | 10 | 28653 | 4426 | 0.1545 | 0.3367 | 0.6247 | 0.1991 |
| random_forest_original | 9 | 27122 | 3908 | 0.1441 | 0.3198 | 0.6616 | 0.2222 |
| random_forest_original | 10 | 28653 | 4426 | 0.1545 | 0.3498 | 0.6880 | 0.2734 |
| lr_cyclic_C1 | 9 | 27122 | 3908 | 0.1441 | 0.3264 | 0.6722 | 0.2336 |
| lr_cyclic_C1 | 10 | 28653 | 4426 | 0.1545 | 0.3610 | 0.6961 | 0.2820 |
| lr_keyed_C1 | 9 | 27122 | 3908 | 0.1441 | 0.3197 | 0.6634 | 0.2239 |
| lr_keyed_C1 | 10 | 28653 | 4426 | 0.1545 | 0.3518 | 0.6856 | 0.2616 |
| lr_interactions_C1 | 9 | 27122 | 3908 | 0.1441 | 0.3259 | 0.6702 | 0.2332 |
| lr_interactions_C1 | 10 | 28653 | 4426 | 0.1545 | 0.3604 | 0.6921 | 0.2727 |
| forest_interactions_d14 | 9 | 27122 | 3908 | 0.1441 | 0.3216 | 0.6655 | 0.2246 |
| forest_interactions_d14 | 10 | 28653 | 4426 | 0.1545 | 0.3553 | 0.6918 | 0.2769 |
| majority | 9 | 27122 | 3908 | 0.1441 | 0.0000 | 0.5000 | 0.1441 |
| majority | 10 | 28653 | 4426 | 0.1545 | 0.0000 | 0.5000 | 0.1545 |
| always_positive | 9 | 27122 | 3908 | 0.1441 | 0.2519 | 0.5000 | 0.1441 |
| always_positive | 10 | 28653 | 4426 | 0.1545 | 0.2676 | 0.5000 | 0.1545 |

## Threshold transfer diagnostic

As a check independent of September–October threshold tuning, apply the median of each candidate’s three internal threshold choices to its full-training probabilities. This is a diagnostic rather than another optimization. Refitting on more months changes the score distribution, so a median of older thresholds is not guaranteed to transfer. No validation outcome determines these thresholds.

| candidate | internal_median_threshold | precision | recall | f1 | roc_auc |
| --- | --- | --- | --- | --- | --- |
| decision_tree_original | 0.1000 | 0.1734 | 0.8957 | 0.2906 | 0.6302 |
| forest_interactions_d14 | 0.2250 | 0.2126 | 0.7586 | 0.3321 | 0.6794 |
| logistic_regression_original | 0.2250 | 0.2021 | 0.7967 | 0.3224 | 0.6754 |
| lr_cyclic_C1 | 0.1500 | 0.2009 | 0.8143 | 0.3223 | 0.6838 |
| lr_interactions_C1 | 0.1500 | 0.2022 | 0.8063 | 0.3233 | 0.6805 |
| lr_keyed_C1 | 0.2250 | 0.2041 | 0.7907 | 0.3245 | 0.6752 |
| random_forest_original | 0.2000 | 0.1704 | 0.9351 | 0.2883 | 0.6755 |

Cyclic logistic F1 falls to 0.3223 at the internally derived threshold, essentially matching original logistic 0.3224. Enhanced forest reaches 0.3321 versus original forest 0.2883. These results caution against presenting the validation-retuned cyclic gain as a stable threshold improvement. The always-positive reference has F1 0.2600 and recall 1.0 but precision only 0.1494; it exposes what can be achieved by flagging every flight.

## Calibration and numerical dashboard risk

Calibration is evaluated on the internally selected depth-14 enhanced forest with a matched raw control: fit the base on January–June, fit a sigmoid (logistic regression on clipped probability logits) on July only, select each threshold on August only, and assess September–October. Sigmoid uses C=1e6 and max_iter=500. This deliberately sacrifices later base-training months to avoid fitting a calibrator on in-sample predictions. These models have a different fit scope from the January–August finalists and must not be directly credited with a calibration-only difference against them.

| method | assessment | threshold | precision | recall | f1 | roc_auc | average_precision | brier | log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw | Sep-Oct | 0.2250 | 0.2132 | 0.7571 | 0.3327 | 0.6801 | 0.2505 | 0.1276 | 0.4207 |
| raw | 9 | 0.2250 | 0.2026 | 0.7451 | 0.3186 | 0.6641 | 0.2220 | 0.1263 | 0.4182 |
| raw | 10 | 0.2250 | 0.2232 | 0.7677 | 0.3458 | 0.6945 | 0.2804 | 0.1288 | 0.4231 |
| sigmoid | Sep-Oct | 0.2500 | 0.2125 | 0.7598 | 0.3321 | 0.6801 | 0.2505 | 0.1421 | 0.4486 |
| sigmoid | 9 | 0.2500 | 0.2019 | 0.7474 | 0.3179 | 0.6641 | 0.2220 | 0.1428 | 0.4503 |
| sigmoid | 10 | 0.2500 | 0.2225 | 0.7707 | 0.3453 | 0.6945 | 0.2804 | 0.1414 | 0.4469 |

Sigmoid worsens Brier score from 0.1276 to 0.1421 and log loss from 0.4207 to 0.4486 on the matched aggregate assessment, and worsens both scores in each month. Ranking is unchanged by the fitted increasing sigmoid. July prevalence was 30.33%, compared with 14.41% in September and 15.45% in October; this observed drift is a plausible reason for poor transfer, not a proven causal explanation.

For context, constant probabilities estimated only from the corresponding base-training months have these proper scores:

| reference | training_prevalence | brier | log_loss |
| --- | --- | --- | --- |
| training_prior_Jan_to_6 | 0.2327 | 0.1340 | 0.4432 |
| training_prior_Jan_to_8 | 0.2414 | 0.1356 | 0.4474 |

Brier score and log loss assess overall probability quality, including discrimination as well as calibration; neither alone proves calibration. Reliability diagrams with counts in each bin accompany the study notebook. Original logistic Brier 0.1488 is worse than its January–August constant-prior reference 0.1356, despite useful ROC-AUC. Cyclic logistic improves Brier to 0.1280. Do not automatically calibrate or label any score as a reliable numerical risk estimate. Before dashboard percentages, retain temporal reliability checks and evaluate any new calibration scheme on a later disjoint development period. The tested July sigmoid is not recommended; no isotonic search or calibration retuning was added.

## Feature-group importance

Permutation importance is computed for the internal champion, fitted on January–August, on September–October only. Each raw feature group is jointly permuted using the same row order within the group; derived features are recomputed by its pipeline. Three fixed-seed repeats measure drops in ROC-AUC, AP and fixed-threshold F1. Permutation shuffles inputs solely for this diagnostic, never observations for fitting or temporal validation.

| group | auc_drop_mean | auc_drop_sd | ap_drop_mean | f1_drop_mean |
| --- | --- | --- | --- | --- |
| clocks | 0.1478 | 0.0025 | 0.0795 | 0.1105 |
| airline_flight | 0.0200 | 0.0003 | 0.0207 | 0.0155 |
| weekday | 0.0062 | 0.0005 | 0.0110 | 0.0067 |
| airports | 0.0050 | 0.0001 | 0.0054 | 0.0032 |
| distance | 0.0006 | 0.0001 | 0.0004 | 0.0003 |
| season | 0.0001 | 0.0000 | 0.0001 | 0.0003 |

Clocks dominate (mean ROC-AUC drop 0.1478), followed by airline/flight (0.0200), weekday (0.0062) and airports (0.0050). These are model-specific associations, not causal effects. Correlated inputs and implausible combinations after permutation can distort attribution; small season/distance importance is not sufficient grounds to remove those predictors. No feature was selected or deleted using this diagnostic.

## Tests and direct dependencies

The original tests cover exact split assignment, feature exclusion, training-only scaler/category fitting, unknown categories, undefined metrics, and saved selected-model inference. They are useful but do not exercise the end-to-end freeze-before-test orchestration, preservation of all historical files, every threshold tie, or the calibration/search chronology. The saved-artifact test can skip without artifacts and hardcodes the current training size.

New tests in tests/test_modeling_study.py cover rejecting test targets before conversion (including an invalid sentinel), strictly ordered fit/threshold/assessment periods, deterministic outcome-free cyclic/interacted features, midnight wrapping, carrier-qualified IDs, estimator cloning, unknown categories, and extended/dummy metrics. The study also saves a SHA-256 preservation manifest. Original workflow tests and files are unchanged. A study-only estimator reuse bug was fixed before importance completion; custom transformer artifacts were normalized to their canonical module so they can be loaded in fresh processes. No model search or validation retuning was repeated to repair diagnostics.

Direct dependencies should be explicit in a future integration: numpy and joblib are directly imported by source, while nbformat and nbclient support notebook authoring/execution, and IPython is directly imported by notebooks. scipy remains transitive unless directly used. Keep requirements.txt unchanged in this study; the verified environment snapshot is saved separately in outputs/experiments/modeling_study/environment.json. Pin a tested compatible set rather than silently upgrading dependencies during modeling.

| package | verified_version |
| --- | --- |
| pandas | 2.3.1 |
| matplotlib | 3.10.7 |
| ipykernel | 7.1.0 |
| notebook | 7.4.7 |
| scikit-learn | 1.7.2 |
| numpy | 2.5.5 |
| scipy | 1.18.1 |
| joblib | 1.6.0 |
| nbformat | 5.11.1 |
| nbclient | 0.11.0 |
| ipython | 9.18.1 |

## Recommendations for review

1. Keep the chronological three-part internal assessment, always-positive reference, expanded metrics, threshold-transfer diagnostics, and preservation checks as scientific improvements.
2. Keep the depth-14 enhanced forest as a promising research candidate: its gains are directionally consistent across all three internal periods and both validation months, but small on validation. Do not replace the historical selected model.
3. Keep cyclic logistic as a simple exploratory alternative because its validation ranking, F1 and probability scores improve; its weak internal F1 and threshold transfer mean performance improvement is not yet robust.
4. Do not keep additional interaction complexity, carrier-qualified IDs alone as a claimed performance improvement, deeper forests, the tried C changes, or engineered decision trees. None offers enough consistent evidence over simpler alternatives. Feature bundles remain a limitation; a future strictly bounded ablation could separate calendar, clock, weekday and identity effects.
5. Do not adopt the tested July sigmoid or advertise calibrated dashboard percentages. Retain probability-quality checks and the original model pending review.
6. Do not evaluate new candidates on November–December. Those outcomes were already reported historically, so the original final holdout is no longer pristine for this project team. This study provides development evidence only; a future claim of generalization needs genuinely new untouched data or a prospectively agreed evaluation, not repeated use of the old test months.

## Reproduction and artifacts

Run python -m experiments.modeling_study once using the verified PythonProject4/.venv interpreter. It refuses an existing study output directory. Run python -m experiments.study_report to render this report from saved tables. Open notebooks/modeling_improvement_study.ipynb to regenerate figures and inspect saved results without fitting or test inference. Outputs, experimental models, raw fold scores, thresholds, validation predictions, reliability bins, group importance, protocol, version snapshot and preservation manifest live only under outputs/experiments/modeling_study/. No dashboard or main-model integration is included.

Conceptual references: [scikit-learn time feature engineering](https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html) discusses periodic encodings. [scikit-learn probability calibration](https://scikit-learn.org/stable/modules/calibration.html) distinguishes reliability diagrams from overall proper scoring rules. The implementation uses the verified scikit-learn 1.7.2 environment, not the current documentation version.
