# Flight departure-delay prediction



Target: departure delay >=15 minutes. Only observed labels are evaluated. All four models use the existing scheduled/calendar feature allowlist.



## Protocol

Training: January-August. Validation: September-October. Test: November-December. No shuffling or refitting on validation. All preprocessing is fitted on training only. Flight number is categorical; numeric variables are median-imputed and standardized, and categorical variables are mode-imputed and one-hot encoded with unseen categories ignored.

The majority baseline always predicts the training majority class at threshold 0.5. Logistic regression provides an additive reference; a bounded decision tree captures interactions; the random forest averages 100 bounded trees. Hyperparameters were fixed before fitting; there was no hyperparameter search.

For each learned model, thresholds 0.10 through 0.50 in steps of 0.025 were compared using validation F1 only. Ties favor the threshold closest to 0.5, then higher. Model ties are broken alphabetically. All model thresholds and the winner were saved before any test prediction. The four frozen models were each assessed once on test for the required comparison; test results do not revise the winner.



## Class coverage

| split | n | positive_count | prevalence | negative_count |
| --- | --- | --- | --- | --- |
| test | 54145 | 11815 | 0.2182 | 42330 |
| train | 218601 | 52765 | 0.2414 | 165836 |
| validation | 55775 | 8334 | 0.1494 | 47441 |



## Model comparison

| model | split | threshold | precision | recall | f1 | roc_auc |
| --- | --- | --- | --- | --- | --- | --- |
| majority_baseline | validation | 0.5000 | undefined | 0.0000 | 0.0000 | 0.5000 |
| logistic_regression | validation | 0.3250 | 0.2330 | 0.6159 | 0.3381 | 0.6754 |
| decision_tree | validation | 0.1750 | 0.2219 | 0.6009 | 0.3241 | 0.6302 |
| random_forest | validation | 0.2500 | 0.2192 | 0.7136 | 0.3354 | 0.6755 |
| majority_baseline | test | 0.5000 | undefined | 0.0000 | 0.0000 | 0.5000 |
| logistic_regression | test | 0.3250 | 0.3053 | 0.6180 | 0.4087 | 0.6533 |
| decision_tree | test | 0.1750 | 0.3032 | 0.5645 | 0.3945 | 0.6158 |
| random_forest | test | 0.2500 | 0.2959 | 0.6621 | 0.4090 | 0.6451 |



Selected model: **logistic_regression**, threshold **0.325**, because it has the highest validation F1 under the declared protocol. The saved dashboard pipeline remains fitted exclusively on January-August.



Validation F1 is 0.3381; test F1 is 0.4087. The selected model trades many false alarms for delay recall; inspect the confusion counts below. Any higher test score from another model does not change the validation-frozen selection.



The selected model has test precision 30.53% and recall 61.80%: it detects 7,302 of 11,815 delayed flights, while producing 16,618 false alarms. The validation F1 lead over the runner-up is only 0.0028, so the comparison does not establish a decisive performance advantage.

Positive prevalence changes from 14.94% in validation to 21.82% in test. Test F1 is higher, while ROC-AUC changes from 0.6754 to 0.6533; higher F1 alone therefore does not imply stronger ranking or generalization. No significance test or probability calibration was performed.



## Monthly performance

| model | split | month | n | negative_count | positive_count | prevalence | precision | recall | f1 | roc_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| majority_baseline | validation | 9 | 27122 | 23214 | 3908 | 0.1441 | undefined | 0.0000 | 0.0000 | 0.5000 |
| majority_baseline | validation | 10 | 28653 | 24227 | 4426 | 0.1545 | undefined | 0.0000 | 0.0000 | 0.5000 |
| logistic_regression | validation | 9 | 27122 | 23214 | 3908 | 0.1441 | 0.2204 | 0.5745 | 0.3186 | 0.6633 |
| logistic_regression | validation | 10 | 28653 | 24227 | 4426 | 0.1545 | 0.2439 | 0.6525 | 0.3551 | 0.6860 |
| decision_tree | validation | 9 | 27122 | 23214 | 3908 | 0.1441 | 0.2111 | 0.5847 | 0.3102 | 0.6357 |
| decision_tree | validation | 10 | 28653 | 24227 | 4426 | 0.1545 | 0.2318 | 0.6152 | 0.3367 | 0.6247 |
| random_forest | validation | 9 | 27122 | 23214 | 3908 | 0.1441 | 0.2074 | 0.6973 | 0.3198 | 0.6616 |
| random_forest | validation | 10 | 28653 | 24227 | 4426 | 0.1545 | 0.2302 | 0.7280 | 0.3498 | 0.6880 |
| majority_baseline | test | 11 | 27035 | 22967 | 4068 | 0.1505 | undefined | 0.0000 | 0.0000 | 0.5000 |
| majority_baseline | test | 12 | 27110 | 19363 | 7747 | 0.2858 | undefined | 0.0000 | 0.0000 | 0.5000 |
| logistic_regression | test | 11 | 27035 | 22967 | 4068 | 0.1505 | 0.2185 | 0.6293 | 0.3243 | 0.6570 |
| logistic_regression | test | 12 | 27110 | 19363 | 7747 | 0.2858 | 0.3886 | 0.6121 | 0.4754 | 0.6523 |
| decision_tree | test | 11 | 27035 | 22967 | 4068 | 0.1505 | 0.2122 | 0.5647 | 0.3085 | 0.6095 |
| decision_tree | test | 12 | 27110 | 19363 | 7747 | 0.2858 | 0.3913 | 0.5643 | 0.4622 | 0.6226 |
| random_forest | test | 11 | 27035 | 22967 | 4068 | 0.1505 | 0.2081 | 0.6748 | 0.3181 | 0.6452 |
| random_forest | test | 12 | 27110 | 19363 | 7747 | 0.2858 | 0.3833 | 0.6555 | 0.4837 | 0.6524 |



## Confusion matrices

Rows are actual classes (0, 1); columns are predicted classes (0, 1). Each matrix is [[TN, FP], [FN, TP]].

| model | split | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- |
| majority_baseline | validation | 47441 | 0 | 8334 | 0 |
| logistic_regression | validation | 30546 | 16895 | 3201 | 5133 |
| decision_tree | validation | 29876 | 17565 | 3326 | 5008 |
| random_forest | validation | 26256 | 21185 | 2387 | 5947 |
| majority_baseline | test | 42330 | 0 | 11815 | 0 |
| logistic_regression | test | 25712 | 16618 | 4513 | 7302 |
| decision_tree | test | 27004 | 15326 | 5146 | 6669 |
| random_forest | test | 23714 | 18616 | 3992 | 7823 |



## Interpretation and limitations

Precision measures how many flagged flights were delayed; recall measures how many delayed flights were found. F1 balances these two at the chosen threshold. ROC-AUC uses positive-class probabilities rather than hard predictions and measures ranking, not probability calibration.

Undefined precision (no predicted positives), recall (no actual positives), F1 (zero denominator), or ROC-AUC (one observed class) is stored as an empty CSV value and shown as undefined here. Zero recall/F1 for the majority baseline is real; its precision is undefined. Confusion matrices and class counts make these cases explicit.

Compare validation and test metrics and the individual months rather than treating validation as a promise of future accuracy. Calendar shifts and changing flight mixes can affect generalization. Threshold selection on two validation months can overfit those months, and the grid is deliberately modest. Tree calendar features cannot extrapolate seasonal trends beyond their training range.

Only 2013 NYC departures are represented. Unknown delay labels are excluded, so results do not estimate cancellation risk or performance among missing outcomes. Schedule associations do not establish causal airline quality. Probabilities are not calibrated operational guarantees. No weather or aircraft history is included.

Full-year EDA was previously available for historical description; no full-year outcome aggregates, clusters, anomaly flags, or actual-time fields are used here. Feature choices and candidate specifications follow the approved scope, not final test results.



## Artifacts and reproducibility

outputs/models/flight_delay_pipeline.joblib stores the selected fitted pipeline. prediction_selection.json stores features, thresholds, parameters, versions, and source checksum. Per-model pipelines, validation threshold scores, comparison metrics, monthly metrics, confusion counts and row-level holdout predictions are also saved under outputs/. Generated artifacts are Git-ignored.

Run python -m src.predict once. An existing selection file blocks accidental repeat holdout evaluation; the prediction notebook loads saved artifacts and never retrains. Use the same scikit-learn environment when loading joblib files.
