# Current findings on NYC flight delays

The published schedule supports an advance alert for departure delays of at least 15 minutes. Recent operating reports improve the measured one-hour F1. At and after scheduled departure, reported status accounts for much of the apparent gain. The scheduled-arrival replay addresses a separate target and has materially different accuracy for all labeled flights and for flights still awaiting a report.

All periods below are in 2013. **All** means flights with an observed label for the relevant target, including early flights. **AP** means average precision. Performance metrics describe each model, timing, threshold and population separately. Every row has its original table path and identifying fields in the [machine-readable results](current_results.json). Later studies used previously examined periods, including November–December in the replay, as exploratory historical comparisons.

## Original historical departure experiment

The original schedule-only model can run before departure without a fixed lead time. Logistic regression was selected by September–October F1 from the baseline, logistic regression, decision tree and random forest. November–December was the frozen assessment in that original protocol. Both rows cover all flights with an observed departure label. Accuracy comes from the saved confusion counts. AP comes from the audit’s recomputation using the original saved scores. [Original comparison](../outputs/tables/prediction_comparison.csv) · [Original selection](../outputs/models/prediction_selection.json) · [Audit metric recomputation](../outputs/experiments/independent_audit/stage1_audit_001/historical_recomputed.csv)

| Model | Fit | Cutoff selection | Assessment | Cutoff | n | Accuracy | Precision | Recall | F1 | ROC-AUC | AP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Original logistic regression | Jan–Aug | Sep–Oct | Sep–Oct | 0.325 | 55,775 | 0.6397 | 0.2330 | 0.6159 | 0.3381 | 0.6754 | 0.2463 |
| Original logistic regression | Jan–Aug | Sep–Oct | Nov–Dec | 0.325 | 54,145 | 0.6097 | 0.3053 | 0.6180 | 0.4087 | 0.6533 | 0.3213 |

## Schedule-only development comparisons

The improvement study’s internal champion was the interaction forest. The September–October row also uses that period to select its cutoff. The independent audit’s fold D compares compact logistic regression F and compact boosting G with thresholds frozen in August. The chronological grid ranks by the equal-weight mean F1 over folds A–C. The decision tree ranked first at **0.5041** and retained that selection status in fold D. Fixed compact boosting had higher measured fold D F1 within the frozen shortlist. These are pre-departure schedule features with no fixed lead time, except the start-of-day replay rows. All rows cover all labeled departures. Start-of-day replay results coincide with its unresolved rows because status resolution is disabled at that checkpoint. [Improvement validation metrics](../outputs/experiments/modeling_study/validation_metrics.csv) · [Audit fold metrics](../outputs/experiments/independent_audit/stage1_audit_001/fold_metrics.csv) · [Grid ranking](../outputs/experiments/chronological_grid_search/ranking.csv) · [Grid selection folds](../outputs/experiments/chronological_grid_search/selection_fold_metrics.csv) · [Grid fold D](../outputs/experiments/chronological_grid_search/september_october_metrics.csv) · [Replay metrics](../outputs/experiments/flight_delay_replay/historical_metrics.csv)

| Model | Fit | Cutoff selection | Assessment | Cutoff | n | Accuracy | Precision | Recall | F1 | ROC-AUC | AP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Interaction forest | Jan–Aug | Sep–Oct | Sep–Oct | 0.275 | 55,775 | 0.6542 | 0.2374 | 0.5941 | 0.3392 | 0.6794 | 0.2511 |
| Audit F · compact logistic | Jan–Jul | Aug | Sep–Oct | 0.225 | 55,775 | 0.5799 | 0.2215 | 0.7205 | 0.3389 | 0.6850 | 0.2564 |
| Audit G · compact boosting | Jan–Jul | Aug | Sep–Oct | 0.250 | 55,775 | 0.6093 | 0.2293 | 0.6839 | 0.3434 | 0.6915 | 0.2697 |
| Grid decision tree A | Jan–Mar | Apr | May | 0.225 | 28,233 | 0.6316 | 0.3556 | 0.6962 | 0.4708 | 0.6928 | 0.3897 |
| Grid decision tree B | Jan–Apr | May | Jun | 0.150 | 27,234 | 0.5984 | 0.4084 | 0.8118 | 0.5434 | 0.7016 | 0.4549 |
| Grid decision tree C | Jan–May | Jun | Jul–Aug | 0.150 | 57,326 | 0.5664 | 0.3597 | 0.8099 | 0.4982 | 0.6890 | 0.4028 |
| Grid decision tree D | Jan–Jul | Aug | Sep–Oct | 0.225 | 55,775 | 0.5423 | 0.2124 | 0.7617 | 0.3322 | 0.6768 | 0.2424 |
| Grid fixed compact boosting D | Jan–Jul | Aug | Sep–Oct | 0.250 | 55,775 | 0.6093 | 0.2293 | 0.6839 | 0.3434 | 0.6915 | 0.2697 |
| Schedule-only boosting at start of day | Jan–Jul | Aug | Sep–Oct | 0.250 | 55,775 | 0.6084 | 0.2290 | 0.6848 | 0.3432 | 0.6913 | 0.2696 |
| Schedule-only boosting at start of day | Jan–Jul | Aug | Nov–Dec | 0.250 | 54,145 | 0.6139 | 0.3104 | 0.6298 | 0.4159 | 0.6653 | 0.3352 |

## One-hour operational departure prediction

All rows are for **60 minutes before scheduled departure**, with **January–July fit** and **August F1 cutoff selection**. The operating features use prior-day history, scheduled traffic, recent reports, waiting counts and reported information about a matched earlier service where available. Operational boosting (`grid_16`) was selected in August. Operational logistic regression (`full_lr`) had the highest observed F1 in both historical periods among carried-forward candidates. It is shown as a historical comparison. This replay scores all labeled flights without resolving by status, so all and unresolved results coincide. [August selection](../outputs/experiments/flight_delay_replay/departure_f1_60_selection.csv) · [Replay metrics](../outputs/experiments/flight_delay_replay/historical_metrics.csv)

| Model | Assessment | Cutoff | n | Accuracy | Precision | Recall | F1 | ROC-AUC | AP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Schedule-only boosting | Sep–Oct | 0.250 | 55,775 | 0.6093 | 0.2293 | 0.6839 | 0.3434 | 0.6915 | 0.2697 |
| Schedule-only boosting | Nov–Dec | 0.250 | 54,145 | 0.6151 | 0.3112 | 0.6295 | 0.4165 | 0.6656 | 0.3351 |
| Operational boosting | Sep–Oct | 0.275 | 55,775 | 0.8530 | 0.5118 | 0.3504 | 0.4160 | 0.7427 | 0.4485 |
| Operational boosting | Nov–Dec | 0.275 | 54,145 | 0.7647 | 0.4658 | 0.5326 | 0.4970 | 0.7571 | 0.5124 |
| Operational logistic regression | Sep–Oct | 0.250 | 55,775 | 0.8466 | 0.4829 | 0.3757 | 0.4226 | 0.7400 | 0.4459 |
| Operational logistic regression | Nov–Dec | 0.250 | 54,145 | 0.7597 | 0.4585 | 0.5583 | 0.5035 | 0.7554 | 0.5097 |

## Status-assisted departure prediction

The timing replay fit models on **January–June** and selected cutoffs by **July–August accuracy**. At scheduled departure it selected an average of logistic regression and boosting using prior-service features. At +5 minutes it selected shallow boosting with waiting features, and at +10 minutes it selected prior-service logistic regression. Unresolved flights exclude those already reported departed less than 15 minutes late. The full timing series, including checkpoints 60, 15 and 5 minutes before departure, is in the [machine-readable results](current_results.json) and [timing chart](../outputs/experiments/flight_delay_replay/departure_timing_accuracy.png).

| Time / model | Assessment | Population | Cutoff | n | Accuracy | Precision | Recall | F1 | ROC-AUC | AP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Scheduled · selected model | Sep–Oct | All | 0.500 | 55,775 | 0.8948 | 0.6618 | 0.6060 | 0.6326 | 0.9484 | 0.7221 |
| Scheduled · selected model | Sep–Oct | Unresolved | 0.500 | 16,537 | 0.6453 | 0.6618 | 0.6060 | 0.6326 | 0.7018 | 0.7221 |
| Scheduled · selected model | Nov–Dec | All | 0.500 | 54,145 | 0.8543 | 0.6562 | 0.6978 | 0.6763 | 0.9269 | 0.7301 |
| Scheduled · selected model | Nov–Dec | Unresolved | 0.500 | 21,789 | 0.6378 | 0.6562 | 0.6978 | 0.6763 | 0.6899 | 0.7301 |
| +5 min · selected model | Sep–Oct | All | 0.405 | 55,775 | 0.9319 | 0.6870 | 0.9994 | 0.8143 | 0.9726 | 0.8154 |
| +5 min · selected model | Sep–Oct | Unresolved | 0.405 | 12,135 | 0.6869 | 0.6870 | 0.9994 | 0.8143 | 0.6575 | 0.8154 |
| +5 min · selected model | Nov–Dec | All | 0.405 | 54,145 | 0.9109 | 0.7101 | 0.9997 | 0.8303 | 0.9593 | 0.8185 |
| +5 min · selected model | Nov–Dec | Unresolved | 0.405 | 16,643 | 0.7100 | 0.7101 | 0.9997 | 0.8303 | 0.6430 | 0.8185 |
| +10 min · selected model | Sep–Oct | All | 0.590 | 55,775 | 0.9747 | 0.8553 | 1.0000 | 0.9220 | 0.9894 | 0.9166 |
| +10 min · selected model | Sep–Oct | Unresolved | 0.590 | 9,744 | 0.8553 | 0.8553 | 1.0000 | 0.9220 | 0.6420 | 0.9166 |
| +10 min · selected model | Nov–Dec | All | 0.590 | 54,145 | 0.9672 | 0.8694 | 0.9998 | 0.9301 | 0.9842 | 0.9175 |
| +10 min · selected model | Nov–Dec | Unresolved | 0.590 | 13,589 | 0.8693 | 0.8694 | 0.9998 | 0.9301 | 0.6237 | 0.9175 |
| +10 min · Waiting rule | Sep–Oct | All | 0.500 | 55,775 | 0.9747 | 0.8553 | 1.0000 | 0.9220 | 0.9851 | 0.8553 |
| +10 min · Waiting rule | Sep–Oct | Unresolved | 0.500 | 9,744 | 0.8553 | 0.8553 | 1.0000 | 0.9220 | 0.5000 | 0.8553 |
| +10 min · Waiting rule | Nov–Dec | All | 0.500 | 54,145 | 0.9672 | 0.8695 | 1.0000 | 0.9302 | 0.9790 | 0.8695 |
| +10 min · Waiting rule | Nov–Dec | Unresolved | 0.500 | 13,589 | 0.8695 | 0.8695 | 1.0000 | 0.9302 | 0.5000 | 0.8695 |

The roughly 97% all-flight accuracy at +10 minutes essentially matches the fixed waiting rule, which flags any flight still without a departure report. The selected model and rule differed on **0 of 55,775** September–October flights and **2 of 54,145** November–December flights. Only five minutes remain before the 15-minute delay definition at that checkpoint. The rule has no fit or cutoff-selection period. Its .500 cutoff applies to a binary status score. [Waiting-rule agreement](../outputs/experiments/flight_delay_replay/waiting_rule_agreement_10.csv) · [Replay metrics](../outputs/experiments/flight_delay_replay/historical_metrics.csv)

## Arrival prediction at scheduled arrival

The target is arrival delay of at least 15 minutes. At the destination-local scheduled arrival, the replay uses available departure information and recent arrival reports from other flights. Recent-arrivals boosting fit on **January–June** and **July–August accuracy** selected its cutoff. The unresolved subset contains flights without an arrival report by that checkpoint. In the all-flight calculation, a reported arrival is scored as resolved. [Arrival selection](../outputs/experiments/flight_delay_replay/arrival_accuracy_0_selection.csv) · [Replay metrics](../outputs/experiments/flight_delay_replay/historical_metrics.csv)

| Model | Assessment | Population | Cutoff | n | Accuracy | Precision | Recall | F1 | ROC-AUC | AP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Recent-arrivals boosting | Sep–Oct | All | 0.465 | 55,628 | 0.9343 | 0.8438 | 0.7139 | 0.7734 | 0.9755 | 0.8948 |
| Recent-arrivals boosting | Sep–Oct | Unresolved | 0.465 | 16,668 | 0.7809 | 0.8438 | 0.7139 | 0.7734 | 0.8553 | 0.8948 |
| Recent-arrivals boosting | Nov–Dec | All | 0.465 | 53,991 | 0.8978 | 0.8197 | 0.7745 | 0.7965 | 0.9638 | 0.9096 |
| Recent-arrivals boosting | Nov–Dec | Unresolved | 0.465 | 24,033 | 0.7703 | 0.8197 | 0.7745 | 0.7965 | 0.8564 | 0.9096 |

## Calibration and interpretation

Calibration was not adopted. The chronological intercept study found improvements in probability error measures in some later months, with inconsistent transfer across periods. Retain the raw scores with their probability-quality limitations. The historical study holds ranking and equivalent-threshold decisions fixed under the monotonic transformation. [Calibration report](chronological_calibration_study.md)

Event reports are assumed available at reconstructed event times. These are historical reconstructions tied to the 2013 sample and local schedule clocks. Weather was not incorporated. The replay report retains the hard-coded comparison benchmarks as supplied references with undocumented derivation. Fixed cutoff **0.265** belongs to the one-hour random forest and boosting ensemble, and **0.510** belongs to recent-arrivals boosting at scheduled arrival. Their original selection is undocumented. Both fixed-cutoff runs remain in the historical metrics under `ensemble_reference_0265` and `arrival_reference_0510`. [Replay report and assumptions](../outputs/experiments/flight_delay_replay/report.md) · [Replay reference definitions](../experiments/flight_delay_replay/run.py)
