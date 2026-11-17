"""Render the study report from saved development-only results; never train models."""
import json
import pandas as pd
from experiments.modeling_study import OUT, ROOT
from src.predict import markdown_table


def render():
    internal=pd.read_csv(OUT/'internal_folds.csv')
    summary=pd.read_csv(OUT/'internal_summary.csv')
    validation=pd.read_csv(OUT/'validation_metrics.csv')
    monthly=pd.read_csv(OUT/'validation_monthly.csv')
    calibration=pd.read_csv(OUT/'calibration_metrics.csv')
    transfer=pd.read_csv(OUT/'threshold_transfer.csv')
    references=pd.read_csv(OUT/'probability_references.csv')
    importance=pd.read_csv(OUT/'group_importance.csv').groupby('group').agg(
        auc_drop_mean=('auc_drop','mean'),auc_drop_sd=('auc_drop','std'),
        ap_drop_mean=('ap_drop','mean'),f1_drop_mean=('f1_drop','mean')).reset_index().sort_values('auc_drop_mean',ascending=False)
    protocol=json.loads((OUT/'protocol.json').read_text())
    comparison=[]
    for name,group in internal.groupby('candidate'):
        family=group.family.iloc[0]
        original=internal.loc[internal.candidate.eq(family+'_original')].set_index('fold')
        group=group.set_index('fold')
        delta=group.f1-original.f1
        comparison.append({'candidate':name,'mean_f1':group.f1.mean(),'min_f1':group.f1.min(),
            'mean_auc':group.roc_auc.mean(),'mean_delta_f1':delta.mean(),
            'better_f1_folds':int((delta>0).sum()),'better_auc_folds':int((group.roc_auc>original.roc_auc).sum())})
    comparison=pd.DataFrame(comparison).sort_values('mean_f1',ascending=False)
    comparison.to_csv(OUT/'stability_comparison.csv',index=False)
    config=pd.DataFrame([{'candidate':r['name'],'mode':r['encoding'],
        'parameters':str({k:v for k,v in r['parameters'].items() if k in ['C','max_depth','min_samples_leaf','max_features','n_estimators']})}
        for r in protocol['candidates']])
    monthly_small=monthly[['candidate','month','n','positive_count','prevalence','f1','roc_auc','average_precision']]
    dependencies=json.loads((OUT/'environment.json').read_text())
    parts=[
        '# Controlled modeling improvement study',
        'This is an exploratory development study, not a replacement of the historical model. No November–December predictions or metrics were computed. Original models, selection metadata, tables, source and findings were verified unchanged by SHA-256 (see the preservation manifest).',
        '## Encoding audit',
        '- The original logistic regression treats month and day_of_year as linear numerical predictors. They are redundant seasonal proxies; with only January–August for fitting, later months require linear extrapolation. Trees cannot extrapolate a smooth seasonal trend beyond their training range. A single partial annual cycle cannot establish seasonality.\n'
        '- Numerical weekday imposes a linear ordering. Weekday categories remove that assumption; is_weekend is retained despite redundancy.\n'
        '- Linear local-clock minutes capture the strong later-departure association but have a discontinuity at midnight. Sine/cosine clock components are investigated alongside the raw ramps. Scheduled arrival clock is destination-local and is never subtracted from departure clock.\n'
        '- One-hot flight number is preferable to a numeric magnitude, but numbers can be shared across carriers. Carrier-qualified flight IDs are investigated. Their higher cardinality increases variance and leaves more future categories unseen.\n'
        '- Scaling for tree models is unnecessary but harmless; it is retained for comparability. Original preprocessing remains training-only and no target encodings are introduced.',
        'Actual category coverage: original flight encoding has 3,260 training categories and 6.28% of validation flights have unseen numbers. Carrier-qualified IDs have 4,527 categories and 14.46% unseen validation IDs. This is a material variance/coverage trade-off; safe unknown handling does not supply an estimated effect for an unseen service.',
        markdown_table(pd.read_csv(OUT/'encoding_coverage.csv')),
        '## Predeclared chronology and search',
        markdown_table(pd.DataFrame([
            {'fold':1,'fit':'Jan-Mar','threshold':'April','assessment':'May'},
            {'fold':2,'fit':'Jan-Apr','threshold':'May','assessment':'June'},
            {'fold':3,'fit':'Jan-May','threshold':'June','assessment':'July-August'}])),
        'Each fold uses earlier data to fit the full pipeline, a later disjoint month to select F1 threshold, and still-later months to score it. No shuffling is used for training or threshold selection. The three assessment scores are averaged with equal fold weight (the third covers two months). Folds share training history and are not independent replications. Their standard deviations are temporal variation, not confidence intervals.',
        'Fourteen configurations, 42 chronological fits: original controls plus modest logistic C, tree depth/leaf-size, and forest depth/leaf-size/feature-sampling alternatives. Random states are fixed at 42; each forest has 100 trees and two workers. Threshold grid is 0.10–0.50 in steps of 0.025, with the original deterministic tie rule. C=0.1 and C=10 test stronger and weaker logistic regularization; bounded tree depth and leaf size control variance, while forest feature sampling changes tree diversity. The shortlist and internal champion were written before validation scoring. There was no expansion of the search after seeing validation results.',
        markdown_table(config),
        'Encoding modes: original uses the historical features unchanged. keyed only changes flight to carrier+flight. cyclic replaces linear month/day_of_year with annual sine/cosine, treats weekday categorically, and adds sine/cosine departure and arrival clocks while retaining raw clocks. interactions extends cyclic with carrier-qualified flight plus carrier×origin, origin×4-hour departure band, and carrier×4-hour departure band categories. No route delay averages, outcomes, anomaly indicators or weather are added. These bundles do not identify the separate causal contribution of each engineering change.',
        '## Internal assessment stability',
        markdown_table(comparison),
        'The depth-14 enhanced forest wins internal mean F1 (0.5029 versus original forest 0.4905), with F1 changes +0.0025, +0.0208 and +0.0140 across the three periods. ROC-AUC also improves in all three periods (mean 0.7089 versus 0.6908). Depth 20 and the larger feature-sampling setting offer no F1 advantage over the depth-14 choice; retain the simpler depth-14 candidate for review.',
        'The carrier-qualified logistic model has only a small mean F1 gain (+0.0008) and loses F1 in June, although ROC-AUC improves in all three folds. Cyclic logistic regression has lower internal mean F1 (0.4394 versus 0.4828) and loses in two periods. It is retained as a predeclared encoding diagnostic, not because it won internal selection. Interaction logistic regression is similarly unstable. C=0.1 improves internal ranking but has poor threshold-transfer F1; C=10 worsens ranking. Engineered decision trees do not beat the original tree on mean F1. Each candidate and every fold is available in internal_folds.csv.',
        '## September–October comparison',
        'Every final candidate is trained on January–August only; the original controls load their preserved fitted pipelines and original thresholds. The best internal candidate in each family plus the three C=1 logistic encoding ablations form the fixed shortlist. Learned new candidates tune their final thresholds on September–October F1. Thus aggregate and monthly F1 below are development scores after threshold selection, not unbiased future estimates. Ranking metrics do not depend on those thresholds, but candidate selection across validation still introduces optimism. No original test metrics were parsed or used for modeling; original test-containing files were only byte-hashed for preservation.',
        'Threshold-dependent metrics:',
        markdown_table(validation[['candidate','threshold','accuracy','balanced_accuracy','precision','recall','f1']]),
        'Threshold-independent ranking metrics and probability scores:',
        markdown_table(validation[['candidate','n','positive_count','prevalence','roc_auc','average_precision','brier','log_loss']]),
        'Average precision (AP) is the non-interpolated precision–recall summary implemented by average_precision_score, often called PR-AUC; it is not trapezoidal PR-curve area. Its constant-score reference equals positive prevalence. Accuracy is dominated by on-time flights; balanced accuracy gives both actual classes equal weight. Undefined denominators or single-class ranking cases remain NaN/undefined. All probabilities, not hard labels, feed ranking metrics. Constant dummy probability scores are extreme and should not be mistaken for useful probability estimates.',
        'Cyclic logistic regression improves validation F1 from 0.3381 to 0.3435 and ROC-AUC from 0.6754 to 0.6838; F1 improves in both September and October. This is a small, exploratory improvement that conflicts with its internal F1 instability. The enhanced forest improves the original forest from F1 0.3354 to 0.3392 and ROC-AUC 0.6755 to 0.6794, also in both validation months. It only narrowly exceeds original logistic F1 and does not beat cyclic logistic on this validation set. Carrier-qualified logistic regression does not improve aggregate validation F1 or ROC-AUC; interactions do not improve on the simpler cyclic encoding.',
        markdown_table(monthly_small),
        '## Threshold transfer diagnostic',
        'As a check independent of September–October threshold tuning, apply the median of each candidate’s three internal threshold choices to its full-training probabilities. This is a diagnostic rather than another optimization. Refitting on more months changes the score distribution, so a median of older thresholds is not guaranteed to transfer. No validation outcome determines these thresholds.',
        markdown_table(transfer[['candidate','internal_median_threshold','precision','recall','f1','roc_auc']]),
        'Cyclic logistic F1 falls to 0.3223 at the internally derived threshold, essentially matching original logistic 0.3224. Enhanced forest reaches 0.3321 versus original forest 0.2883. These results caution against presenting the validation-retuned cyclic gain as a stable threshold improvement. The always-positive reference has F1 0.2600 and recall 1.0 but precision only 0.1494; it exposes what can be achieved by flagging every flight.',
        '## Calibration and numerical dashboard risk',
        'Calibration is evaluated on the internally selected depth-14 enhanced forest with a matched raw control: fit the base on January–June, fit a sigmoid (logistic regression on clipped probability logits) on July only, select each threshold on August only, and assess September–October. Sigmoid uses C=1e6 and max_iter=500. This deliberately sacrifices later base-training months to avoid fitting a calibrator on in-sample predictions. These models have a different fit scope from the January–August finalists and must not be directly credited with a calibration-only difference against them.',
        markdown_table(calibration[['method','assessment','threshold','precision','recall','f1','roc_auc','average_precision','brier','log_loss']]),
        'Sigmoid worsens Brier score from 0.1276 to 0.1421 and log loss from 0.4207 to 0.4486 on the matched aggregate assessment, and worsens both scores in each month. Ranking is unchanged by the fitted increasing sigmoid. July prevalence was 30.33%, compared with 14.41% in September and 15.45% in October; this observed drift is a plausible reason for poor transfer, not a proven causal explanation.',
        'For context, constant probabilities estimated only from the corresponding base-training months have these proper scores:',
        markdown_table(references[['reference','training_prevalence','brier','log_loss']]),
        'Brier score and log loss assess overall probability quality, including discrimination as well as calibration; neither alone proves calibration. Reliability diagrams with counts in each bin accompany the study notebook. Original logistic Brier 0.1488 is worse than its January–August constant-prior reference 0.1356, despite useful ROC-AUC. Cyclic logistic improves Brier to 0.1280. Do not automatically calibrate or label any score as a reliable numerical risk estimate. Before dashboard percentages, retain temporal reliability checks and evaluate any new calibration scheme on a later disjoint development period. The tested July sigmoid is not recommended; no isotonic search or calibration retuning was added.',
        '## Feature-group importance',
        'Permutation importance is computed for the internal champion, fitted on January–August, on September–October only. Each raw feature group is jointly permuted using the same row order within the group; derived features are recomputed by its pipeline. Three fixed-seed repeats measure drops in ROC-AUC, AP and fixed-threshold F1. Permutation shuffles inputs solely for this diagnostic, never observations for fitting or temporal validation.',
        markdown_table(importance),
        'Clocks dominate (mean ROC-AUC drop 0.1478), followed by airline/flight (0.0200), weekday (0.0062) and airports (0.0050). These are model-specific associations, not causal effects. Correlated inputs and implausible combinations after permutation can distort attribution; small season/distance importance is not sufficient grounds to remove those predictors. No feature was selected or deleted using this diagnostic.',
        '## Tests and direct dependencies',
        'The original tests cover exact split assignment, feature exclusion, training-only scaler/category fitting, unknown categories, undefined metrics, and saved selected-model inference. They are useful but do not exercise the end-to-end freeze-before-test orchestration, preservation of all historical files, every threshold tie, or the calibration/search chronology. The saved-artifact test can skip without artifacts and hardcodes the current training size.',
        'New tests in tests/test_modeling_study.py cover rejecting test targets before conversion (including an invalid sentinel), strictly ordered fit/threshold/assessment periods, deterministic outcome-free cyclic/interacted features, midnight wrapping, carrier-qualified IDs, estimator cloning, unknown categories, and extended/dummy metrics. The study also saves a SHA-256 preservation manifest. Original workflow tests and files are unchanged. A study-only estimator reuse bug was fixed before importance completion; custom transformer artifacts were normalized to their canonical module so they can be loaded in fresh processes. No model search or validation retuning was repeated to repair diagnostics.',
        'Direct dependencies should be explicit in a future integration: numpy and joblib are directly imported by source, while nbformat and nbclient support notebook authoring/execution, and IPython is directly imported by notebooks. scipy remains transitive unless directly used. Keep requirements.txt unchanged in this study; the verified environment snapshot is saved separately in outputs/experiments/modeling_study/environment.json. Pin a tested compatible set rather than silently upgrading dependencies during modeling.',
        markdown_table(pd.DataFrame([{'package':k,'verified_version':v} for k,v in dependencies.items()])),
        '## Recommendations for review',
        '1. Keep the chronological three-part internal assessment, always-positive reference, expanded metrics, threshold-transfer diagnostics, and preservation checks as scientific improvements.\n'
        '2. Keep the depth-14 enhanced forest as a promising research candidate: its gains are directionally consistent across all three internal periods and both validation months, but small on validation. Do not replace the historical selected model.\n'
        '3. Keep cyclic logistic as a simple exploratory alternative because its validation ranking, F1 and probability scores improve; its weak internal F1 and threshold transfer mean performance improvement is not yet robust.\n'
        '4. Do not keep additional interaction complexity, carrier-qualified IDs alone as a claimed performance improvement, deeper forests, the tried C changes, or engineered decision trees. None offers enough consistent evidence over simpler alternatives. Feature bundles remain a limitation; a future strictly bounded ablation could separate calendar, clock, weekday and identity effects.\n'
        '5. Do not adopt the tested July sigmoid or advertise calibrated dashboard percentages. Retain probability-quality checks and the original model pending review.\n'
        '6. Do not evaluate new candidates on November–December. Those outcomes were already reported historically, so the original final holdout is no longer pristine for this project team. This study provides development evidence only; a future claim of generalization needs genuinely new untouched data or a prospectively agreed evaluation, not repeated use of the old test months.',
        '## Reproduction and artifacts',
        'Run python -m experiments.modeling_study once using the verified PythonProject4/.venv interpreter. It refuses an existing study output directory. Run python -m experiments.study_report to render this report from saved tables. Open notebooks/modeling_improvement_study.ipynb to regenerate figures and inspect saved results without fitting or test inference. Outputs, experimental models, raw fold scores, thresholds, validation predictions, reliability bins, group importance, protocol, version snapshot and preservation manifest live only under outputs/experiments/modeling_study/. No dashboard or main-model integration is included.',
        'Conceptual references: [scikit-learn time feature engineering](https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html) discusses periodic encodings. [scikit-learn probability calibration](https://scikit-learn.org/stable/modules/calibration.html) distinguishes reliability diagrams from overall proper scoring rules. The implementation uses the verified scikit-learn 1.7.2 environment, not the current documentation version.'
    ]
    (ROOT/'reports/modeling_improvement_study.md').write_text('\n\n'.join(parts)+'\n')
    return comparison


if __name__=='__main__': render()
