"""Render report/notebook from saved score tables; no fitting or new predictions."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import nbformat as nb


def table(frame):
    def value(x):
        if isinstance(x,(float,np.floating)): return 'undefined' if pd.isna(x) else f'{x:.6f}'
        return str(x)
    return '\n'.join(['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']+
        ['| '+' | '.join(value(x) for x in row)+' |' for row in frame.itertuples(index=False,name=None)])


def render(out):
    from .run import ROOT,NEW_REPORT,NEW_NOTEBOOK,verify_preservation
    report=ROOT/NEW_REPORT;notebook=ROOT/NEW_NOTEBOOK
    if report.exists() or notebook.exists():raise FileExistsError('Refuse overwriting an existing independent report/notebook')
    protocol=json.loads((out/'protocol.json').read_text())
    fold=pd.read_csv(out/'fold_metrics.csv');primary=fold.loc[fold.grid.eq('primary')]
    monthly=pd.read_csv(out/'fold_monthly_metrics.csv')
    block=pd.read_csv(out/'fold_d_block_bootstrap.csv')
    historical=pd.read_csv(out/'historical_recomputed.csv') if (out/'historical_recomputed.csv').exists() else pd.DataFrame()
    hist_month=pd.read_csv(out/'historical_saved_monthly.csv') if (out/'historical_saved_monthly.csv').exists() else pd.DataFrame()
    rank=pd.read_csv(out/'historical_train_validation_ranking.csv') if (out/'historical_train_validation_ranking.csv').exists() else pd.DataFrame()
    status=json.loads((out/'history_status.json').read_text())
    decomposition=json.loads((out/'prevalence_decomposition.json').read_text())
    data=json.loads((out/'data_audit.json').read_text())
    convergence=json.loads((out/'convergence.json').read_text())
    fresh=json.loads((out/'fresh_original_lr.json').read_text()) if (out/'fresh_original_lr.json').exists() else {}
    preservation=verify_preservation(out)
    parts=['# Independent model audit — Stage 1',
        f'Run `{out.name}`. This report separates verification of historical saved test results, recomputation of earlier-study development results, and newly fitted exploratory development evidence. All months and full-year EDA have previously been seen. **No new November–December model predictions were made. No period is a newly untouched test set.** Production models, original findings, lock/selection files and the prior study remain unchanged.',
        '## Provenance and exact commands',
        f'Interpreter: `{protocol["interpreter"]}`. Python: `{protocol["python"]}`. Git commit: `{protocol["git_commit"]}`. Initial status (including pre-existing untracked work):\n\n```text\n'+protocol['git_status']+'```',
        table(pd.DataFrame([{'package':name,'version':version} for name,version in protocol['versions'].items()])),
        '```text\n'+'\n'.join(protocol['commands'])+'\n```',
        'No packages were upgraded. NumPy, joblib, SciPy and threadpoolctl are directly imported by this audit but transitively supplied by the current requirements; nbformat/nbclient support notebook tooling. These should be explicitly declared in a future dependency update, with the verified versions above. Existing requirements are preserved. No applicable AGENTS.md was found in repository/ancestor inspection.',
        'protocol.json was written before verification/fitting and contains exact candidate parameters, folds, threshold grids, seed 42, bootstrap design, versions, Git status and SHA-256 checksums of audit source and protected data/artifacts. New run creation refuses an existing directory. Permanent phase locks prevent accidental duplicate phases; a completed phase is skipped. No lock is deleted. All outputs are under '+str(out.relative_to(ROOT))+'/.',
        f'Preservation: **{preservation["protected_file_count"]} existing files byte-identical** before/after. The protected set includes cached raw/processed data, source, original reports/notebooks/tests, production artifacts and prior-study files, including untracked work. Manifests are in protocol.json and preservation_final.json (created by the final check command).',
        '## 1. Independent reconstruction and data integrity',
        table(pd.DataFrame([{'check':key,'result':value} for key,value in data.items() if key!='classifier_missing_inputs'])),
        'The cached raw snapshot was checksum-verified against source.json. Target, dates, calendar features, schedules and split were independently reconstructed in memory without invoking prepare() or main.py. Exact column/key coverage and every value/missing mask match descriptive, classifier and missing-label exports by flight_id. No approved classifier input is missing; no new imputer was introduced. The original pipelines retain their already-fitted imputers.',
        table(pd.read_csv(out/'label_boundaries.csv')),
        table(pd.read_csv(out/'natural_keys.csv')),
        table(pd.read_csv(out/'local_clock_consistency.csv')),
        'The independent clock checker accepts 2400 as midnight and rejects malformed HHMM. Actual and scheduled departure clocks use origin local time, and arrival clocks use destination local time. Delay consistency is checked within each same-local-clock pair modulo 1440; arrival is never subtracted from departure to infer flight duration. Negative delays and observed extreme delays are retained. Modulo checks cannot identify an incorrect whole-day offset. Missing labels remain unknown, never zero or an assumed cancellation.',
        table(pd.read_csv(out/'historical_missing_by_month.csv')),
        'All reconstructed-column missing counts are in data_missingness.csv, including outcomes not eligible for prediction.',
        '## 2. Historical saved-score verification — including old test results',
        str(status),
        'Saved row probabilities were aligned to exact classifier IDs/dates/splits/targets; stored predictions must equal probability >= their stored threshold. Confusion counts and metrics were independently recalculated. No model was called on November–December. Missing artifacts, if any, are listed above rather than fabricated or recreated.',
    ]
    if not historical.empty:
        parts.append(table(historical[['model','split','threshold','n','mean_probability','prevalence','accuracy','precision','recall','f1','roc_auc']]))
    parts.extend(['Historical monthly probability/calibration diagnostics (November–December rows below use saved probabilities only):'])
    if not hist_month.empty:
        parts.append(table(hist_month[['model','month','prevalence','mean_probability','precision','recall','f1','roc_auc','average_precision','tpr','fpr','f1_q20','brier','log_loss']]))
    parts.extend(['## 3. Historical pipeline and preprocessing audit',
        'Compatible saved pipelines were loaded and called only on January–October. Their September–October probabilities were compared by flight_id to saved probabilities. Scaler means/sample counts and category vocabularies were checked against January–August only; unseen-category probes must produce finite probabilities. Numeric roles are the original approved numeric list, including month/day_of_year/weekday and schedule clocks; carrier/flight/origin/dest are one-hot categories. Full feature names, dimensions and original n_iter_ are saved in historical_pipeline_checks.csv.',
        table(pd.read_csv(out/'historical_pipeline_checks.csv')) if (out/'historical_pipeline_checks.csv').exists() else 'Pipeline artifacts unavailable.',
        'Fresh development-only original LR comparison (same C/solver/seed, iteration ceiling raised to 1000; no demand for bit-identical floats):\n\n```json\n'+json.dumps(fresh,indent=2)+'\n```',
        'Matched train and development ranking: training scores below are in-sample and optimistic; they are not future performance estimates. A large train/development gap is diagnostic, not proof that one feature alone causes overfitting.',
        table(rank[['model','period','n','prevalence','mean_probability','roc_auc','average_precision','f1','brier','log_loss']]) if not rank.empty else 'Unavailable.',
        'Monthly train/development probability means and all metrics are in historical_development_monthly.csv; original training uses in-sample predictions. Category sizes and per-month unknown counts/shares are in historical_category_coverage.csv. Original January–August controls (majority, always positive, constant training prior) are evaluated per development month in historical_development_controls.csv.',
        '## 4. Temporal covariates and prevalence decomposition',
        table(pd.read_csv(out/'covariate_distances.csv')),
        'TV is half the summed absolute difference of categorical proportions over their union; KS is the maximum numerical empirical-CDF gap. Month/day-of-year ranges are necessarily disjoint by design. A KS distance of 1 for calendar variables is not itself evidence of harmful concept drift or causality.',
        table(pd.DataFrame([{'quantity':key,'value':value} for key,value in decomposition.items()])),
        'Common strata are origin × carrier × four-hour scheduled-departure bands with observations in both training and validation; no outcome-based support cutoffs are selected. On that common support, training prevalence uses training weights; expected validation rate applies validation weights to training stratum rates; the residual is observed validation prevalence minus that expected rate. Coverage is reported separately so unsupported schedules are not silently included. The common-support change equals mix component plus residual, not necessarily the all-data change. Coarse observational strata do not establish pure label shift, concept drift mechanism, airline causality, or weather effects.',
        '## 5. Prior modeling study — reproduced saved results, not a rerun',
        str(json.loads((out/'prior_study_status.json').read_text())),
    ])
    if (out/'prior_study_recomputed.csv').exists():
        parts.append(table(pd.read_csv(out/'prior_study_recomputed.csv')[['candidate','f1','roc_auc','average_precision','mean_probability','prevalence']]))
    parts.extend(['Previous internal fold tables were inspected as saved; their models were not refit. Earlier published validation metrics were recomputed from prior saved row probabilities aligned to development IDs. Reproduction of those arithmetic results does not validate prior feature/model selection as unbiased.',
        '## 6. Exactly seven fixed candidates; no search',
        table(pd.DataFrame([
            {'candidate':'A','change':'Original LR; C=1, liblinear, max_iter=1000, random_state=42'},
            {'candidate':'B','change':'A minus month and day_of_year'},
            {'candidate':'C','change':'A minus flight number'},
            {'candidate':'D','change':'A with categorical weekday; weekend retained'},
            {'candidate':'E','change':'B with departure spline, five uniform knots, degree 3, no bias, linear extrapolation'},
            {'candidate':'F','change':'Compact LR: no month/day_of_year/flight; categorical carrier/origin/dest/weekday'},
            {'candidate':'G','change':'Compact native-categorical HGB on exactly F inputs; fixed parameters'}])),
        'F/G numerical predictors are is_weekend, sched_dep_minutes, sched_arr_minutes and distance. LR numerics and the E spline basis are scaled on fit data; categories are train-fitted one-hot with unknowns ignored. G uses train-fitted OrdinalEncoder with unknowns -> NaN, explicit categorical column indices [0,1,2,3], and raw numerics. Codes are not treated as ordered continuous values. HGB uses max_iter=100, max_leaf_nodes=15, min_samples_leaf=100, learning_rate=0.1, l2_regularization=10, early_stopping=False, random_state=42; no random validation is performed. Train cardinalities are checked against the installed max_bins=255 and recorded in convergence.json. There is no class weighting, SMOTE, weather, target encoding, tail number, rolling outcome variable or extra fitted algorithm.',
        table(pd.DataFrame([{'fold':key,'fit':f'Jan–month {a}','threshold_month':b,'assessment_months':str(c)} for key,(a,b,c) in __import__('experiments.independent_audit.core',fromlist=['FOLDS']).FOLDS.items()])),
        'All are reused exploratory development periods. Each full preprocessing pipeline is freshly fit on fit months only, with no shared estimator instances. Thresholds use only their designated later month. The primary threshold grid is 0.10–0.50 step 0.025; ties favor closest to 0.5 then higher. A separate 0–1 step .01 diagnostic grid includes always positive and 1.01 for all negative. Neither assessment labels nor a comparison of assessment grids chooses which result to keep: the primary grid remains primary. Both sets of fixed thresholds/curves are retained.',
        'All following assessment metrics are calculated by reading development_predictions.csv, not by trusting metrics from a training loop. Row scores contain flight_id/date/fold/candidate/grid/probability/threshold/target. Tune-only probabilities and threshold curves are saved separately.',
    ])
    for key,g in primary.groupby('fold'):
        parts.extend([f'### Fold {key} — primary grid',table(g[['candidate','threshold','n','prevalence','mean_probability','accuracy','balanced_accuracy','precision','recall','f1','roc_auc','average_precision','f1_q20','brier','log_loss']])])
    parts.extend(['### Equal-weight summary of (a)–(c) only',table(pd.read_csv(out/'equal_weight_abc.csv')),
        'The first three folds receive equal weight despite the longer July–August assessment. The fourth fold is shown separately; assessments overlap earlier threshold/fit history and are not pooled as an independent test estimate.',
        '### Matched ablation changes',table(pd.read_csv(out/'ablation_differences.csv')),
        'Compare B/C/D to A, E to B, and G to F. F versus A is a feature bundle and cannot identify the effect of any single removal/encoding. Results that improve only one period are unstable rather than general improvement.',
        '### Per-month assessment metrics',
        table(monthly.loc[monthly.grid.eq('primary'),['fold','candidate','month','n','positive_count','prevalence','mean_probability','precision','recall','f1','roc_auc','average_precision','tpr','fpr','f1_q20','brier','log_loss']]),
        'Full numeric diagnostics including accuracy/confusion counts are in fold_monthly_metrics.csv. Mean probability versus observed prevalence exposes over/underprediction; Brier/log loss and reliability bins measure related aspects of probability quality. They combine calibration and discrimination and do not alone prove calibration. No calibrator is fit in Stage 1.',
        'AP is non-interpolated average precision, not trapezoidal PR area; constant-score AP equals positive prevalence, so compare AP to each period’s prevalence. ROC-AUC measures ranking at all thresholds. Precision/recall/F1 and accuracy depend on the selected operating threshold. An F1-selected low threshold can flag many on-time flights and lower accuracy versus majority prediction while improving delay recall; accuracy optimizes a different error trade-off. Undefined precision remains NaN/undefined.',
        'Fixed-prevalence F1 is 2qTPR/[q(1+TPR)+(1-q)FPR], q=0.20. It descriptively standardizes prevalence while holding observed TPR/FPR fixed. It is not a causal adjustment, probability recalibration, forecast or evidence that rates transport to another population.',
        '### Diagnostic wider-grid results (not a replacement operating point)',
        table(fold.loc[fold.grid.eq('diagnostic'),['fold','candidate','threshold','f1','roc_auc','average_precision','mean_probability']]),
        '### Convergence and dimensions',
        table(pd.DataFrame([{'fold':r['fold'],'candidate':r['candidate'],'fit_n':r['n_fit'],'dimension':r['encoded_dimension'],
            'iterations':str(r['iterations']),'convergence_warnings':r['convergence_warnings'],'warnings':str(r['warnings']),
            'category_counts':str(r.get('categorical_cardinalities','n/a'))} for r in convergence])),
        '## 7. Paired uncertainty with predictions and thresholds fixed',
        'Historical comparisons resample whole September–October calendar days, using identical sampled days for each pair. These are existing saved scores, some with historically validation-tuned thresholds. No historical model is refit for this bootstrap.',
    ])
    if (out/'historical_day_bootstrap.csv').exists():parts.append(table(pd.read_csv(out/'historical_day_bootstrap.csv').query("metric == 'f1'")))
    parts.extend(['Fold (d): 500 fixed-seed replicates use moving blocks of seven consecutive calendar days. Blocks are sampled with replacement from valid starts within the September–October calendar and concatenated/truncated to its original length. No artificial wrap joins October to September. All A/C/F/G pairs share each replicate’s day weights. Point differences and 95% percentile intervals:',table(block),
        'These intervals condition on the fitted scores/thresholds. They do not correct for previous selection, repeatedly examined validation, overlapping folds, nonstationarity, or all seasonal dependence. Whole-day resampling addresses within-day flight dependence; seven-day blocks retain some short-range dependence, not every temporal mechanism. Flight count alone is not a reason to call a small gain significant. Endpoint dates have fewer possible moving-block memberships; interpret intervals as limited diagnostics.',
        '## 8. Independent findings and proposed next decision',
    ])
    d=primary.loc[primary.fold.eq('d')].set_index('candidate')
    saved_lr=historical.loc[historical.model.eq('logistic_regression')&historical.split.eq('validation')].iloc[0] if not historical.empty else None
    if saved_lr is not None:
        parts.append(f'Historical LR validation mean probability {saved_lr.mean_probability:.6f}, observed prevalence {saved_lr.prevalence:.6f}, AUC {saved_lr.roc_auc:.6f}, F1 {saved_lr.f1:.6f}. This is the original Jan–Aug fit with Sep–Oct-tuned threshold 0.325, not fold (d).')
    parts.append(table(d.loc[list('ACFG'),['threshold','prevalence','mean_probability','f1','roc_auc','average_precision','brier','log_loss']].reset_index()))
    parts.append('Independent numeric checks are comparisons, never optimization targets. Expected approximate fold (d) reference values and actual differences:')
    checks=[]
    for name,auc,f1 in [('A',.67035,.32355),('F',.68496,.33884),('G',.69148,.34345)]:
        checks.append({'candidate':name,'actual_auc':d.loc[name,'roc_auc'],'reference_auc':auc,'auc_difference':d.loc[name,'roc_auc']-auc,
            'actual_f1':d.loc[name,'f1'],'reference_f1':f1,'f1_difference':d.loc[name,'f1']-f1})
    parts.append(table(pd.DataFrame(checks)))
    parts.extend(['Interpret changes across all four fold tables and paired intervals, not just aggregate September–October scores. Flight removal is isolated by C versus A; compact capacity by G versus F. If the compact candidates improve ranking/F1 but still overpredict average risk, feature simplification has not solved temporal calibration drift.',
        'Proposed next decision: review the single-feature C ablation and the matched F/G results with their temporal stability and block intervals. Retain the historical production LR until an explicit integration decision. Any further calibration study needs a separately agreed chronological design; no calibration, extra search, old-test reassessment, dashboard, commit or push is performed here.',
        '## 9. Tests and limitations',
        'Focused tests cover independent mismatch detection, unknown-label preservation, threshold/boundary cases, Nov–Dec sentinel exclusion before conversion, strict fold chronology, LR and native-HGB unknown categories, isolated estimator instances, outcome/metadata isolation, weighted rank metrics versus explicit row replication, phase duplicate refusal and protected-file mutation detection. Existing tests run unchanged. Exact test command: `'+protocol['interpreter']+' -B -m unittest discover -s tests -v`.',
        'Only one year of NYC departures is represented. Missing departure labels may be nonrandom. Calendar gaps are structural and the prevalence decomposition is coarse and observational. Original and earlier-study outputs were already reviewed. The dataset has no newly untouched future period for this team. No feature is approved for production merely because it wins one reused development comparison.',
        'Execution note: Mac sleep interrupted code construction before any independent-audit run existed; process inspection confirmed no active experiment. The completed run starts from that partial code, not a duplicated experiment. The prior study is preserved rather than reconstructed. Notebook charts load saved audit tables only and never retrain or call a model.'] )
    report.write_text('\n\n'.join(parts)+'\n',encoding='utf-8')
    create_notebook(notebook,out)


def create_notebook(path,out):
    from .run import ROOT
    md=nb.v4.new_markdown_cell;code=nb.v4.new_code_cell
    notebook=nb.v4.new_notebook()
    relative=out.relative_to(ROOT).as_posix()
    notebook.cells=[md('''# Independent model audit — Stage 1

This notebook **only reads saved audit tables**. It does not fit models, call predict_proba, regenerate source data, or write figures/reports. Historical saved November–December verification is separate from new January–October development evidence. All periods were previously examined; none is a newly untouched test set.

Read `reports/independent_model_audit.md` for protocol, commands, diagnostics and limits.'''),
        code(f'''from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
ROOT=Path.cwd()
if not (ROOT/'experiments/independent_audit').exists(): ROOT=ROOT.parent
OUT=ROOT/{relative!r}
fold=pd.read_csv(OUT/'fold_metrics.csv')
primary=fold.loc[fold.grid.eq('primary')]
monthly=pd.read_csv(OUT/'fold_monthly_metrics.csv')
print('Run:',OUT.name)
display(pd.DataFrame(json.loads((OUT/'protocol.json').read_text())['versions'].items(),columns=['package','version']))'''),
        md('## Historical verification — saved test probabilities only\nFull-year reconstruction is audited by flight_id. These charts load saved verification tables; no new inference is performed.'),
        code('''display(pd.read_csv(OUT/'historical_recomputed.csv'))
display(pd.read_csv(OUT/'historical_pipeline_checks.csv'))
history=pd.read_csv(OUT/'historical_development_monthly.csv')
saved=pd.read_csv(OUT/'historical_saved_monthly.csv')
fig,axes=plt.subplots(1,2,figsize=(14,4))
for name in history.model.unique():
    g=history.loc[history.model.eq(name)]
    axes[0].plot(g.month,g.mean_probability,marker='.',label=name)
obs=history.loc[history.model.eq('logistic_regression')]
axes[0].plot(obs.month,obs.prevalence,'k--',label='Observed')
axes[0].set(title='Jan–Aug in-sample / Sep–Oct development',xlabel='Month',ylabel='Mean probability / observed prevalence')
axes[0].axvline(8.5,color='gray',linestyle=':');axes[0].legend(fontsize=7)
g=saved.loc[saved.model.eq('logistic_regression')]
axes[1].plot(g.month,g.mean_probability,marker='o',label='Saved LR mean probability')
axes[1].plot(g.month,g.prevalence,'k--',label='Observed')
axes[1].set(title='Historical saved scores (no new test inference)',xlabel='Month');axes[1].legend()
plt.tight_layout();plt.show()
display(pd.read_csv(OUT/'historical_train_validation_ranking.csv'))'''),
        md('## Covariates and observational prevalence decomposition\nDisjoint calendar ranges alone do not prove harmful concept drift. Stratum rates separate a coarse schedule-mix component from a residual within-stratum change on reported common support; neither identifies causes.'),
        code('''display(pd.read_csv(OUT/'covariate_distances.csv'))
display(pd.Series(json.loads((OUT/'prevalence_decomposition.json').read_text())))
display(pd.read_csv(OUT/'prior_study_recomputed.csv'))'''),
        md('## New development evidence: fixed ablations and threshold transfer\nA original; B no month/day-of-year; C no flight; D categorical weekday; E B with departure spline; F compact LR; G matched compact native-categorical HGB. Each threshold is fitted only on its designated past month. Fold d uses Jan–Jul fit / August threshold / Sep–Oct assessment. The diagnostic wider grid does not replace the primary grid.'),
        code('''for label,g in primary.groupby('fold'):
    print('Fold',label)
    display(g[['candidate','threshold','n','prevalence','mean_probability','accuracy','balanced_accuracy','precision','recall','f1','roc_auc','average_precision','brier','log_loss']])
display(pd.read_csv(OUT/'equal_weight_abc.csv'))
fig,axes=plt.subplots(1,2,figsize=(14,4))
for name in 'ABCDEFG':
    g=primary.loc[primary.candidate.eq(name)]
    axes[0].plot(g.fold,g.f1,marker='o',label=name)
    axes[1].plot(g.fold,g.roc_auc,marker='o',label=name)
axes[0].set(title='Assessment F1: fixed past-month threshold',ylabel='F1');axes[1].set(title='Assessment ROC-AUC',ylabel='AUC')
axes[1].legend(ncol=4);plt.tight_layout();plt.show()'''),
        md('## Monthly probability quality and reliability\nTraining metrics are optimistic. AP depends on prevalence. Low F1-selected thresholds can reduce accuracy while increasing recall. q=0.20 F1 is descriptive standardization, not calibration or a causal adjustment.'),
        code('''g=monthly.loc[monthly.fold.eq('d')&monthly.grid.eq('primary')]
display(g[['candidate','month','n','positive_count','prevalence','mean_probability','precision','recall','f1','roc_auc','average_precision','tpr','fpr','f1_q20','brier','log_loss']])
bins=pd.read_csv(OUT/'fold_reliability.csv')
fig,axes=plt.subplots(1,2,figsize=(14,4))
for name in 'ACFG':
    for month in [9,10]:
        b=bins.loc[bins.fold.eq('d')&bins.grid.eq('primary')&bins.candidate.eq(name)&bins.month.eq(month)]
        axes[0].plot(b.mean_probability,b.observed_rate,marker='.',label=f'{name} month {month}')
        axes[1].plot(b.mean_probability,b.n,marker='.',label=f'{name} month {month}')
axes[0].plot([0,1],[0,1],'k--');axes[0].set(title='Out-of-time development reliability',xlabel='Mean probability',ylabel='Observed rate')
axes[1].set(title='Reliability bin counts',xlabel='Mean probability',ylabel='Flights',yscale='log')
axes[0].legend(fontsize=7);plt.tight_layout();plt.show()
display(bins.loc[bins.fold.eq('d')&bins.grid.eq('primary')&bins.candidate.isin(list('ACFG'))])'''),
        md('## Paired day and seven-day block uncertainty\n500 fixed-seed replicates retain fitted predictions/thresholds. Intervals do not correct prior selection, reused data, overlapping folds or all seasonality. Do not infer significance merely from flight count.'),
        code('''intervals=pd.read_csv(OUT/'fold_d_block_bootstrap.csv')
display(intervals)
fig,axes=plt.subplots(1,3,figsize=(16,5))
for ax,metric in zip(axes,['f1','roc_auc','average_precision']):
    g=intervals.loc[intervals.metric.eq(metric)].reset_index(drop=True)
    labels=g.candidate+' minus '+g.reference
    for i,row in g.iterrows():
        ax.plot([row.ci_low,row.ci_high],[i,i],color='steelblue')
        ax.plot(row.point_difference,i,'ko')
    ax.axvline(0,color='gray',linestyle='--');ax.set(yticks=range(len(g)),yticklabels=labels,title=metric)
plt.tight_layout();plt.show()
display(pd.read_csv(OUT/'historical_day_bootstrap.csv').query("metric == 'f1'"))
display(pd.DataFrame(json.loads((OUT/'convergence.json').read_text()))[['fold','candidate','iterations','encoded_dimension','convergence_warnings']])'''),
        md('## Review boundary\nNo production model, previous report, source dataset, prior study or selection file is replaced. This notebook performs no next-stage work. Review the independent report and paired differences before deciding whether any simplification merits integration.')]
    notebook.metadata={'kernelspec':{'display_name':'Python 3 (verified audit environment)','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.13.13'}}
    nb.write(notebook,path)
