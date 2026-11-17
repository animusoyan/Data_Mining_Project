"""Isolated Stage 2 commands: start, execute, render, check. Never refit a base model."""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.independent_audit.core import FEATURES,load_development,probability
from .core import (COMPARISONS,EPS,GRID,acquire_scores,block_bootstrap,calibrated,check_chronology,
                   choose,equivalent_decisions,evaluate,fit_intercept,reliability,threshold_curve)

ROOT=Path(__file__).resolve().parents[2]
STAGE1=ROOT/'outputs/experiments/independent_audit/stage1_audit_001'
BASE=ROOT/'outputs/experiments/calibration_study'
PREEXISTING=Path('/private/tmp/nyc_stage2_preexisting.json')
REPORT=ROOT/'reports/chronological_calibration_study.md'
NOTEBOOK=ROOT/'notebooks/chronological_calibration_study.ipynb'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def source_hashes():
    paths=list((ROOT/'experiments/calibration_study').glob('*.py'))+[ROOT/'tests/test_calibration_study.py']
    return {p.relative_to(ROOT).as_posix():digest(p) for p in paths if p.exists()}


def verify_files(manifest,root=ROOT):
    changed=[name for name,expected in manifest.items() if not (root/name).is_file() or digest(root/name)!=expected]
    if changed:raise ValueError('Pre-existing protected files changed: '+str(changed))
    return {'unchanged':True,'protected_file_count':len(manifest)}


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def check(out):
    protocol=json.loads((out/'protocol.json').read_text())
    result=verify_files(protocol['preexisting_files'])
    if source_hashes()!=protocol['stage2_source_hashes']:raise ValueError('Stage 2 source changed after protocol')
    return result


def start(out,baseline=PREEXISTING):
    # Portable reruns may pass an explicitly captured before-additions inventory.
    if not baseline.exists():raise FileNotFoundError('Pre-existing inventory required; do not invent or bypass it')
    before=json.loads(baseline.read_text());verify_files(before)
    out.mkdir(parents=True,exist_ok=False)
    models={}
    for name,period in COMPARISONS.items():
        for candidate in ['F','G']:
            path=STAGE1/f'fold_{period["fold"]}_{candidate}.joblib'
            if not path.exists():raise FileNotFoundError('Frozen Stage 1 model absent: '+str(path))
            model=joblib.load(path)
            models[f'{name}_{candidate}']={'path':str(path.relative_to(ROOT)),'sha256':digest(path),
                'parameters':model.named_steps['model'].get_params(),
                'approved_inputs':list(model.named_steps['inputs'].columns)}
    inputs={}
    paths=[ROOT/'data/processed/classifier.csv',STAGE1/'protocol.json',STAGE1/'convergence.json',STAGE1/'development_predictions.csv']
    paths += [STAGE1/f'tuning_{period["fold"]}_{candidate}.csv' for period in COMPARISONS.values() for candidate in ['F','G']]
    for path in paths:
        if path.exists():inputs[str(path.relative_to(ROOT))]=digest(path)
    versions={name:importlib.metadata.version(name) for name in ['numpy','scipy','pandas','scikit-learn','joblib','matplotlib','threadpoolctl','nbformat','nbclient','ipykernel']}
    protocol={'run_id':out.name,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'interpreter':sys.executable,'python':sys.version,'versions':versions,'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'git_status':subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
        'preexisting_files':before,'stage2_source_hashes':source_hashes(),'input_hashes':inputs,'base_models':models,
        'comparisons':COMPARISONS,'methods':['raw unchanged','intercept-only logit offset; slope fixed 1'],
        'fit':'unpenalized binary log loss; brentq root of mean(expit(logit(clipped p)+b))-prevalence, xtol 1e-12',
        'clipping':{'epsilon':EPS,'raw':'unchanged','calibration_logit':'clip to [epsilon,1-epsilon]',
            'log_loss':'clip each method only for evaluation logs to [epsilon,1-epsilon]',
            'boundary':'equivalent-decision comparisons allow documented <=1e-12 boundary proximity only'},
        'controls':['base-training-label prevalence','calibration-label prevalence'],
        'thresholds':{'grid':GRID.tolist(),'tie':'maximum threshold-month F1, then closest to 0.5, then higher',
            'methods':'raw and offset selected independently; primary loss comparisons never use thresholds'},
        'metrics':['Brier','binary log loss','prevalence','mean_probability','mean_prediction_error','reliability bins with n',
            'ROC-AUC','average_precision','precision','recall','F1','accuracy','TN','FP','FN','TP'],
        'bootstrap':{'replicates':500,'seed':42,'unit':'paired seven-calendar-day moving blocks; noncircular valid starts',
            'contrast':'calibrated minus raw; negative helps','scope':'each assessment month and combined period only within comparison 2'},
        'label_availability':'All calibration-month labels known before threshold month; all threshold-month labels known before assessment. Batch-month retrospective assumption, not validated real-time operation.',
        'rules':['No base model/preprocessor fit','Reuse exact Stage 1 scores; infer only missing January-October rows',
            'Never infer or use November-December outcomes','No pooled independent estimate across overlapping comparisons',
            'No method/search expansion, calibration updates, or production integration'],
        'execution_environment':{'LOKY_MAX_CPU_COUNT':os.environ.get('LOKY_MAX_CPU_COUNT'),'native_thread_limit':2}}
    write_json(out/'preexisting_inventory.json',before)
    write_json(out/'protocol.json',protocol)
    print('Protocol saved before calibration:',out,flush=True)


def claim(out,phase):
    check(out)
    if (out/f'{phase}.done.json').exists():print('Already complete; skip',phase);return False
    with (out/f'{phase}.lock').open('x') as f:json.dump({'pid':os.getpid(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},f)
    return True


def caches():
    parts=[]
    for chunk in pd.read_csv(STAGE1/'development_predictions.csv',chunksize=150000,float_precision='round_trip'):
        part=chunk.loc[chunk.candidate.isin(['F','G'])&chunk.grid.eq('primary')]
        if len(part):parts.append(part[['flight_id','date','fold','candidate','probability','target']])
    saved=pd.concat(parts,ignore_index=True)
    result={}
    for period in COMPARISONS.values():
        fold=period['fold']
        for candidate in ['F','G']:
            pieces=[saved.loc[saved.fold.eq(fold)&saved.candidate.eq(candidate),['flight_id','date','target','probability']]]
            path=STAGE1/f'tuning_{fold}_{candidate}.csv'
            if path.exists():pieces.append(pd.read_csv(path,float_precision='round_trip'))
            cache=pd.concat(pieces,ignore_index=True)
            if cache.flight_id.duplicated().any():raise ValueError('Duplicate Stage 1 cached scores')
            if pd.to_datetime(cache.date).max()>pd.Timestamp('2013-10-31'):raise ValueError('Cache exceeds development boundary')
            result[(fold,candidate)]=cache
    return result


def summaries_from_saved(out):
    saved=pd.read_csv(out/'assessment_predictions.csv',parse_dates=['date'],float_precision='round_trip')
    rows=[];bins=[]
    for (comparison,base,method,month),g in saved.groupby(['comparison','base','method','month']):
        if g.threshold.nunique()!=1:raise ValueError('Assessment thresholds must remain fixed')
        rows.append({'comparison':comparison,'base':base,'method':method,'month':month,'threshold':float(g.threshold.iloc[0]),
            **evaluate(g.target,g.probability,float(g.threshold.iloc[0]))})
        bins.append(reliability(g.target,g.probability).assign(comparison=comparison,base=base,method=method,month=month))
    result=pd.DataFrame(rows)
    result.to_csv(out/'monthly_metrics.csv',index=False)
    pd.concat(bins,ignore_index=True).to_csv(out/'monthly_reliability.csv',index=False)
    raw=result.loc[result.method.eq('raw')].set_index(['comparison','base','month'])
    adjusted=result.loc[result.method.eq('intercept')].set_index(['comparison','base','month'])
    if not raw.index.equals(adjusted.index):raise ValueError('Methods have different assessment groups')
    delta=adjusted[['brier','log_loss','f1','roc_auc','average_precision','mean_prediction_error']]-raw[['brier','log_loss','f1','roc_auc','average_precision','mean_prediction_error']]
    delta.add_suffix('_calibrated_minus_raw').reset_index().to_csv(out/'monthly_differences.csv',index=False)
    bootstrap_rows=[];bootstrap_draws=[]
    for (comparison,base),g in saved.groupby(['comparison','base']):
        a=g.loc[g.method.eq('raw')].set_index('flight_id').sort_index()
        b=g.loc[g.method.eq('intercept')].set_index('flight_id').sort_index()
        if not a.index.equals(b.index) or not a.target.equals(b.target) or not a.date.equals(b.date):raise ValueError('Unpaired method scores')
        paired=pd.DataFrame({'date':a.date,'target':a.target,'raw_probability':a.probability,'calibrated_probability':b.probability})
        scopes=[(str(month),p) for month,p in paired.groupby(paired.date.dt.month)]
        if paired.date.dt.month.nunique()>1:scopes.append(('combined_within_comparison',paired))
        for scope,frame in scopes:
            summary,draws=block_bootstrap(frame,500,42)
            bootstrap_rows.append(summary.assign(comparison=comparison,base=base,assessment_scope=scope))
            bootstrap_draws.append(draws.assign(comparison=comparison,base=base,assessment_scope=scope))
    pd.concat(bootstrap_rows,ignore_index=True).to_csv(out/'paired_block_intervals.csv',index=False)
    pd.concat(bootstrap_draws,ignore_index=True).to_csv(out/'paired_block_draws.csv',index=False)


def execute(out):
    data=load_development(ROOT/'data/processed/classifier.csv')
    cache=caches();provenance=[];parameters=[];raw_scores=[];assessment_rows=[];curves=[];thresholds=[];equivalence=[];model_integrity=[]
    for comparison,period in COMPARISONS.items():
        fit=data.loc[data.month<=period['fit_end']]
        cal=data.loc[data.month.eq(period['calibration'])]
        tune=data.loc[data.month.eq(period['threshold'])]
        assess=data.loc[data.month.isin(period['assessment'])]
        check_chronology(fit,cal,tune,assess)
        training_prior=float(fit.delayed_15.mean());calibration_prior=float(cal.delayed_15.mean())
        for candidate in ['F','G']:
            print('Comparison',comparison,'frozen base',candidate,flush=True)
            path=STAGE1/f'fold_{period["fold"]}_{candidate}.joblib'
            disk_hash=digest(path);model=joblib.load(path);memory_hash=joblib.hash(model)
            probabilities={}
            for label,frame in [('calibration',cal),('threshold',tune),('assessment',assess)]:
                p,counts=acquire_scores(frame,cache[(period['fold'],candidate)],model,probability)
                probabilities[label]=p
                provenance.append({'comparison':comparison,'base':candidate,'period':label,'first_date':str(frame.date.min().date()),'last_date':str(frame.date.max().date()),**counts})
                raw_scores.append(frame[['flight_id','date','month']].assign(comparison=comparison,base=candidate,
                    period=label,target=frame.delayed_15,raw_probability=p))
            fitted=fit_intercept(probabilities['calibration'],cal.delayed_15)
            intercept=fitted['intercept']
            parameters.append({'comparison':comparison,'base':candidate,**fitted,'base_path':str(path.relative_to(ROOT)),
                'base_sha256':disk_hash,'base_fit_first_date':str(fit.date.min().date()),'base_fit_last_date':str(fit.date.max().date()),
                'calibration_first_date':str(cal.date.min().date()),'calibration_last_date':str(cal.date.max().date()),
                'threshold_first_date':str(tune.date.min().date()),'threshold_last_date':str(tune.date.max().date()),
                'assessment_first_date':str(assess.date.min().date()),'assessment_last_date':str(assess.date.max().date()),
                'training_prior':training_prior,'calibration_prior':calibration_prior})
            method_thresholds={}
            for method in ['raw','intercept']:
                transform=(lambda p:p) if method=='raw' else (lambda p:calibrated(p,intercept))
                curve=threshold_curve(tune.delayed_15,transform(probabilities['threshold']))
                chosen=choose(curve);method_thresholds[method]=chosen
                curves.append(curve.assign(comparison=comparison,base=candidate,method=method))
                thresholds.append({'comparison':comparison,'base':candidate,'method':method,'threshold':chosen})
                assessment_rows.append(assess[['flight_id','date','month']].assign(comparison=comparison,base=candidate,method=method,
                    probability=transform(probabilities['assessment']),threshold=chosen,target=assess.delayed_15))
            for month,group in assess.groupby('month'):
                positions=assess.index.get_indexer(group.index)
                result=equivalent_decisions(probabilities['assessment'][positions],intercept,method_thresholds['raw'])
                equivalence.append({'comparison':comparison,'base':candidate,'month':month,**result})
            for method,value in [('training_prior',training_prior),('calibration_prior',calibration_prior)]:
                assessment_rows.append(assess[['flight_id','date','month']].assign(comparison=comparison,base=candidate,method=method,
                    probability=value,threshold=.5,target=assess.delayed_15))
            if digest(path)!=disk_hash or joblib.hash(model)!=memory_hash:raise ValueError('Frozen base model was changed by scoring')
            model_integrity.append({'comparison':comparison,'base':candidate,'disk_sha256_before':disk_hash,'disk_sha256_after':digest(path),
                'memory_hash_before':memory_hash,'memory_hash_after':joblib.hash(model),'unchanged':True,'base_fit_calls':0})
    pd.concat(raw_scores,ignore_index=True).to_csv(out/'raw_period_scores.csv',index=False)
    pd.concat(assessment_rows,ignore_index=True).to_csv(out/'assessment_predictions.csv',index=False)
    pd.concat(curves,ignore_index=True).to_csv(out/'threshold_curves.csv',index=False)
    pd.DataFrame(thresholds).to_csv(out/'selected_thresholds.csv',index=False)
    pd.DataFrame(provenance).to_csv(out/'score_provenance.csv',index=False)
    pd.DataFrame(equivalence).to_csv(out/'equivalent_thresholds.csv',index=False)
    pd.DataFrame(model_integrity).to_csv(out/'base_integrity.csv',index=False)
    write_json(out/'calibration_parameters.json',parameters)
    summaries_from_saved(out)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['start','execute','render','check']);parser.add_argument('--run-id',required=True)
    parser.add_argument('--inventory',type=Path,default=PREEXISTING)
    args=parser.parse_args()
    if not args.run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.run_id):raise ValueError('Invalid run ID')
    out=BASE/args.run_id
    if args.phase=='start':start(out,args.inventory);return
    if args.phase=='check':
        result=check(out)
        if not (out/'preservation_final.json').exists():write_json(out/'preservation_final.json',result)
        print('Pre-existing files unchanged:',result['protected_file_count']);return
    if not claim(out,args.phase):return
    with threadpool_limits(limits=2):
        if args.phase=='execute':execute(out)
        elif args.phase=='render':
            from .report import render
            render(out)
    result=check(out)
    write_json(out/f'{args.phase}.done.json',{'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**result})
    print(args.phase,'complete',flush=True)


if __name__=='__main__':main()
