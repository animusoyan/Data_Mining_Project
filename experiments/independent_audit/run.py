"""Stage 1 commands: start, historical, experiment, uncertainty, render, check."""
from pathlib import Path
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .core import (FEATURES,FOLDS,GRID,DIAGNOSTIC_GRID,build,choose,fit_record,
                   load_development,metrics,partitions,probability,reliability,roles,threshold_curve)

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'outputs/experiments/independent_audit'
NEW_REPORT='reports/independent_model_audit.md'
NEW_NOTEBOOK='notebooks/independent_model_audit.ipynb'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def protected_hashes(root=ROOT):
    """Include raw/processed data and the prior untracked study, not only Git files."""
    files=[]
    for directory in ['src','data','reports','notebooks','tests','outputs','experiments']:
        for p in (root/directory).rglob('*'):
            relative=p.relative_to(root).as_posix()
            if not p.is_file() or '__pycache__' in p.parts or '.ipynb_checkpoints' in p.parts: continue
            if relative.startswith(('experiments/independent_audit/','outputs/experiments/independent_audit/')): continue
            if relative in [NEW_REPORT,NEW_NOTEBOOK,'tests/test_independent_audit.py']: continue
            files.append(p)
    files.extend(p for p in [root/'README.md',root/'requirements.txt',root/'main.py',root/'.gitignore'] if p.exists())
    return {p.relative_to(root).as_posix():digest(p) for p in sorted(set(files))}


def source_hashes():
    paths=list((ROOT/'experiments/independent_audit').glob('*.py'))+[ROOT/'tests/test_independent_audit.py']
    return {p.relative_to(ROOT).as_posix():digest(p) for p in paths if p.exists()}


def json_write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def start(run_id):
    if not run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in run_id):
        raise ValueError('Run ID must be a simple nonempty name')
    out=BASE/run_id
    out.mkdir(parents=True,exist_ok=False)
    specs={}
    for candidate in 'ABCDEFG':
        model=build(candidate)
        nums,cats=roles(candidate)
        specs[candidate]={'numeric':nums,'categorical':cats,'model_parameters':model.named_steps['model'].get_params(),
            'spline':(
                {'n_knots':5,'knots':'uniform','degree':3,'include_bias':False,'extrapolation':'linear','basis_scaled_training_only':True} if candidate=='E' else None)}
    versions={name:importlib.metadata.version(name) for name in ['numpy','scipy','pandas','scikit-learn','matplotlib','joblib','threadpoolctl','nbclient','nbformat','ipykernel','notebook']}
    protected=protected_hashes()
    protocol={'run_id':run_id,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'interpreter':sys.executable,'python':sys.version,'versions':versions,
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'git_status':subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
        'source_checksums':source_hashes(),'protected_before':protected,'seed':42,'thread_limit':2,
        'folds':{key:{'fit_through_month':a,'threshold_month':b,'assessment_months':c} for key,(a,b,c) in FOLDS.items()},
        'candidates':specs,'primary_grid':GRID.tolist(),'diagnostic_grid':DIAGNOSTIC_GRID.tolist(),
        'threshold_tie':'highest tuning F1, then closest to 0.5, then higher',
        'metrics':['accuracy','balanced_accuracy','precision','recall','f1','roc_auc','average_precision','tpr','fpr','f1_q20','mean_probability','prevalence','brier','log_loss','confusion_counts'],
        'bootstrap':{'replicates':500,'seed':42,'historical':'paired whole calendar days, existing Sep-Oct saved scores',
            'fold_d':'paired 7-calendar-day moving blocks, all six pairwise comparisons among A/C/F/G',
            'interval':'2.5th and 97.5th percentiles; fitted scores and thresholds fixed'},
        'constraints':['No production or prior-study modification','No new November-December inference',
            'No model search beyond A-G','All periods reused exploratory development evidence','No imputation of unexpected missing approved inputs'],
        'commands':[f'{sys.executable} -B -m experiments.independent_audit.run {phase} --run-id {run_id}' for phase in ['start','historical','experiment','uncertainty','render','check']]}
    json_write(out/'protocol.json',protocol)
    print('Created protocol:',out,flush=True)
    return out


def verify_preservation(out):
    before=json.loads((out/'protocol.json').read_text())['protected_before']
    after=protected_hashes()
    if before!=after:
        changes=[key for key in set(before)|set(after) if before.get(key)!=after.get(key)]
        raise ValueError('Protected files changed: '+str(changes))
    return {'unchanged':True,'protected_file_count':len(before),'hashes':after}


def claim_phase(out,phase):
    if (out/f'{phase}.done.json').exists():
        print(phase,'already completed; skipping',flush=True);return False
    if not (out/'protocol.json').exists(): raise ValueError('Start the run before executing phases')
    protocol=json.loads((out/'protocol.json').read_text())
    if source_hashes()!=protocol['source_checksums']: raise ValueError('Audit source changed after protocol; start a new run')
    verify_preservation(out)
    # Retain markers permanently; never delete locks to re-enter a phase.
    path=out/f'{phase}.lock'
    with path.open('x') as stream:
        json.dump({'pid':os.getpid(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},stream)
    return True


def summarize_rows(out):
    """All assessment metrics are independently calculated from persisted row scores."""
    scores=pd.read_csv(out/'development_predictions.csv',parse_dates=['date'],float_precision='round_trip')
    rows=[];months=[];bins=[]
    for (fold,candidate,grid),g in scores.groupby(['fold','candidate','grid']):
        if g.threshold.nunique()!=1:raise ValueError('Assessment threshold not fixed')
        threshold=float(g.threshold.iloc[0])
        rows.append({'fold':fold,'candidate':candidate,'grid':grid,'threshold':threshold,**metrics(g.target,g.probability,threshold)})
        for month,mg in g.groupby(g.date.dt.month):
            months.append({'fold':fold,'candidate':candidate,'grid':grid,'month':month,'threshold':threshold,**metrics(mg.target,mg.probability,threshold)})
            bins.append(reliability(mg.target,mg.probability).assign(fold=fold,candidate=candidate,grid=grid,month=month))
    table=pd.DataFrame(rows)
    table.to_csv(out/'fold_metrics.csv',index=False)
    pd.DataFrame(months).to_csv(out/'fold_monthly_metrics.csv',index=False)
    pd.concat(bins,ignore_index=True).to_csv(out/'fold_reliability.csv',index=False)
    abc=table.loc[table.fold.isin(['a','b','c'])&table.grid.eq('primary')]
    metric_names=['f1','roc_auc','average_precision','accuracy','balanced_accuracy','precision','recall','brier','log_loss','f1_q20']
    abc.groupby('candidate')[metric_names].mean().reset_index().to_csv(out/'equal_weight_abc.csv',index=False)
    pairs=[('B','A'),('C','A'),('D','A'),('E','B'),('F','A'),('G','F')]
    differences=[]
    for fold,g in table.loc[table.grid.eq('primary')].groupby('fold'):
        index=g.set_index('candidate')
        for candidate,reference in pairs:
            differences.append({'fold':fold,'candidate':candidate,'reference':reference,
                **{name:index.loc[candidate,name]-index.loc[reference,name] for name in ['f1','roc_auc','average_precision','brier','log_loss']}})
    pd.DataFrame(differences).to_csv(out/'ablation_differences.csv',index=False)


def experiment(out):
    from .history import covariate_audit
    data=load_development(ROOT/'data/processed/classifier.csv')
    covariate_audit(data,out)
    # No imported historical reconstruction data is used here.
    json_write(out/'development_contract.json',{'max_date':str(data.date.max().date()),'rows':len(data),
        'missing_approved_inputs':int(data[FEATURES].isna().sum().sum()),'label_limit':'<=October before target conversion'})
    rows=[];curves=[];convergence=[];thresholds=[];training=[]
    for fold in FOLDS:
        fit,tune,assess=partitions(data,fold)
        print('FOLD',fold,'fit',len(fit),'tune',len(tune),'assess',len(assess),flush=True)
        for candidate in 'ABCDEFG':
            print('Fitting',fold,candidate,flush=True)
            model=build(candidate)
            info=fit_record(model,fit)
            convergence.append({'fold':fold,'candidate':candidate,**info})
            p_tune=probability(model,tune);p_assess=probability(model,assess)
            # Store tuning predictions too, so curves are independently reproducible.
            tune_rows=tune[['flight_id','date']].assign(target=tune.delayed_15,probability=p_tune)
            tune_rows.to_csv(out/f'tuning_{fold}_{candidate}.csv',index=False)
            for grid_name,grid in [('primary',GRID),('diagnostic',DIAGNOSTIC_GRID)]:
                curve=threshold_curve(tune.delayed_15,p_tune,grid)
                t=choose(curve)
                curves.append(curve.assign(fold=fold,candidate=candidate,grid=grid_name))
                thresholds.append({'fold':fold,'candidate':candidate,'grid':grid_name,'threshold':t})
                result=assess[['flight_id','date']].copy()
                result['fold'],result['candidate'],result['grid']=fold,candidate,grid_name
                result['probability'],result['threshold'],result['target']=p_assess,t,assess.delayed_15
                rows.append(result)
            # Matched training scores are optimistic diagnostics, not assessments.
            p_train=probability(model,fit)
            training.append({'fold':fold,'candidate':candidate,'score_scope':'training_optimistic',**metrics(fit.delayed_15,p_train,.5)})
            joblib.dump(model,out/f'fold_{fold}_{candidate}.joblib')
            json_write(out/'convergence.json',convergence)
        prior=float(fit.delayed_15.mean())
        for name,p in [('majority',0.),('always_positive',1.),('training_prior',prior)]:
            result=assess[['flight_id','date']].copy()
            result['fold'],result['candidate'],result['grid']=fold,name,'primary'
            result['probability'],result['threshold'],result['target']=p,.5,assess.delayed_15
            rows.append(result)
    pd.concat(rows,ignore_index=True).to_csv(out/'development_predictions.csv',index=False)
    pd.concat(curves,ignore_index=True).to_csv(out/'threshold_tune_curves.csv',index=False)
    pd.DataFrame(thresholds).to_csv(out/'chosen_thresholds.csv',index=False)
    pd.DataFrame(training).to_csv(out/'new_training_ranking.csv',index=False)
    summarize_rows(out)


def prior_study_verification(out):
    """Recalculate existing development scores, not reconstruct absent prior experiments."""
    base=ROOT/'outputs/experiments/modeling_study'
    pred=base/'validation_predictions.csv';table=base/'validation_metrics.csv'
    if not pred.exists() or not table.exists():
        json_write(out/'prior_study_status.json',{'available':False,'reason':'Saved predictions or comparison absent'});return
    saved=pd.read_csv(pred,float_precision='round_trip');reported=pd.read_csv(table).set_index('candidate')
    dev=load_development(ROOT/'data/processed/classifier.csv').query('month >= 9').set_index('flight_id')
    result=[]
    for name,g in saved.groupby('candidate'):
        if not g.flight_id.is_unique or set(g.flight_id)!=set(dev.index):raise ValueError('Prior study ID mismatch')
        aligned=dev.loc[g.flight_id]
        if not np.array_equal(g.target.to_numpy(),aligned.delayed_15.to_numpy()):raise ValueError('Prior target mismatch')
        m=metrics(g.target,g.probability,float(g.threshold.iloc[0]))
        for metric in ['f1','roc_auc','average_precision']:
            if not np.isclose(m[metric],reported.loc[name,metric],atol=1e-12):raise ValueError('Prior saved metric mismatch')
        result.append({'candidate':name,**m})
    pd.DataFrame(result).to_csv(out/'prior_study_recomputed.csv',index=False)
    # Preserve published folds as reported, without claiming a fresh independent refit.
    folds=base/'internal_folds.csv'
    json_write(out/'prior_study_status.json',{'available':True,'saved_validation_metrics_recomputed':True,
        'internal_folds_available':folds.exists(),'prior_models_retrained':False,
        'interpretation':'Internal fold tables inspected as saved; only saved validation metrics independently recomputed'})


def uncertainty(out):
    from .uncertainty import bootstrap,wide_predictions
    import itertools
    historical_path=ROOT/'outputs/tables/prediction_predictions.csv'
    if historical_path.exists():
        saved=pd.read_csv(historical_path,float_precision='round_trip')
        saved=saved.loc[saved.split.eq('validation')].rename(columns={'model':'candidate','delayed_15':'target'})
        prior_path=ROOT/'outputs/experiments/modeling_study/validation_predictions.csv'
        if prior_path.exists():
            prior=pd.read_csv(prior_path,float_precision='round_trip')
            prior=prior.loc[prior.candidate.isin(['lr_cyclic_C1','forest_interactions_d14'])].copy()
            # Prior file has month but no date; align dates through saved historical IDs.
            dates=saved.loc[saved.candidate.eq('logistic_regression'),['flight_id','date']].set_index('flight_id').date
            prior['date']=prior.flight_id.map(dates)
            saved=pd.concat([saved,prior],ignore_index=True)
        names=[name for name in saved.candidate.unique() if name not in ['majority_baseline','logistic_regression']]
        summary,draws=bootstrap(wide_predictions(saved),[(name,'logistic_regression') for name in names],'whole_day',500,42)
        summary.to_csv(out/'historical_day_bootstrap.csv',index=False)
        draws.to_csv(out/'historical_day_bootstrap_draws.csv',index=False)
    else:json_write(out/'historical_bootstrap_unavailable.json',{'reason':'Saved historical predictions absent'})
    new=pd.read_csv(out/'development_predictions.csv',float_precision='round_trip')
    new=new.loc[new.fold.eq('d')&new.grid.eq('primary')&new.candidate.isin(list('ACFG'))]
    pairs=[(second,first) for first,second in itertools.combinations(list('ACFG'),2)]
    summary,draws=bootstrap(wide_predictions(new),pairs,'moving_7_day',500,42)
    summary.to_csv(out/'fold_d_block_bootstrap.csv',index=False)
    draws.to_csv(out/'fold_d_block_bootstrap_draws.csv',index=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['start','historical','experiment','uncertainty','render','check'])
    parser.add_argument('--run-id',required=True)
    args=parser.parse_args()
    if args.phase=='start':start(args.run_id);return
    if not args.run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.run_id):raise ValueError('Invalid run ID')
    out=BASE/args.run_id
    if args.phase=='check':
        result=verify_preservation(out)
        path=out/'preservation_final.json'
        if not path.exists():json_write(path,result)
        print('Preservation passed:',result['protected_file_count'],'files');return
    if not claim_phase(out,args.phase):return
    with threadpool_limits(limits=2):
        if args.phase=='historical':
            from .history import historical_verification
            historical_verification(ROOT,out);prior_study_verification(out)
        elif args.phase=='experiment':experiment(out)
        elif args.phase=='uncertainty':uncertainty(out)
        elif args.phase=='render':
            from .report import render
            render(out)
    result=verify_preservation(out)
    json_write(out/f'{args.phase}.done.json',{'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'preservation_unchanged':result['unchanged'],'protected_file_count':result['protected_file_count']})
    print(args.phase,'completed',flush=True)


if __name__=='__main__':main()
