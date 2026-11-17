"""Historical verification route: full-year reconstruction and SAVED test scores only."""
import json
import hashlib

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from .core import (FEATURES, build, clock, compare_by_id, fit_record, load_development,
                   metrics, probability, reconstruct, reliability)


def verify_saved_rows(scores,classifier,selection):
    expected=classifier.loc[classifier.split.isin(['validation','test'])].set_index('flight_id')
    rows=[]
    for name,group in scores.groupby('model'):
        if not group.flight_id.is_unique or set(group.flight_id)!=set(expected.index):
            raise ValueError(f'Saved scores do not cover exact holdout IDs: {name}')
        group=group.set_index('flight_id').sort_index()
        labels=expected.loc[group.index]
        for col in ['split','month','delayed_15']:
            if not group[col].eq(labels[col]).all(): raise ValueError(f'Saved {col} mismatch')
        if not pd.to_datetime(group.date).eq(pd.to_datetime(labels.date)).all(): raise ValueError('Saved date mismatch')
        if not group.threshold.eq(selection['thresholds'][name]).all(): raise ValueError('Stored threshold mismatch')
        if not group.prediction.eq(group.probability.ge(group.threshold).astype(int)).all():
            raise ValueError('Prediction does not use stored threshold')
        for split,g in group.groupby('split'):
            rows.append({'model':name,'split':split,'threshold':float(g.threshold.iloc[0]),
                         **metrics(g.delayed_15,g.probability,g.threshold.iloc[0])})
    return pd.DataFrame(rows)


def audit_data(root,out):
    raw_path=root/'data/raw/flights.csv'
    checksum=hashlib.sha256(raw_path.read_bytes()).hexdigest()
    manifest=json.loads((root/'data/raw/source.json').read_text())
    if checksum!=manifest['sha256']:raise ValueError('Cached raw checksum disagrees with source manifest')
    raw=pd.read_csv(raw_path)
    expected=reconstruct(raw)
    clean=pd.read_csv(root/'data/processed/flights_clean.csv')
    classifier=pd.read_csv(root/'data/processed/classifier.csv')
    compare_by_id(expected,clean)
    expected_classifier=expected.loc[expected.label_observed,['flight_id','date','split',*FEATURES,'delayed_15']]
    compare_by_id(expected_classifier,classifier)
    unknown_path=root/'data/processed/missing_departure_delay.csv'
    if unknown_path.exists(): compare_by_id(expected.loc[~expected.label_observed],pd.read_csv(unknown_path))
    pd.DataFrame({'column':expected.columns,'missing_count':expected.isna().sum().to_numpy()}).to_csv(out/'data_missingness.csv',index=False)
    missing=expected.groupby('month').agg(n=('flight_id','size'),known=('label_observed','sum'))
    missing['missing_count']=missing.n-missing.known;missing['missing_share']=missing.missing_count/missing.n
    missing.to_csv(out/'historical_missing_by_month.csv')
    keys=[]
    for columns in [['date','origin','carrier','flight'],['date','origin','carrier','flight','dest','sched_dep_time']]:
        duplicates=expected.duplicated(columns,keep=False)
        keys.append({'key':' + '.join(columns),'duplicate_rows':int(duplicates.sum()),
                     'duplicate_groups':int(expected.loc[duplicates].groupby(columns).ngroups)})
    pd.DataFrame(keys).to_csv(out/'natural_keys.csv',index=False)
    consistency=[]
    for actual,scheduled,delay in [('dep_time','sched_dep_time','dep_delay'),('arr_time','sched_arr_time','arr_delay')]:
        observed=expected[[actual,scheduled,delay]].notna().all(axis=1)
        residual=(clock(expected.loc[observed,actual])-clock(expected.loc[observed,scheduled])-expected.loc[observed,delay]).mod(1440)
        bad=residual.ne(0)
        consistency.append({'clock':actual,'comparable_n':int(observed.sum()),'inconsistent_modulo_1440':int(bad.sum())})
        if bad.any(): expected.loc[observed].loc[bad,['flight_id',actual,scheduled,delay]].to_csv(out/f'clock_discrepancies_{actual}.csv',index=False)
    pd.DataFrame(consistency).to_csv(out/'local_clock_consistency.csv',index=False)
    boundary=expected.loc[expected.dep_delay.isin([14,15,16])].groupby('dep_delay').agg(n=('flight_id','size'),label_min=('delayed_15','min'),label_max=('delayed_15','max'))
    boundary.to_csv(out/'label_boundaries.csv')
    summary={'raw_sha256_matches_manifest':True,'raw_sha256':checksum,'all_processed_comparisons':'exact by flight_id including missing masks',
        'rows':len(expected),'classification_rows':len(classifier),'unknown_labels':int((~expected.label_observed).sum()),
        'duplicate_source_content_rows':int(raw.drop(columns='rownames').duplicated().sum()),
        'classifier_missing_inputs':classifier[FEATURES].isna().sum().to_dict(),
        'negative_departure_delays':int(expected.dep_delay.lt(0).sum()),
        'departure_delay_min':float(expected.dep_delay.min()),'departure_delay_max':float(expected.dep_delay.max()),
        'departure_delay_over_1440':int(expected.dep_delay.gt(1440).sum()),
        'missing_departure_clock_with_observed_delay':int((expected.dep_time.isna()&expected.dep_delay.notna()).sum())}
    (out/'data_audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    return classifier


def historical_verification(root,out):
    classifier=audit_data(root,out)
    pred_path=root/'outputs/tables/prediction_predictions.csv'
    selection_path=root/'outputs/models/prediction_selection.json'
    status={'new_Nov_Dec_inference':False,'missing_artifacts':[]}
    if not pred_path.exists() or not selection_path.exists():
        status['missing_artifacts']=[str(p.relative_to(root)) for p in [pred_path,selection_path] if not p.exists()]
        (out/'history_status.json').write_text(json.dumps(status,indent=2)+'\n')
        return
    selection=json.loads(selection_path.read_text())
    scores=pd.read_csv(pred_path,float_precision='round_trip')
    recomputed=verify_saved_rows(scores,classifier,selection)
    recomputed.to_csv(out/'historical_recomputed.csv',index=False)
    stored_path=root/'outputs/tables/prediction_comparison.csv'
    differences=[]
    if stored_path.exists():
        saved=pd.read_csv(stored_path)
        for row in recomputed.to_dict('records'):
            old=saved.loc[saved.model.eq(row['model'])&saved.split.eq(row['split'])].iloc[0]
            for metric in ['n','positive_count','tn','fp','fn','tp','precision','recall','f1','roc_auc']:
                delta=float(row[metric]-old[metric])
                if not (pd.isna(row[metric]) and pd.isna(old[metric])) and not np.isclose(row[metric],old[metric],atol=1e-12):
                    raise ValueError(f'Stored historical metric mismatch: {metric}')
                differences.append({'model':row['model'],'split':row['split'],'metric':metric,'difference':delta})
    pd.DataFrame(differences).to_csv(out/'historical_metric_differences.csv',index=False)
    monthly=[];bins=[]
    for (name,month),g in scores.groupby(['model','month']):
        monthly.append({'model':name,'month':month,'origin':'SAVED historical predictions',**metrics(g.delayed_15,g.probability,g.threshold.iloc[0])})
        bins.append(reliability(g.delayed_15,g.probability).assign(model=name,month=month))
    pd.DataFrame(monthly).to_csv(out/'historical_saved_monthly.csv',index=False)
    pd.concat(bins).to_csv(out/'historical_saved_reliability.csv',index=False)
    # Entirely separate development route for any new calls to predict_proba.
    dev=load_development(root/'data/processed/classifier.csv')
    train=dev.loc[dev.month<=8];val=dev.loc[dev.month.between(9,10)]
    diagnostics=[];pipeline_checks=[];categories=[];dev_monthly=[];dev_bins=[]
    for name in selection['thresholds']:
        path=root/f'outputs/models/prediction_{name}.joblib'
        if not path.exists(): status['missing_artifacts'].append(str(path.relative_to(root)));continue
        model=joblib.load(path)
        threshold=selection['thresholds'][name]
        for label,frame in [('train_optimistic',train),('validation_reused',val)]:
            p=probability(model,frame)
            pd.DataFrame({'flight_id':frame.flight_id,'date':frame.date,'month':frame.month,
                'target':frame.delayed_15,'probability':p,'threshold':threshold}).to_csv(out/f'history_dev_{name}_{label}.csv',index=False)
            diagnostics.append({'model':name,'period':label,**metrics(frame.delayed_15,p,threshold)})
            for month,g in frame.groupby('month'):
                positions=frame.index.get_indexer(g.index)
                dev_monthly.append({'model':name,'month':month,'score_scope':label,**metrics(g.delayed_15,p[positions],threshold)})
                dev_bins.append(reliability(g.delayed_15,p[positions]).assign(model=name,month=month,score_scope=label))
            if label=='validation_reused':
                old=scores.loc[scores.model.eq(name)&scores.split.eq('validation')].set_index('flight_id').loc[frame.flight_id]
                pipeline_checks.append({'model':name,'max_abs_probability_difference':float(np.max(np.abs(p-old.probability.to_numpy()))),
                    'mean_abs_probability_difference':float(np.mean(np.abs(p-old.probability.to_numpy()))),
                    'input_names':str(list(model.feature_names_in_)),
                    'encoded_dimension':int(model.named_steps['preprocess'].transform(frame.iloc[:1][FEATURES]).shape[1]),
                    'iterations':str(getattr(model.named_steps['model'],'n_iter_',None))})
        pre=model.named_steps['preprocess']
        numeric=pre.named_transformers_['numeric'];scaler=numeric.named_steps['scale']
        if not np.allclose(scaler.mean_,train[next(cols for name,transform,cols in pre.transformers_ if name=='numeric')].mean(),atol=1e-12):
            raise ValueError('Historical scaler mean differs from training only')
        if scaler.n_samples_seen_!=len(train): raise ValueError('Historical scaler training size mismatch')
        encoder=pre.named_transformers_['categorical'].named_steps['encode']
        for col,known in zip(next(cols for name,transform,cols in pre.transformers_ if name=='categorical'),encoder.categories_):
            if set(known)!=set(train[col]):raise ValueError('Category vocabulary differs from training')
            for month,g in dev.groupby('month'):
                categories.append({'model':name,'column':col,'month':month,'fit_categories':len(known),
                    'unknown_n':int((~g[col].isin(known)).sum()),'unknown_share':float((~g[col].isin(known)).mean())})
        altered=val.iloc[:2].copy();altered['carrier']='UNSEEN';altered['flight']=-999;altered['origin']='NEW';altered['dest']='NEW'
        if not np.isfinite(probability(model,altered)).all():raise ValueError('Unknown category inference not finite')
    pd.DataFrame(diagnostics).to_csv(out/'historical_train_validation_ranking.csv',index=False)
    pd.DataFrame(pipeline_checks).to_csv(out/'historical_pipeline_checks.csv',index=False)
    pd.DataFrame(categories).to_csv(out/'historical_category_coverage.csv',index=False)
    pd.DataFrame(dev_monthly).to_csv(out/'historical_development_monthly.csv',index=False)
    pd.concat(dev_bins).to_csv(out/'historical_development_reliability.csv',index=False)
    fresh=build('A');record=fit_record(fresh,train)
    p=probability(fresh,val)
    old=scores.loc[scores.model.eq('logistic_regression')&scores.split.eq('validation')].set_index('flight_id').loc[val.flight_id].probability.to_numpy()
    record.update({'max_abs_probability_difference':float(abs(p-old).max()),'mean_abs_probability_difference':float(abs(p-old).mean()),
        'auc_difference':metrics(val.delayed_15,p,.325)['roc_auc']-metrics(val.delayed_15,old,.325)['roc_auc']})
    (out/'fresh_original_lr.json').write_text(json.dumps(record,indent=2)+'\n')
    prior=float(train.delayed_15.mean())
    controls=[]
    for month,g in dev.groupby('month'):
        for name,value in [('majority',0.),('always_positive',1.),('training_prior',prior)]:
            controls.append({'model':name,'month':month,**metrics(g.delayed_15,np.full(len(g),value),.5)})
    pd.DataFrame(controls).to_csv(out/'historical_development_controls.csv',index=False)
    status.update({'saved_metrics_verified':True,'pipelines_checked':len(pipeline_checks)})
    (out/'history_status.json').write_text(json.dumps(status,indent=2)+'\n')


def covariate_audit(dev,out):
    train=dev.loc[dev.month<=8];val=dev.loc[dev.month.between(9,10)]
    rows=[]
    cats=['carrier','flight','origin','dest','day_of_week','is_weekend']
    for col in FEATURES:
        if col in cats:
            a=train[col].value_counts(normalize=True);b=val[col].value_counts(normalize=True)
            keys=a.index.union(b.index)
            distance=float(.5*np.abs(a.reindex(keys,fill_value=0)-b.reindex(keys,fill_value=0)).sum())
            rows.append({'feature':col,'distance_type':'TV','distance':distance,'train_min':None,'train_max':None,'validation_min':None,'validation_max':None})
        else:
            rows.append({'feature':col,'distance_type':'KS','distance':float(ks_2samp(train[col],val[col]).statistic),
                'train_min':float(train[col].min()),'train_max':float(train[col].max()),
                'validation_min':float(val[col].min()),'validation_max':float(val[col].max())})
    pd.DataFrame(rows).to_csv(out/'covariate_distances.csv',index=False)
    groupcols=['origin','carrier','dep_band']
    data=dev.assign(dep_band=(dev.sched_dep_minutes//240).astype(int))
    table=[]
    for label,frame in [('train',data.loc[data.month<=8]),('validation',data.loc[data.month.between(9,10)])]:
        summary=frame.groupby(groupcols).delayed_15.agg(n='size',rate='mean')
        summary.columns=[label+'_'+col for col in summary.columns]
        table.append(summary)
    strata=table[0].join(table[1],how='outer')
    common=strata.dropna()
    train_rate=float((common.train_n*common.train_rate).sum()/common.train_n.sum())
    expected=float((common.validation_n*common.train_rate).sum()/common.validation_n.sum())
    observed=float((common.validation_n*common.validation_rate).sum()/common.validation_n.sum())
    result={'strata':len(strata),'common_strata':len(common),'train_support_coverage':float(common.train_n.sum()/len(train)),
        'validation_support_coverage':float(common.validation_n.sum()/len(val)),
        'all_training_prevalence':float(train.delayed_15.mean()),'all_validation_prevalence':float(val.delayed_15.mean()),
        'common_training_prevalence':train_rate,'expected_validation_mix_training_rates':expected,
        'common_observed_validation_prevalence':observed,'mix_component_common_support':expected-train_rate,
        'residual_within_strata':observed-expected,'common_total_change':observed-train_rate}
    strata.to_csv(out/'prevalence_strata.csv')
    (out/'prevalence_decomposition.json').write_text(json.dumps(result,indent=2)+'\n')
