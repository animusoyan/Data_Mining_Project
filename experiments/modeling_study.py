"""Controlled chronological study. Run python -m experiments.modeling_study once."""
from pathlib import Path
import csv
import hashlib
import importlib.metadata
import json
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.prepare import FEATURES
from src.predict import build_pipeline, candidates, evaluate, markdown_table, select_threshold

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/experiments/modeling_study'
FOLDS = [(3, 4, [5]), (4, 5, [6]), (5, 6, [7, 8])]


def load_development(path):
    """Exclude Nov-Dec rows before inspecting or converting their target values."""
    records = []
    with Path(path).open(newline='') as handle:
        reader = csv.DictReader(handle)
        expected = {'flight_id', 'date', 'split', 'delayed_15', *FEATURES}
        if set(reader.fieldnames) != expected:
            raise ValueError('Unexpected classifier schema')
        for row in reader:
            month = int(row['month'])
            if month > 10:
                continue  # Never inspect the target for these records.
            if row['split'] != ('train' if month <= 8 else 'validation'):
                raise ValueError('Invalid development split')
            records.append(row)
    data = pd.DataFrame(records)
    for name in ['flight_id', 'delayed_15', *[c for c in FEATURES if c not in ['carrier','origin','dest','is_weekend']]]:
        data[name] = pd.to_numeric(data[name], errors='raise')
    data['is_weekend'] = data.is_weekend.map({'True': True, 'False': False})
    data['date'] = pd.to_datetime(data.date, errors='raise')
    if (data.isna().any().any() or not data.delayed_15.isin([0,1]).all()
            or not data.flight_id.is_unique or not data.date.dt.year.eq(2013).all()
            or not data.month.eq(data.date.dt.month).all()):
        raise ValueError('Invalid development data')
    return data.sort_values(['date','flight_id']).reset_index(drop=True)


class ScheduleFeatures(TransformerMixin, BaseEstimator):
    """Deterministic functions of the approved predictors; no targets or aggregates."""
    def __init__(self, mode='cyclic'):
        self.mode = mode

    def fit(self, X, y=None):
        self.feature_names_in_ = np.array(FEATURES)
        return self

    def transform(self, X):
        result = X.loc[:, FEATURES].copy()
        if self.mode == 'keyed':
            result['flight'] = result.carrier.astype(str) + '_' + result.flight.astype(str)
            return result
        if self.mode not in ['cyclic', 'interactions']:
            raise ValueError('Unknown feature mode')
        for name, period, offset in [('day_of_year',365,1),('sched_dep_minutes',1440,0),('sched_arr_minutes',1440,0)]:
            angle = 2*np.pi*(result[name]-offset)/period
            result[name+'_sin'], result[name+'_cos'] = np.sin(angle), np.cos(angle)
        # Keep raw clock ramps as well as cycles; drop redundant linear seasonal trends.
        result = result.drop(columns=['month','day_of_year'])
        result['day_of_week'] = result.day_of_week.astype(str)
        if self.mode == 'interactions':
            result['flight'] = result.carrier.astype(str) + '_' + result.flight.astype(str)
            band = (result.sched_dep_minutes//240).astype(int).astype(str)
            result['carrier_origin'] = result.carrier.astype(str)+'_'+result.origin.astype(str)
            result['origin_band'] = result.origin.astype(str)+'_'+band
            result['carrier_band'] = result.carrier.astype(str)+'_'+band
        return result


def engineered_pipeline(estimator, mode):
    estimator = clone(estimator)
    if mode == 'original': return build_pipeline(estimator)
    cats = ['carrier','flight','origin','dest']
    nums = [c for c in FEATURES if c not in cats]
    if mode != 'keyed':
        cats += ['day_of_week']
        nums = [c for c in nums if c not in ['month','day_of_year','day_of_week']]
        nums += [name+suffix for name in ['day_of_year','sched_dep_minutes','sched_arr_minutes'] for suffix in ['_sin','_cos']]
    if mode == 'interactions': cats += ['carrier_origin','origin_band','carrier_band']
    pre = ColumnTransformer([
        ('numeric', Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]), nums),
        ('categorical',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),
            ('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=True))]),cats)],remainder='drop')
    return Pipeline([('features',ScheduleFeatures(mode)),('preprocess',pre),('model',estimator)])


def specifications():
    specs = []
    for family in ['logistic_regression','decision_tree','random_forest']:
        specs.append((family+'_original',family,'original',candidates()[family]))
    for mode in ['cyclic','keyed','interactions']:
        specs.append(('lr_'+mode+'_C1','logistic_regression',mode,
            LogisticRegression(C=1,solver='liblinear',max_iter=500,random_state=42)))
    for c in [.1,10]:
        specs.append(('lr_interactions_C'+str(c),'logistic_regression','interactions',
            LogisticRegression(C=c,solver='liblinear',max_iter=500,random_state=42)))
    for depth, leaf in [(6,200),(12,100),(16,50)]:
        specs.append((f'tree_interactions_d{depth}','decision_tree','interactions',
            DecisionTreeClassifier(max_depth=depth,min_samples_leaf=leaf,random_state=42)))
    for depth,leaf,features in [(14,30,'sqrt'),(20,10,'sqrt'),(18,30,.1)]:
        specs.append((f'forest_interactions_d{depth}','random_forest','interactions',
            RandomForestClassifier(n_estimators=100,max_depth=depth,min_samples_leaf=leaf,
                max_features=features,n_jobs=2,random_state=42)))
    return specs


def probability(model, data):
    return model.predict_proba(data.loc[:,FEATURES])[:,list(model.classes_).index(1)]


def metrics(y, p, threshold):
    result = evaluate(y,p,threshold)
    result['accuracy'] = (result['tp']+result['tn'])/result['n']
    specificity = result['tn']/(result['tn']+result['fp']) if result['tn']+result['fp'] else np.nan
    result['balanced_accuracy'] = (result['recall']+specificity)/2 if np.isfinite(specificity) else np.nan
    result['average_precision'] = average_precision_score(y,p) if np.sum(y)>0 else np.nan
    result['brier'] = brier_score_loss(y,p)
    result['log_loss'] = log_loss(y,np.clip(p,1e-12,1-1e-12),labels=[0,1])
    return result


def protected_hashes():
    files = [ROOT/'reports/prediction_findings.md',ROOT/'src/predict.py',ROOT/'src/prepare.py',ROOT/'requirements.txt']
    files += list((ROOT/'outputs/models').glob('prediction*'))+[ROOT/'outputs/models/flight_delay_pipeline.joblib']
    files += list((ROOT/'outputs/tables').glob('prediction*'))
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def calibration_bins(y,p,name):
    data = pd.DataFrame({'y':np.asarray(y),'p':p})
    data['bin'] = pd.cut(data.p,np.linspace(0,1,11),include_lowest=True)
    bins = data.groupby('bin',observed=True).agg(n=('y','size'),predicted=('p','mean'),observed=('y','mean')).reset_index()
    bins['bin'] = bins.bin.astype(str)
    bins['model'] = name
    return bins


def logit(p):
    p = np.clip(p,1e-6,1-1e-6)
    return np.log(p/(1-p)).reshape(-1,1)


def run():
    if OUT.exists(): raise FileExistsError('Study artifacts already exist; inspect rather than repeat the search.')
    OUT.mkdir(parents=True)
    hashes = protected_hashes()
    specs = specifications()
    protocol = {'folds':[{'fit_through':a,'threshold_month':b,'assessment_months':c} for a,b,c in FOLDS],
        'candidates':[{'name':n,'family':f,'encoding':m,'parameters':e.get_params()} for n,f,m,e in specs],
        'ranking':'mean assessment F1 across three folds, then mean ROC-AUC, then name',
        'external_shortlist':'three original models, best internal per family, plus LR C1 encoding ablations',
        'calibration':'internal winner: fit Jan-Jun, sigmoid on July, threshold August, assess Sep-Oct; matched raw control',
        'importance':'internal winner, grouped permutation on Sep-Oct only, three repeats, ranking metrics; no refit',
        'threshold_grid':'0.10 to 0.50 step 0.025; always-positive reference separately',
        'test_policy':'Nov-Dec excluded before target parsing; no new holdout assessment',
        'versions':{'python':platform.python_version(),'sklearn':sklearn.__version__,'numpy':np.__version__,'pandas':pd.__version__},
        'original_hashes':hashes}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (OUT/'environment.json').write_text(json.dumps({n:importlib.metadata.version(n) for n in
        ['pandas','matplotlib','ipykernel','notebook','scikit-learn','numpy','scipy','joblib','nbformat','nbclient','ipython']},indent=2)+'\n')
    data = load_development(ROOT/'data/processed/classifier.csv')
    train, validation = data.loc[data.month<=8],data.loc[data.month.between(9,10)]
    (OUT/'development_sha256.txt').write_text(hashlib.sha256(data.to_csv(index=False).encode()).hexdigest()+'\n')
    rows=[]
    for name,family,mode,estimator in specs:
        print('Internal',name,flush=True)
        for index,(end,tune_month,assessment) in enumerate(FOLDS,1):
            fit = train.loc[train.month<=end]
            tune = train.loc[train.month.eq(tune_month)]
            assess = train.loc[train.month.isin(assessment)]
            model = engineered_pipeline(estimator,mode)
            model.fit(fit[FEATURES],fit.delayed_15)
            threshold,_=select_threshold(tune.delayed_15,probability(model,tune))
            rows.append({'candidate':name,'family':family,'mode':mode,'fold':index,'fit_through':end,
                'threshold_month':tune_month,'assessment_months':str(assessment),'threshold':threshold,
                **metrics(assess.delayed_15,probability(model,assess),threshold)})
        pd.DataFrame(rows).to_csv(OUT/'internal_folds.csv',index=False)
    internal = pd.DataFrame(rows)
    summary=internal.groupby(['candidate','family','mode']).agg(mean_f1=('f1','mean'),min_f1=('f1','min'),
        std_f1=('f1','std'),mean_roc_auc=('roc_auc','mean'),mean_ap=('average_precision','mean')).reset_index()
    summary=summary.sort_values(['mean_f1','mean_roc_auc','candidate'],ascending=[False,False,True])
    summary.to_csv(OUT/'internal_summary.csv',index=False)
    winners=summary.drop_duplicates('family').candidate.tolist()
    champion=summary.iloc[0].candidate
    shortlist=sorted(set(winners+[f+'_original' for f in ['logistic_regression','decision_tree','random_forest']]+
        ['lr_cyclic_C1','lr_keyed_C1','lr_interactions_C1']))
    # Freeze the shortlist and internal champion before any validation prediction.
    (OUT/'shortlist.json').write_text(json.dumps({'candidates':shortlist,'internal_champion':champion},indent=2)+'\n')
    external=[]; monthly=[]; bins=[]; prediction_rows=[]; final_models={}; thresholds={}
    original_selection=json.loads((ROOT/'outputs/models/prediction_selection.json').read_text())
    for name,family,mode,estimator in specs:
        if name not in shortlist: continue
        print('Validation',name,flush=True)
        if mode=='original':
            model=joblib.load(ROOT/f'outputs/models/prediction_{family}.joblib')
            threshold=original_selection['thresholds'][family]
        else:
            model=engineered_pipeline(estimator,mode)
            model.fit(train[FEATURES],train.delayed_15)
            threshold,_=select_threshold(validation.delayed_15,probability(model,validation))
        p=probability(model,validation)
        thresholds[name]=threshold; final_models[name]=model
        external.append({'candidate':name,'threshold':threshold,**metrics(validation.delayed_15,p,threshold)})
        for month,group in validation.groupby('month'):
            positions=validation.index.get_indexer(group.index)
            monthly.append({'candidate':name,'month':month,'threshold':threshold,**metrics(group.delayed_15,p[positions],threshold)})
        bins.append(calibration_bins(validation.delayed_15,p,name))
        prediction_rows.append(pd.DataFrame({'flight_id':validation.flight_id,'month':validation.month,'target':validation.delayed_15,'candidate':name,'probability':p,'threshold':threshold}))
        joblib.dump(model,OUT/f'{name}.joblib')
    for name,estimator in [('majority',DummyClassifier(strategy='most_frequent')),('always_positive',DummyClassifier(strategy='constant',constant=1))]:
        model=build_pipeline(estimator).fit(train[FEATURES],train.delayed_15)
        p=probability(model,validation)
        external.append({'candidate':name,'threshold':.5,**metrics(validation.delayed_15,p,.5)})
        for month,group in validation.groupby('month'):
            positions=validation.index.get_indexer(group.index)
            monthly.append({'candidate':name,'month':month,'threshold':.5,**metrics(group.delayed_15,p[positions],.5)})
    pd.DataFrame(external).to_csv(OUT/'validation_metrics.csv',index=False)
    pd.DataFrame(monthly).to_csv(OUT/'validation_monthly.csv',index=False)
    pd.concat(prediction_rows,ignore_index=True).to_csv(OUT/'validation_predictions.csv',index=False)
    # Calibration uses a separate base and future calibration/threshold periods, never in-sample scores.
    name,family,mode,estimator=next(s for s in specs if s[0]==champion)
    base=engineered_pipeline(estimator,mode).fit(train.loc[train.month<=6,FEATURES],train.loc[train.month<=6,'delayed_15'])
    july,august=train.loc[train.month.eq(7)],train.loc[train.month.eq(8)]
    sigmoid=LogisticRegression(C=1e6,solver='lbfgs',max_iter=500).fit(logit(probability(base,july)),july.delayed_15)
    calibration=[]
    for method in ['raw','sigmoid']:
        def transform(p): return sigmoid.predict_proba(logit(p))[:,1] if method=='sigmoid' else p
        threshold,_=select_threshold(august.delayed_15,transform(probability(base,august)))
        p=transform(probability(base,validation))
        calibration.append({'method':method,'assessment':'Sep-Oct','threshold':threshold,**metrics(validation.delayed_15,p,threshold)})
        for month,group in validation.groupby('month'):
            calibration.append({'method':method,'assessment':str(month),'threshold':threshold,**metrics(group.delayed_15,transform(probability(base,group)),threshold)})
        bins.append(calibration_bins(validation.delayed_15,p,'calibration_'+method))
    pd.DataFrame(calibration).to_csv(OUT/'calibration_metrics.csv',index=False)
    pd.concat(bins,ignore_index=True).to_csv(OUT/'reliability_bins.csv',index=False)
    joblib.dump({'base':base,'sigmoid':sigmoid,'base_fit':'Jan-Jun','calibration_fit':'July'},OUT/'calibration_experiment.joblib')
    complete_diagnostics()
    print('Study complete. Original artifacts unchanged.',flush=True)


def complete_diagnostics():
    protocol=json.loads((OUT/'protocol.json').read_text())
    hashes=protocol['original_hashes']
    champion=json.loads((OUT/'shortlist.json').read_text())['internal_champion']
    data=load_development(ROOT/'data/processed/classifier.csv')
    validation=data.loc[data.month.between(9,10)]
    external=pd.read_csv(OUT/'validation_metrics.csv').set_index('candidate')
    thresholds=external.threshold.to_dict()
    # Grouped permutation: preserve within-group dependency, deterministic and validation-only.
    groups={'season':['month','day_of_year'],'weekday':['day_of_week','is_weekend'],
        'clocks':['sched_dep_minutes','sched_arr_minutes'],'airline_flight':['carrier','flight'],
        'airports':['origin','dest'],'distance':['distance']}
    reference=joblib.load(OUT/f'{champion}.joblib'); base_metrics=metrics(validation.delayed_15,probability(reference,validation),thresholds[champion])
    rng=np.random.default_rng(42); importance=[]
    for group,columns in groups.items():
        for repeat in range(3):
            altered=validation[FEATURES].copy()
            order=rng.permutation(len(altered))
            altered.loc[:,columns]=altered.iloc[order][columns].to_numpy()
            permuted=metrics(validation.delayed_15,probability(reference,altered),thresholds[champion])
            importance.append({'group':group,'repeat':repeat,'auc_drop':base_metrics['roc_auc']-permuted['roc_auc'],
                'ap_drop':base_metrics['average_precision']-permuted['average_precision'],
                'f1_drop':base_metrics['f1']-permuted['f1']})
    pd.DataFrame(importance).to_csv(OUT/'group_importance.csv',index=False)
    after=protected_hashes()
    if hashes!=after: raise AssertionError('Historical baseline changed')
    (OUT/'preservation_check.json').write_text(json.dumps({'all_original_hashes_match':True,'files':hashes},indent=2)+'\n')

    # Threshold transfer is reported separately from validation-retuned scores.
    predictions=pd.read_csv(OUT/'validation_predictions.csv')
    internal=pd.read_csv(OUT/'internal_folds.csv')
    transfer=[]
    for name,group in predictions.groupby('candidate'):
        threshold=float(internal.loc[internal.candidate.eq(name),'threshold'].median())
        transfer.append({'candidate':name,'internal_median_threshold':threshold,
            **metrics(group.target,group.probability,threshold)})
    pd.DataFrame(transfer).to_csv(OUT/'threshold_transfer.csv',index=False)
    references=[]
    for end in [6,8]:
        prior=float(data.loc[data.month<=end,'delayed_15'].mean())
        references.append({'reference':f'training_prior_Jan_to_{end}','training_prevalence':prior,
            **metrics(validation.delayed_15,np.full(len(validation),prior),.5)})
    pd.DataFrame(references).to_csv(OUT/'probability_references.csv',index=False)
    data.groupby('month').delayed_15.agg(n='size',positive_count='sum',prevalence='mean').to_csv(OUT/'development_monthly_coverage.csv')
    encoding_coverage=[]
    for mode in ['original','keyed','cyclic','interactions']:
        features=data[FEATURES] if mode=='original' else ScheduleFeatures(mode).fit_transform(data)
        categories=['carrier','flight','origin','dest']
        if mode in ['cyclic','interactions']: categories+=['day_of_week']
        if mode=='interactions': categories+=['carrier_origin','origin_band','carrier_band']
        for column in categories:
            known=set(features.loc[data.month<=8,column])
            future=features.loc[data.month.between(9,10),column]
            encoding_coverage.append({'mode':mode,'feature':column,'training_categories':len(known),
                'validation_unseen_n':int((~future.isin(known)).sum()),
                'validation_unseen_share':float((~future.isin(known)).mean())})
    pd.DataFrame(encoding_coverage).to_csv(OUT/'encoding_coverage.csv',index=False)




if __name__=='__main__':
    # Import canonical module so custom transformers remain loadable outside this process.
    from experiments.modeling_study import run as execute
    execute()
