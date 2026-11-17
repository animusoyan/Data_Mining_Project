"""Independent data contracts, fixed candidate specifications and metric functions."""
from pathlib import Path
import csv
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, SplineTransformer, StandardScaler

from src.prepare import FEATURES

FOLDS = {'a': (3,4,[5]), 'b': (4,5,[6]), 'c': (5,6,[7,8]), 'd': (7,8,[9,10])}
GRID = np.round(np.arange(.1,.501,.025),3)
DIAGNOSTIC_GRID = np.r_[np.round(np.arange(0,1.001,.01),2),1.01]
SEED = 42


def clock(values):
    values = pd.to_numeric(values)
    valid = values.isna() | (values.eq(values.round()) & ((values.between(0,2359) & values.mod(100).lt(60)) | values.eq(2400)))
    if not valid.all(): raise ValueError('Invalid HHMM')
    return (values.floordiv(100)*60+values.mod(100)).mod(1440)


def target(delay):
    return delay.ge(15).astype('Int64').where(delay.notna())


def reconstruct(raw):
    """No call to prepare(): independently rederive every engineered/export field."""
    out=raw.rename(columns={'rownames':'flight_id'}).copy()
    out['date']=pd.to_datetime(out[['year','month','day']])
    for col in ['dep_time','arr_time','sched_dep_time','sched_arr_time']: clock(out[col])
    out['sched_dep_minutes']=clock(out.sched_dep_time)
    out['sched_arr_minutes']=clock(out.sched_arr_time)
    out['day_of_week']=out.date.dt.dayofweek
    out['day_of_year']=out.date.dt.dayofyear
    out['is_weekend']=out.day_of_week.isin([5,6])
    out['route']=out.origin+'-'+out.dest
    out['label_observed']=out.dep_delay.notna()
    out['delayed_15']=target(out.dep_delay)
    out['split']=np.select([out.month<=8,out.month<=10],['train','validation'],default='test')
    return out


def compare_by_id(expected,actual):
    """Exact IDs, fields and values; normalize CSV dtypes, never tolerate lost unknowns."""
    if not expected.flight_id.is_unique or not actual.flight_id.is_unique:
        raise ValueError('Duplicate IDs')
    if set(expected.columns)!=set(actual.columns) or set(expected.flight_id)!=set(actual.flight_id):
        raise ValueError('Schema or exact key coverage mismatch')
    a=expected.set_index('flight_id').sort_index(); b=actual.set_index('flight_id').sort_index()
    for col in a:
        x,y=a[col],b[col]
        if col=='date': x,y=pd.to_datetime(x),pd.to_datetime(y)
        if not x.isna().eq(y.isna()).all(): raise ValueError(f'Missingness mismatch: {col}')
        known=x.notna()
        if not x[known].eq(y[known]).all(): raise ValueError(f'Value mismatch: {col}')


def load_development(path):
    """CSV records are filtered before accessing/converting Nov-Dec target strings."""
    records=[]
    with Path(path).open(newline='') as handle:
        reader=csv.DictReader(handle)
        if set(reader.fieldnames)!={'flight_id','date','split','delayed_15',*FEATURES}:
            raise ValueError('Unexpected classifier schema')
        for row in reader:
            month=int(row['month'])
            if month>10: continue
            records.append(row)
    data=pd.DataFrame(records)
    if data.empty: raise ValueError('Empty development data')
    for col in ['flight_id','delayed_15',*[c for c in FEATURES if c not in ['carrier','origin','dest','is_weekend']]]:
        data[col]=pd.to_numeric(data[col],errors='raise')
    data['date']=pd.to_datetime(data.date,errors='raise')
    data['is_weekend']=data.is_weekend.map({'True':True,'False':False})
    if data.isna().any().any(): raise ValueError('Unexpected missing development inputs or labels; no silent imputation')
    if not data.delayed_15.isin([0,1]).all() or not data.flight_id.is_unique: raise ValueError('Invalid labels/IDs')
    if not data.date.dt.year.eq(2013).all() or data.date.max()>pd.Timestamp('2013-10-31'):
        raise ValueError('Development date boundary violated')
    if not data.date.dt.month.eq(data.month).all(): raise ValueError('Date/month mismatch')
    split=np.where(data.month<=8,'train','validation')
    if not data.split.eq(split).all(): raise ValueError('Development split mismatch')
    return data.sort_values(['date','flight_id']).reset_index(drop=True)


def partitions(data,fold):
    end,tune,months=FOLDS[fold]
    fit=data.loc[data.month<=end]; tuning=data.loc[data.month.eq(tune)]; assess=data.loc[data.month.isin(months)]
    if any(x.empty for x in [fit,tuning,assess]): raise ValueError('Empty fold period')
    if not fit.date.max()<tuning.date.min() or not tuning.date.max()<assess.date.min():
        raise ValueError('Require strict fit < threshold < assessment chronology')
    if assess.date.max()>pd.Timestamp('2013-10-31'): raise ValueError('Assessment outside development')
    if set(fit.flight_id)&set(tuning.flight_id) or set(tuning.flight_id)&set(assess.flight_id):
        raise ValueError('Overlapping fold IDs')
    return fit,tuning,assess


def roles(candidate):
    cats=['carrier','flight','origin','dest']
    nums=[c for c in FEATURES if c not in cats]
    if candidate in ['B','E','F','G']: nums=[c for c in nums if c not in ['month','day_of_year']]
    if candidate in ['C','F','G']: cats.remove('flight')
    if candidate in ['D','F','G']:
        nums.remove('day_of_week'); cats.append('day_of_week')
    return nums,cats


class ApprovedInput(TransformerMixin,BaseEstimator):
    """Ignore all metadata/outcomes; fail on missing approved inputs."""
    def __init__(self,columns): self.columns=columns
    def fit(self,X,y=None):
        self.transform(X)
        return self
    def transform(self,X):
        result=X.loc[:,self.columns].copy()
        if result.isna().any().any(): raise ValueError('Missing approved inputs')
        return result


class CardinalityGuard(TransformerMixin,BaseEstimator):
    def __init__(self,n_categories,max_bins=255):
        self.n_categories=n_categories; self.max_bins=max_bins
    def fit(self,X,y=None):
        X=np.asarray(X,dtype=float)
        self.cardinalities_=[len(np.unique(X[np.isfinite(X[:,i]),i])) for i in range(self.n_categories)]
        if max(self.cardinalities_,default=0)>self.max_bins: raise ValueError('HGB categorical max_bins exceeded')
        return self
    def transform(self,X): return np.asarray(X,dtype=float)


def build(candidate):
    if candidate not in 'ABCDEFG' or len(candidate)!=1: raise ValueError('Only seven fixed candidates allowed')
    nums,cats=roles(candidate)
    approved=[c for c in FEATURES if c in nums+cats]
    if candidate=='G':
        pre=ColumnTransformer([('categorical',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=np.nan),cats),
                               ('numeric','passthrough',nums)],remainder='drop',sparse_threshold=0)
        estimator=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,min_samples_leaf=100,
            learning_rate=.1,l2_regularization=10,early_stopping=False,random_state=42,
            categorical_features=list(range(len(cats))))
        return Pipeline([('inputs',ApprovedInput(approved)),('preprocess',pre),
                         ('cardinality',CardinalityGuard(len(cats),estimator.max_bins)),('model',estimator)])
    transforms=[]
    if candidate=='E':
        nums.remove('sched_dep_minutes')
        transforms.append(('departure_spline',Pipeline([
            ('spline',SplineTransformer(n_knots=5,knots='uniform',degree=3,include_bias=False,extrapolation='linear')),
            ('scale',StandardScaler())]),['sched_dep_minutes']))
    # Match original numeric / categorical column order for A and its single ablations.
    transforms=[('numeric',StandardScaler(),nums),('categorical',OneHotEncoder(handle_unknown='ignore',sparse_output=True),cats)]+transforms
    pre=ColumnTransformer(transforms,remainder='drop')
    model=LogisticRegression(C=1,solver='liblinear',max_iter=1000,random_state=42)
    return Pipeline([('inputs',ApprovedInput(approved)),('preprocess',pre),('model',model)])


def fit_record(model,frame):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model.fit(frame,frame.delayed_15)
    estimator=model.named_steps['model']
    record={'warnings':[str(w.message) for w in caught],
            'convergence_warnings':sum(issubclass(w.category,ConvergenceWarning) for w in caught),
            'iterations':np.asarray(getattr(estimator,'n_iter_',[])).tolist(),
            'encoded_dimension':int(estimator.n_features_in_),'n_fit':len(frame)}
    if 'cardinality' in model.named_steps:
        record['categorical_cardinalities']=model.named_steps['cardinality'].cardinalities_
        record['max_bins']=estimator.max_bins
    return record


def probability(model,frame):
    # ApprovedInput for new candidates; explicit allowlist for historical models.
    if 'date' in frame and pd.to_datetime(frame.date).max()>pd.Timestamp('2013-10-31'):
        raise ValueError('New inference outside development is forbidden')
    return model.predict_proba(frame[FEATURES])[:,list(model.classes_).index(1)]


def metrics(y,p,t=.5):
    y=np.asarray(y);p=np.asarray(p,dtype=float)
    if len(y)!=len(p) or not len(y) or not np.isin(y,[0,1]).all() or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Metrics require aligned observed binary labels and finite probabilities')
    y=y.astype(int)
    pred=p>=t
    tp=int(((y==1)&pred).sum());fp=int(((y==0)&pred).sum())
    fn=int(((y==1)&~pred).sum());tn=int(((y==0)&~pred).sum())
    divide=lambda a,b:a/b if b else np.nan
    tpr=divide(tp,tp+fn);fpr=divide(fp,fp+tn);q=.2
    safe=np.clip(p,1e-15,1-1e-15)
    return {'n':len(y),'positive_count':int(y.sum()),'negative_count':int((y==0).sum()),
        'prevalence':float(y.mean()),'mean_probability':float(p.mean()),'tn':tn,'fp':fp,'fn':fn,'tp':tp,
        'accuracy':(tp+tn)/len(y),'balanced_accuracy':(tpr+1-fpr)/2,
        'precision':divide(tp,tp+fp),'recall':tpr,'f1':divide(2*tp,2*tp+fp+fn),
        'tpr':tpr,'fpr':fpr,'f1_q20':divide(2*q*tpr,q*(1+tpr)+(1-q)*fpr),
        'roc_auc':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else np.nan,
        'average_precision':float(average_precision_score(y,p)) if y.sum() else np.nan,
        'brier':float(np.mean((y-p)**2)),'log_loss':float(-np.mean(y*np.log(safe)+(1-y)*np.log(1-safe)))}


def threshold_curve(y,p,grid=GRID):
    rows=[]
    for threshold in grid:
        yhat=np.asarray(p)>=threshold; truth=np.asarray(y)==1
        tp=(yhat&truth).sum();fp=(yhat&~truth).sum();fn=(~yhat&truth).sum()
        den=2*tp+fp+fn
        rows.append({'threshold':float(threshold),'f1':float(2*tp/den) if den else 0.})
    return pd.DataFrame(rows)


def choose(curve):
    return float(max(curve.to_dict('records'),key=lambda r:(r['f1'],-abs(r['threshold']-.5),r['threshold']))['threshold'])


def reliability(y,p):
    d=pd.DataFrame({'target':np.asarray(y),'probability':p})
    d['bin']=pd.cut(d.probability,np.linspace(0,1,11),include_lowest=True)
    r=d.groupby('bin',observed=True).agg(n=('target','size'),mean_probability=('probability','mean'),observed_rate=('target','mean')).reset_index()
    r['bin']=r.bin.astype(str)
    return r
