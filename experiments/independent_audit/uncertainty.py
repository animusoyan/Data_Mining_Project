"""Paired fixed-score day / moving-calendar-block uncertainty; never refits."""
import numpy as np
import pandas as pd
from .core import metrics


class RankCache:
    """Weighted metrics with pre-sorted ties; integer day weights equal row replication."""
    def __init__(self,y,p,threshold):
        self.y=np.asarray(y,dtype=int);self.p=np.asarray(p,dtype=float);self.pred=self.p>=threshold
        self.order=np.argsort(-self.p,kind='stable')
        self.ends=np.r_[np.flatnonzero(np.diff(self.p[self.order])!=0),len(self.p)-1]
    def compute(self,weights):
        weights=np.asarray(weights,dtype=float)
        positive=weights*self.y;negative=weights*(1-self.y)
        tp=float(positive[self.pred].sum());fp=float(negative[self.pred].sum());fn=float(positive[~self.pred].sum())
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan
        pos=positive.sum();neg=negative.sum()
        if not pos or not neg:return np.array([f1,np.nan,np.nan])
        tps=np.cumsum(positive[self.order])[self.ends];fps=np.cumsum(negative[self.order])[self.ends]
        tpr=np.r_[0,tps/pos];fpr=np.r_[0,fps/neg]
        precision=np.divide(tps,tps+fps,out=np.zeros_like(tps),where=(tps+fps)>0)
        auc=np.trapezoid(tpr,fpr);ap=np.sum(np.diff(tpr)*precision)
        return np.array([f1,auc,ap])


def bootstrap(frame,pairs,kind,n_replicates=500,seed=42):
    """frame has date,target and p_/t_ columns. Same sampled days for both candidates."""
    dates=pd.to_datetime(frame.date).dt.normalize()
    calendar=pd.date_range(dates.min(),dates.max(),freq='D')
    positions=calendar.get_indexer(dates)
    names=sorted({name for pair in pairs for name in pair})
    caches={name:RankCache(frame.target,frame['p_'+name],frame['t_'+name].iloc[0]) for name in names}
    base={name:cache.compute(np.ones(len(frame))) for name,cache in caches.items()}
    rng=np.random.default_rng(seed);values={pair:[] for pair in pairs}
    for replicate in range(n_replicates):
        if kind=='whole_day': sampled=rng.integers(0,len(calendar),len(calendar))
        elif kind=='moving_7_day':
            if len(calendar)<7:raise ValueError('Need seven calendar days')
            starts=rng.integers(0,len(calendar)-6,int(np.ceil(len(calendar)/7)))
            sampled=np.concatenate([np.arange(s,s+7) for s in starts])[:len(calendar)]
        else:raise ValueError('Unknown bootstrap scheme')
        dayweights=np.bincount(sampled,minlength=len(calendar))
        weights=dayweights[positions]
        estimates={name:cache.compute(weights) for name,cache in caches.items()}
        for pair in pairs:values[pair].append(estimates[pair[0]]-estimates[pair[1]])
    rows=[];replicate_rows=[]
    for (candidate,reference),draws in values.items():
        draws=np.array(draws)
        for index,metric in enumerate(['f1','roc_auc','average_precision']):
            point=(base[candidate]-base[reference])[index]
            lo,hi=np.nanquantile(draws[:,index],[.025,.975])
            rows.append({'candidate':candidate,'reference':reference,'metric':metric,'point_difference':point,
                'ci_low':lo,'ci_high':hi,'scheme':kind,'replicates':n_replicates,'seed':seed,
                'valid_replicates':int(np.isfinite(draws[:,index]).sum())})
            replicate_rows.extend({'candidate':candidate,'reference':reference,'metric':metric,'replicate':i,
                'difference':float(value)} for i,value in enumerate(draws[:,index]))
    return pd.DataFrame(rows),pd.DataFrame(replicate_rows)


def wide_predictions(predictions,candidate_column='candidate'):
    wide=None
    for name,g in predictions.groupby(candidate_column):
        g=g.sort_values('flight_id').set_index('flight_id')
        if not g.index.is_unique:raise ValueError('Duplicate score IDs')
        part=g[['date','target']].copy()
        part['p_'+name]=g.probability;part['t_'+name]=g.threshold
        if wide is None:wide=part
        else:
            if not wide.index.equals(part.index) or not wide.date.equals(part.date) or not wide.target.equals(part.target):
                raise ValueError('Paired scores must have identical IDs, dates and labels')
            wide=wide.join(part.drop(columns=['date','target']))
    return wide.reset_index()
