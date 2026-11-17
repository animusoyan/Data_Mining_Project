"""Stable offset fitting, score-cache contracts, and paired fixed-score uncertainty."""
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit
from experiments.independent_audit.core import metrics,threshold_curve,choose,GRID,reliability

EPS=1e-12
SEED=42
COMPARISONS={
    '1':{'fold':'a','fit_end':3,'calibration':4,'threshold':5,'assessment':[6]},
    '2':{'fold':'b','fit_end':4,'calibration':5,'threshold':6,'assessment':[7,8]},
    '3':{'fold':'c','fit_end':5,'calibration':6,'threshold':7,'assessment':[8]},
    '4':{'fold':'d','fit_end':7,'calibration':8,'threshold':9,'assessment':[10]},
}


def valid_probability(p):
    p=np.asarray(p,dtype=float)
    if p.ndim!=1 or not len(p) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Require a nonempty finite 1-D probability vector in [0,1]')
    return p


def logits(p):
    p=np.clip(valid_probability(p),EPS,1-EPS)
    return np.log(p)-np.log1p(-p)


def calibrated(p,intercept):
    if not np.isfinite(intercept):raise ValueError('Intercept must be finite')
    return expit(logits(p)+float(intercept))


def fit_intercept(p,target):
    z=logits(p);y=np.asarray(target)
    if y.ndim!=1 or len(y)!=len(z) or not np.isin(y,[0,1]).all():raise ValueError('Aligned observed binary labels required')
    prevalence=float(y.mean())
    if not 0<prevalence<1:raise ValueError('Single-class calibration has no finite unpenalized intercept optimum')
    # Convex loss derivative; bracket covers clipped logits without unstable exp().
    bound=float(np.max(np.abs(z))+40)
    intercept=float(brentq(lambda b:float(expit(z+b).mean()-prevalence),-bound,bound,xtol=1e-12))
    loss=float(np.mean(np.logaddexp(0,z+intercept)-y*(z+intercept)))
    return {'intercept':intercept,'slope':1.,'epsilon':EPS,'n':len(y),'prevalence':prevalence,
            'calibrated_mean':float(expit(z+intercept).mean()),'gradient':float(expit(z+intercept).mean()-prevalence),
            'calibration_log_loss':loss,'optimizer':'brentq root of convex log-loss derivative; xtol=1e-12'}


def check_chronology(fit,calibration,threshold,assessment):
    frames=[fit,calibration,threshold,assessment]
    if any(x.empty for x in frames):raise ValueError('Empty chronological period')
    if any(x.date.isna().any() for x in frames):raise ValueError('Unknown dates')
    if not all(a.date.max()<b.date.min() for a,b in zip(frames,frames[1:])):
        raise ValueError('Require fit < calibration < threshold < assessment')
    if assessment.date.max()>pd.Timestamp('2013-10-31'):raise ValueError('No November-December inference')
    ids=[set(x.flight_id) for x in frames]
    if any(ids[i]&ids[j] for i in range(4) for j in range(i+1,4)):raise ValueError('Period ID overlap')


def acquire_scores(frame,cache,model,predictor):
    """Validate cached labels/dates; infer only missing IDs using approved development data."""
    if frame.empty or frame.date.max()>pd.Timestamp('2013-10-31'):raise ValueError('Development scores only')
    if not frame.flight_id.is_unique:raise ValueError('Nonunique requested IDs')
    if not cache.flight_id.is_unique:raise ValueError('Nonunique cached IDs')
    current=cache.set_index('flight_id').reindex(frame.flight_id)
    existing=current.probability.notna().to_numpy()
    if existing.any():
        known=current.iloc[np.flatnonzero(existing)]
        source=frame.iloc[np.flatnonzero(existing)]
        if not np.array_equal(known.target.to_numpy(),source.delayed_15.to_numpy()):raise ValueError('Cached target mismatch')
        if not np.array_equal(pd.to_datetime(known.date).to_numpy(),source.date.to_numpy()):raise ValueError('Cached date mismatch')
        valid_probability(known.probability.to_numpy())
    p=current.probability.to_numpy(dtype=float)
    missing=~existing
    if missing.any():p[missing]=valid_probability(predictor(model,frame.iloc[np.flatnonzero(missing)]))
    valid_probability(p)
    return p,{'reused_n':int(existing.sum()),'generated_n':int(missing.sum())}


def equivalent_decisions(raw,intercept,threshold):
    raw=valid_probability(raw);adjusted=calibrated(raw,intercept)
    mapped=float(calibrated(np.array([threshold]),intercept)[0])
    raw_decision=raw>=threshold;cal_decision=adjusted>=mapped
    mismatch=raw_decision!=cal_decision
    # At boundaries, expit rounding may collapse adjacent floats; explicitly audit.
    boundary=(np.isclose(raw,threshold,rtol=0,atol=1e-12)|np.isclose(adjusted,mapped,rtol=0,atol=1e-12))
    if (mismatch&~boundary).any():raise ValueError('Equivalent thresholds changed non-boundary decisions')
    return {'raw_threshold':float(threshold),'mapped_threshold':mapped,'n':len(raw),
        'mismatches':int(mismatch.sum()),'boundary_rows':int(boundary.sum()),
        'nonboundary_mismatches':int((mismatch&~boundary).sum()),'boundary_tolerance':1e-12}


def losses(y,p):
    p=valid_probability(p);y=np.asarray(y,dtype=float)
    if len(y)!=len(p) or not np.isin(y,[0,1]).all():raise ValueError('Aligned observed labels required')
    safe=np.clip(p,EPS,1-EPS)
    return np.column_stack(((p-y)**2,-(y*np.log(safe)+(1-y)*np.log1p(-safe))))


def evaluate(y,p,threshold):
    result=metrics(y,p,threshold)
    # Override scoring convention: exactly EPS=1e-12 for both methods and controls.
    loss=losses(y,p)
    result['brier'],result['log_loss']=map(float,loss.mean(axis=0))
    result['mean_prediction_error']=result['mean_probability']-result['prevalence']
    return result


def block_bootstrap(frame,replicates=500,seed=42):
    """Paired calibrated-minus-raw losses; whole seven-calendar-day moving blocks."""
    delta=losses(frame.target,frame.calibrated_probability)-losses(frame.target,frame.raw_probability)
    dates=pd.to_datetime(frame.date).dt.normalize()
    calendar=pd.date_range(dates.min(),dates.max(),freq='D')
    if len(calendar)<7:raise ValueError('Need at least seven calendar days')
    positions=calendar.get_indexer(dates)
    daily=np.column_stack([np.bincount(positions,weights=delta[:,i],minlength=len(calendar)) for i in range(2)])
    counts=np.bincount(positions,minlength=len(calendar))
    rng=np.random.default_rng(seed);draws=[]
    for _ in range(replicates):
        starts=rng.integers(0,len(calendar)-6,int(np.ceil(len(calendar)/7)))
        sampled=np.concatenate([np.arange(s,s+7) for s in starts])[:len(calendar)]
        weight=np.bincount(sampled,minlength=len(calendar))
        draws.append((daily*weight[:,None]).sum(axis=0)/(counts*weight).sum())
    draws=np.asarray(draws);point=delta.mean(axis=0);rows=[]
    for index,metric in enumerate(['brier','log_loss']):
        lo,hi=np.quantile(draws[:,index],[.025,.975])
        rows.append({'metric':metric,'difference':float(point[index]),'ci_low':float(lo),'ci_high':float(hi),
            'replicates':replicates,'seed':seed,'scheme':'paired seven-calendar-day moving blocks'})
    return pd.DataFrame(rows),pd.DataFrame(draws,columns=['brier','log_loss']).rename_axis('replicate').reset_index()
