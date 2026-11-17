"""Fixed-base Stage 2 calibration contracts; no performance expectations or base refits."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import joblib
import numpy as np
import pandas as pd

from experiments.calibration_study.core import (EPS,COMPARISONS,acquire_scores,block_bootstrap,calibrated,
    check_chronology,choose,equivalent_decisions,evaluate,fit_intercept,losses,threshold_curve,valid_probability)
from experiments.calibration_study.run import digest,verify_files
from experiments.independent_audit.core import FEATURES,probability


class FrozenProbe:
    classes_=np.array([0,1])
    def fit(self,*args,**kwargs):raise AssertionError('No base estimator refit')
    def predict_proba(self,frame):
        if list(frame.columns)!=FEATURES:raise AssertionError('Metadata/outcomes entered model')
        p=frame.sched_dep_minutes.to_numpy(dtype=float)/2000
        return np.column_stack([1-p,p])


class CalibrationStudyTests(unittest.TestCase):
    def sample(self):
        dates=pd.date_range('2013-01-01',periods=10,freq='MS')
        return pd.DataFrame({'flight_id':range(10),'date':dates,'month':dates.month,
            'delayed_15':[0,1]*5,'carrier':['AA']*10,'flight':[1]*10,'origin':['JFK']*10,'dest':['LAX']*10,
            'day_of_week':dates.dayofweek,'day_of_year':dates.dayofyear,'is_weekend':False,
            'sched_dep_minutes':np.arange(10)*100+100,'sched_arr_minutes':np.arange(10)*100+200,
            'distance':1000,'split':['train']*8+['validation']*2})

    def test_intercept_optimum_and_slope_fixed(self):
        p=np.full(10,.2);y=np.array([0]*6+[1]*4)
        result=fit_intercept(p,y)
        expected=np.log(.4/.6)-np.log(.2/.8)
        self.assertAlmostEqual(result['intercept'],expected,places=10)
        self.assertEqual(result['slope'],1.)
        self.assertAlmostEqual(result['calibrated_mean'],.4,places=12)
        self.assertLess(abs(result['gradient']),1e-12)
        optimum=losses(y,calibrated(p,result['intercept']))[:,1].mean()
        for offset in [-.1,.1]:self.assertGreater(losses(y,calibrated(p,result['intercept']+offset))[:,1].mean(),optimum)
        with self.assertRaisesRegex(ValueError,'Single-class'):fit_intercept(p,np.zeros(10))
        with self.assertRaises(ValueError):fit_intercept(p,np.full(10,np.nan))

    def test_probability_finiteness_clipping_and_monotonicity(self):
        for p in [[np.nan],[np.inf],[-.1],[1.1],[],[[.1,.2]]]:
            with self.assertRaises(ValueError):valid_probability(p)
        p=np.r_[0.,np.linspace(1e-7,1-1e-7,200),1.]
        adjusted=calibrated(p,.7)
        self.assertTrue(np.isfinite(adjusted).all());self.assertTrue((np.diff(adjusted)>0).all())
        self.assertTrue(np.isfinite(losses(np.arange(len(p))%2,adjusted)).all())
        with self.assertRaises(ValueError):calibrated(p,np.inf)
        fitted=fit_intercept(np.array([0.,1.,.2,.8]),[0,1,1,0])
        self.assertTrue(np.isfinite(fitted['intercept']))

    def test_equivalent_decisions_and_threshold_grid(self):
        p=np.array([.1,.2,np.nextafter(.3,0),.3,.4,.9])
        result=equivalent_decisions(p,-.6,.3)
        self.assertEqual(result['nonboundary_mismatches'],0)
        self.assertLessEqual(result['mismatches'],result['boundary_rows'])
        self.assertAlmostEqual(result['mapped_threshold'],calibrated(np.array([.3]),-.6)[0])
        self.assertEqual(choose(threshold_curve([0,1],[.1,.9])),.5)
        result=evaluate([0,1],[.3,.3],.3)
        self.assertEqual([result[name] for name in ['tn','fp','fn','tp']],[0,1,0,1])
        self.assertTrue(np.isnan(evaluate([0,1],[.1,.2],.5)['precision']))

    def test_chronology_and_no_late_assessment(self):
        data=self.sample()
        for period in COMPARISONS.values():
            frames=[data.loc[data.month<=period['fit_end']],
                data.loc[data.month.eq(period['calibration'])],data.loc[data.month.eq(period['threshold'])],
                data.loc[data.month.isin(period['assessment'])]]
            check_chronology(*frames)
            with self.assertRaisesRegex(ValueError,'chronology|Require fit'):check_chronology(frames[0],frames[2],frames[1],frames[3])
        late=frames[3].copy();late.date=pd.Timestamp('2013-11-01')
        with self.assertRaisesRegex(ValueError,'November'):check_chronology(*frames[:3],late)

    def test_missing_only_scores_no_base_fit_and_input_isolation(self):
        frame=self.sample().iloc[:4].copy();model=FrozenProbe()
        cache=frame.iloc[:2][['flight_id','date']].assign(target=frame.delayed_15.iloc[:2],probability=[.05,.1])
        predictor=Mock(wraps=probability)
        before=joblib.hash(model)
        p,counts=acquire_scores(frame,cache,model,predictor)
        self.assertEqual(counts,{'reused_n':2,'generated_n':2})
        self.assertEqual(predictor.call_count,1)
        self.assertEqual(predictor.call_args.args[1].flight_id.tolist(),[2,3])
        np.testing.assert_array_equal(p,[.05,.1,.15,.2])
        self.assertEqual(joblib.hash(model),before)
        original=probability(model,frame)
        altered=frame.assign(delayed_15=999,flight_id=-1,split='anything',dep_delay=999)
        np.testing.assert_array_equal(original,probability(model,altered))
        full=frame[['flight_id','date']].assign(target=frame.delayed_15,probability=p)
        never=Mock(side_effect=AssertionError('Unnecessary base inference'))
        acquire_scores(frame,full,model,never);never.assert_not_called()
        wrong=full.copy();wrong.loc[0,'target']=1
        with self.assertRaisesRegex(ValueError,'Cached target'):acquire_scores(frame,wrong,model,never)

    def test_calibrator_metadata_isolation(self):
        frame=self.sample();p=np.linspace(.1,.7,len(frame))
        before=fit_intercept(p,frame.delayed_15)
        changed=frame.assign(date='invalid',flight_id=-1,split='bad',dep_delay=999)
        self.assertEqual(before,fit_intercept(p,changed.delayed_15))
        self.assertNotEqual(before['intercept'],fit_intercept(p,np.array([1]*7+[0]*3))['intercept'])

    def test_block_bootstrap_is_paired_and_reproducible(self):
        frame=pd.DataFrame({'date':pd.date_range('2013-06-01',periods=30),'target':[0,1]*15,
            'raw_probability':.3,'calibrated_probability':.3})
        a,draws=block_bootstrap(frame,30,42)
        self.assertTrue((a[['difference','ci_low','ci_high']].to_numpy()==0).all())
        frame.calibrated_probability=.4
        a,draws=block_bootstrap(frame,30,42);b,other=block_bootstrap(frame,30,42)
        pd.testing.assert_frame_equal(a,b);pd.testing.assert_frame_equal(draws,other)
        actual=losses(frame.target,frame.calibrated_probability)-losses(frame.target,frame.raw_probability)
        np.testing.assert_allclose(a.difference,actual.mean(axis=0))

    def test_existing_file_preservation_distinguishes_new_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path=root/'old.model';path.write_bytes(b'frozen')
            manifest={'old.model':digest(path)}
            (root/'new-stage2.txt').write_text('new')
            self.assertTrue(verify_files(manifest,root)['unchanged'])
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'protected files'):verify_files(manifest,root)


if __name__=='__main__':unittest.main()
