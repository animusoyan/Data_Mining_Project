"""Stage 1 contracts and integration checks; never assert expected model performance."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.independent_audit.core import (FEATURES,FOLDS,build,choose,clock,compare_by_id,
    fit_record,load_development,metrics,partitions,probability,reconstruct,target,threshold_curve)
from experiments.independent_audit.history import verify_saved_rows
from experiments.independent_audit.run import claim_phase,protected_hashes,verify_preservation
from experiments.independent_audit.uncertainty import RankCache,bootstrap


class IndependentAuditTests(unittest.TestCase):
    def sample(self):
        months=np.arange(1,13)
        return pd.DataFrame({'flight_id':months,'date':pd.to_datetime([f'2013-{m:02}-01' for m in months]),
            'month':months,'day_of_year':[1,32,60,91,121,152,182,213,244,274,305,335],
            'split':['train']*8+['validation']*2+['test']*2,'delayed_15':[0,1]*6,
            'day_of_week':[1,4,4,0,2,5,0,3,6,1,4,6],'is_weekend':[False]*12,
            'sched_dep_minutes':months*100,'sched_arr_minutes':months*100+100,
            'carrier':['AA','DL']*6,'flight':[1,2]*6,'origin':['JFK']*12,'dest':['LAX']*12,'distance':[2475]*12})

    def test_exact_mismatch_detection_and_unknown_target(self):
        frame=self.sample().iloc[:3].copy()
        frame['delayed_15']=target(pd.Series([14,15,np.nan]))
        compare_by_id(frame,frame.iloc[::-1])
        wrong=frame.copy();wrong.loc[2,'delayed_15']=0
        with self.assertRaisesRegex(ValueError,'Missingness'):compare_by_id(frame,wrong)
        wrong=frame.copy();wrong.loc[0,'distance']=1
        with self.assertRaisesRegex(ValueError,'Value mismatch'):compare_by_id(frame,wrong)
        with self.assertRaises(ValueError):compare_by_id(frame,frame.iloc[:2])
        with self.assertRaises(ValueError):compare_by_id(frame,pd.concat([frame,frame.iloc[:1]]))

    def test_boundaries_and_clock(self):
        labels=target(pd.Series([-43,14,15,16,1301,np.nan]))
        self.assertEqual(labels.iloc[:5].tolist(),[0,0,1,1,1]);self.assertTrue(pd.isna(labels.iloc[5]))
        result=clock(pd.Series([0,59,100,2359,2400,np.nan]))
        self.assertEqual(result.iloc[:5].tolist(),[0,59,60,1439,0]);self.assertTrue(pd.isna(result.iloc[5]))
        for invalid in [-1,60,1260,2401,123.5]:
            with self.assertRaises(ValueError):clock(pd.Series([invalid]))

    def test_filter_sentinel_before_target_conversion(self):
        data=self.sample().astype({'delayed_15':object})
        data.loc[data.month>10,'delayed_15']='DO_NOT_PARSE_THIS_TEST_TARGET'
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'classifier.csv';data.to_csv(path,index=False)
            result=load_development(path)
            self.assertEqual(len(result),10);self.assertLessEqual(result.date.max(),pd.Timestamp('2013-10-31'))
            data.loc[0,'distance']=np.nan;data.to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,'missing'):load_development(path)

    def test_fold_order_and_inference_date_boundary(self):
        data=self.sample()
        for fold in FOLDS:
            a,b,c=partitions(data,fold)
            self.assertLess(a.date.max(),b.date.min());self.assertLess(b.date.max(),c.date.min())
        wrong=data.copy();wrong.loc[wrong.month.eq(4),'date']=pd.Timestamp('2013-01-01')
        with self.assertRaisesRegex(ValueError,'chronology'):partitions(wrong,'a')
        with self.assertRaisesRegex(ValueError,'outside development'):probability(build('A'),data)

    def test_threshold_ties_metrics_edges_and_saved_row_alignment(self):
        curve=threshold_curve([0,1],[.1,.9])
        self.assertEqual(choose(curve),.5)
        self.assertEqual(choose(pd.DataFrame({'threshold':[.4,.6],'f1':[1.,1.]})),.6)
        result=metrics([0,1],[.3,.3],.3)
        self.assertEqual([result[k] for k in ['tn','fp','fn','tp']],[0,1,0,1])
        result=metrics([0,1],[.1,.2],.5)
        self.assertTrue(np.isnan(result['precision']));self.assertEqual(result['recall'],0)
        self.assertEqual(result['roc_auc'],1)
        self.assertTrue(np.isnan(metrics([0,0],[.1,.2])['roc_auc']))
        classifier=self.sample().query("split != 'train'")
        scores=classifier[['flight_id','date','split','month','delayed_15']].copy()
        scores=scores.assign(model='lr',probability=[.3,.7,.4,.9],threshold=.4,prediction=[0,1,1,1])
        verify_saved_rows(scores,classifier,{'thresholds':{'lr':.4}})
        wrong=scores.copy();wrong.loc[wrong.index[0],'prediction']=1
        with self.assertRaisesRegex(ValueError,'stored threshold'):verify_saved_rows(wrong,classifier,{'thresholds':{'lr':.4}})

    def test_models_fresh_instances_and_unknown_categories(self):
        data=self.sample().iloc[:8]
        with threadpool_limits(limits=2):
            for candidate in ['A','G']:
                a,b=build(candidate),build(candidate)
                self.assertIsNot(a.named_steps['model'],b.named_steps['model'])
                self.assertIsNot(a.named_steps['preprocess'],b.named_steps['preprocess'])
                fit_record(a,data)
                unknown=data.iloc[:2].copy().assign(carrier='NEW',origin='NEW',dest='NEW',day_of_week=99)
                self.assertTrue(np.isfinite(probability(a,unknown)).all())
                if candidate=='G':
                    transformed=a.named_steps['preprocess'].transform(a.named_steps['inputs'].transform(unknown))
                    self.assertTrue(np.isnan(np.asarray(transformed[:,:4],dtype=float)).all())
                    np.testing.assert_array_equal(a.named_steps['model'].is_categorical_,[True]*4+[False]*4)
                else:
                    scaler=a.named_steps['preprocess'].named_transformers_['numeric']
                    self.assertEqual(scaler.n_samples_seen_,len(data))

    def test_model_input_isolation_and_no_silent_imputation(self):
        data=self.sample().iloc[:8]
        for candidate in 'ABCDEF':
            model=build(candidate);fit_record(model,data)
            changed=data.assign(delayed_15=999,date='not_a_date',flight_id=-1,dep_delay=1000,split='test')
            np.testing.assert_array_equal(model.predict_proba(data),model.predict_proba(changed))
            used=model.named_steps['inputs'].columns
            self.assertFalse({'delayed_15','date','flight_id','dep_delay','split'}&set(used))
            bad=data.copy();bad.loc[bad.index[0],'distance']=np.nan
            with self.assertRaisesRegex(ValueError,'Missing approved'):model.predict_proba(bad)
        self.assertNotIn('flight',build('C').named_steps['inputs'].columns)
        self.assertEqual(set(build('F').named_steps['inputs'].columns),set(build('G').named_steps['inputs'].columns))

    def test_weighted_bootstrap_metrics_match_replication(self):
        y=np.array([0,1,0,1,1,0]);p=np.array([.1,.2,.2,.7,.7,.9]);weights=np.array([2,1,0,3,1,2])
        result=RankCache(y,p,.3).compute(weights)
        full=metrics(np.repeat(y,weights),np.repeat(p,weights),.3)
        np.testing.assert_allclose(result,[full['f1'],full['roc_auc'],full['average_precision']])
        days=pd.date_range('2013-09-01',periods=10)
        frame=pd.DataFrame({'date':np.repeat(days,2),'target':[0,1]*10,'p_A':[.2,.6]*10,'t_A':.5,'p_C':[.3,.7]*10,'t_C':.5})
        a,_=bootstrap(frame,[('C','A')],'moving_7_day',10,42)
        b,_=bootstrap(frame,[('C','A')],'moving_7_day',10,42)
        pd.testing.assert_frame_equal(a,b)

    def test_phase_duplicate_and_historical_preservation_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);(out/'protocol.json').write_text(json.dumps({'source_checksums':{}}))
            with patch('experiments.independent_audit.run.source_hashes',return_value={}),patch('experiments.independent_audit.run.verify_preservation',return_value={}):
                self.assertTrue(claim_phase(out,'experiment'))
                with self.assertRaises(FileExistsError):claim_phase(out,'experiment')
                (out/'experiment.done.json').write_text('{}')
                self.assertFalse(claim_phase(out,'experiment'))
            (out/'src').mkdir();source=out/'src/original.py';source.write_text('original')
            before=protected_hashes(out);source.write_text('modified');after=protected_hashes(out)
            self.assertNotEqual(before,after)
            (out/'protocol.json').write_text(json.dumps({'protected_before':before}))
            with patch('experiments.independent_audit.run.protected_hashes',return_value=after):
                with self.assertRaisesRegex(ValueError,'Protected files changed'):verify_preservation(out)


if __name__=='__main__':unittest.main()
