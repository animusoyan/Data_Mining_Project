"""Guard experimental chronology, input isolation, and deterministic feature engineering."""
import tempfile
import subprocess
import sys
import json
import joblib
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from experiments.modeling_study import (FOLDS, ScheduleFeatures, engineered_pipeline,
                                        load_development, metrics, probability)
from src.prepare import FEATURES


class ModelingStudyTests(unittest.TestCase):
    def sample(self):
        return pd.DataFrame({'flight_id':[1,2,3,4], 'date':['2013-01-01','2013-02-01','2013-09-01','2013-11-01'],
            'split':['train','train','validation','test'],'delayed_15':[0,1,0,1],
            'month':[1,2,9,11],'day_of_year':[1,32,244,305], 'day_of_week':[1,4,6,4],
            'is_weekend':[False,False,True,False],'sched_dep_minutes':[0,1439,600,1200],
            'sched_arr_minutes':[100,200,800,1400], 'carrier':['AA','DL','NEW','AA'],
            'flight':[1,1,999,1], 'origin':['JFK']*4, 'dest':['LAX']*4, 'distance':[2475]*4})

    def test_test_targets_are_not_parsed(self):
        data=self.sample().astype({'delayed_15':object})
        data.loc[3,'delayed_15']='FORBIDDEN_INVALID_TEST_LABEL'
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'classifier.csv'
            data.to_csv(path,index=False)
            result=load_development(path)
        self.assertEqual(len(result),3)
        self.assertLessEqual(result.month.max(),10)
        self.assertEqual(result.delayed_15.tolist(),[0,1,0])

    def test_fold_threshold_and_assessment_are_separate_and_future(self):
        for fit_end,threshold_month,assessment_months in FOLDS:
            self.assertLess(fit_end,threshold_month)
            self.assertLess(threshold_month,min(assessment_months))
            self.assertLessEqual(max(assessment_months),8)

    def test_features_ignore_targets_and_clock_cycle_wraps(self):
        data=self.sample()
        transformer=ScheduleFeatures('interactions').fit(data)
        a=transformer.transform(data)
        b=transformer.transform(data.assign(delayed_15=999,flight_id=-1,date='invalid',dep_delay=999))
        pd.testing.assert_frame_equal(a,b)
        self.assertNotIn('delayed_15',a)
        self.assertNotEqual(a.flight.iloc[0],a.flight.iloc[1])
        self.assertLess(abs(a.sched_dep_minutes_sin.iloc[0]-a.sched_dep_minutes_sin.iloc[1]),.005)
        self.assertLess(abs(a.sched_dep_minutes_cos.iloc[0]-a.sched_dep_minutes_cos.iloc[1]),.005)
        self.assertNotIn('day_of_year',a)

    def test_estimator_cloning_and_unseen_categories(self):
        estimator=LogisticRegression(solver='liblinear')
        a=engineered_pipeline(estimator,'interactions')
        b=engineered_pipeline(estimator,'interactions')
        self.assertIsNot(a.named_steps['model'],b.named_steps['model'])
        self.assertIsNot(a.named_steps['model'],estimator)
        data=self.sample()
        a.fit(data.iloc[:2][FEATURES],data.iloc[:2].delayed_15)
        self.assertTrue(np.isfinite(probability(a,data.iloc[2:3])).all())
        scaler=a.named_steps['preprocess'].named_transformers_['numeric'].named_steps['scale']
        self.assertEqual(scaler.n_samples_seen_,2)

    def test_extended_metrics_and_positive_dummy_reference(self):
        result=metrics([0,0,0,1],np.ones(4),.5)
        self.assertEqual(result['accuracy'],.25)
        self.assertEqual(result['balanced_accuracy'],.5)
        self.assertEqual(result['f1'],.4)
        self.assertEqual(result['roc_auc'],.5)
        self.assertEqual(result['average_precision'],.25)
        result=metrics([0,0],np.zeros(2),.5)
        self.assertTrue(np.isnan(result['balanced_accuracy']))
        self.assertTrue(np.isnan(result['average_precision']))

    def test_experimental_pipeline_loads_in_fresh_process(self):
        data=self.sample()
        pipeline=engineered_pipeline(LogisticRegression(solver='liblinear'),'interactions')
        pipeline.fit(data.iloc[:2][FEATURES],data.iloc[:2].delayed_15)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model.joblib'
            joblib.dump(pipeline,path)
            result=subprocess.run([sys.executable,'-B','-c',
                "import joblib,sys; m=joblib.load(sys.argv[1]); assert type(m.named_steps['features']).__module__ == 'experiments.modeling_study'",str(path)],
                capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)

    def test_original_artifact_hashes_preserved(self):
        from experiments.modeling_study import OUT, protected_hashes
        manifest=OUT/'preservation_check.json'
        if not manifest.exists(): self.skipTest('Run the controlled study first')
        saved=json.loads(manifest.read_text())
        self.assertTrue(saved['all_original_hashes_match'])
        self.assertEqual(saved['files'],protected_hashes())


if __name__=='__main__': unittest.main()
