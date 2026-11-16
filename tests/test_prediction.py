"""Prediction contract checks using small synthetic fits; no holdout evaluation."""
import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.prepare import FEATURES
from src.predict import (CATEGORICAL, NUMERIC, build_pipeline, evaluate,
                         positive_probability, select_threshold, validate_data)

ROOT = Path(__file__).resolve().parents[1]


class PredictionTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({
            'flight_id': range(6), 'date': ['2013-01-01', '2013-02-01', '2013-09-01',
                '2013-10-01', '2013-11-01', '2013-12-01'],
            'split': ['train', 'train', 'validation', 'validation', 'test', 'test'],
            'month': [1, 2, 9, 10, 11, 12], 'day_of_week': [1]*6,
            'day_of_year': [1, 32, 244, 274, 305, 335], 'is_weekend': [False]*6,
            'sched_dep_minutes': [300, 1200, 400, 1300, 500, 1400],
            'sched_arr_minutes': [500, 1400, 600, 1500, 700, 1600],
            'carrier': ['AA', 'DL', 'NEW', 'AA', 'DL', 'AA'],
            'flight': [1, 2, 999, 1, 2, 1], 'origin': ['JFK']*6,
            'dest': ['LAX']*6, 'distance': [2475]*6, 'delayed_15': [0, 1, 0, 1, 0, 1]})

    def test_exact_chronological_contract_and_schema(self):
        validate_data(self.data)
        for field, value in [('split', 'test'), ('month', 12), ('date', '2014-01-01'),
                             ('delayed_15', np.nan)]:
            bad = self.data.copy()
            bad.loc[0, field] = value
            with self.assertRaises(ValueError): validate_data(bad)
        with self.assertRaises(ValueError):
            validate_data(self.data.assign(dep_delay=20))
        self.assertEqual(set(CATEGORICAL + NUMERIC), set(FEATURES))
        self.assertFalse({'date', 'flight_id', 'split', 'delayed_15'} & set(FEATURES))

    def test_training_only_preprocessing_and_unknown_categories(self):
        train = self.data.iloc[:2]
        pipeline = build_pipeline(LogisticRegression())
        pipeline.fit(train, train.delayed_15)
        transform = pipeline.named_steps['preprocess']
        scaler = transform.named_transformers_['numeric'].named_steps['scale']
        np.testing.assert_allclose(scaler.mean_, train[NUMERIC].mean().to_numpy(dtype=float))
        encoder = transform.named_transformers_['categorical'].named_steps['encode']
        self.assertNotIn('NEW', encoder.categories_[0])
        self.assertNotIn(999, encoder.categories_[1])
        before = scaler.mean_.copy()
        probabilities = positive_probability(pipeline, self.data.iloc[2:])
        self.assertTrue(np.isfinite(probabilities).all())
        np.testing.assert_array_equal(before, scaler.mean_)
        altered = self.data.assign(dep_delay=999, delayed_15=1, flight_id=-1)
        np.testing.assert_array_equal(positive_probability(pipeline, self.data),
                                      positive_probability(pipeline, altered))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'pipeline.joblib'
            joblib.dump(pipeline, path)
            np.testing.assert_array_equal(probabilities,
                positive_probability(joblib.load(path), self.data.iloc[2:]))

    def test_metrics_probabilities_and_undefined_cases(self):
        metrics = evaluate([0, 0, 1, 1], [.1, .2, .3, .4], .5)
        self.assertEqual(metrics['roc_auc'], 1.)
        self.assertTrue(np.isnan(metrics['precision']))
        self.assertEqual(metrics['recall'], 0.)
        self.assertEqual(metrics['f1'], 0.)
        self.assertEqual([metrics[k] for k in ['tn','fp','fn','tp']], [2,0,2,0])
        single = evaluate([0, 0], [.1, .1], .5)
        for key in ['recall', 'f1', 'roc_auc']: self.assertTrue(np.isnan(single[key]))
        threshold, _ = select_threshold([0, 1], [.1, .9])
        self.assertEqual(threshold, .5)

    def test_saved_selected_pipeline_inference(self):
        path = ROOT/'outputs/models/flight_delay_pipeline.joblib'
        if not path.exists(): self.skipTest('Run python -m src.predict to create the artifact')
        metadata = json.loads((path.parent/'prediction_selection.json').read_text())
        self.assertEqual(metadata['features'], FEATURES)
        pipeline = joblib.load(path)
        model = pipeline.named_steps['model']
        self.assertEqual(int(pipeline.named_steps['preprocess'].named_transformers_['numeric']
                             .named_steps['scale'].n_samples_seen_), 218601)
        batch = self.data.iloc[:2]
        probabilities = positive_probability(pipeline, batch)
        self.assertEqual(probabilities.shape, (2,))
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())
        self.assertEqual(set(model.classes_), {0, 1})


if __name__ == '__main__':
    unittest.main()
