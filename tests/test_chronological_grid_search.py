import unittest
from collections import Counter

import pandas as pd

from experiments import chronological_grid_search as grid
from experiments.independent_audit import core as audit


class ChronologicalGridTests(unittest.TestCase):
    def test_grid_coverage_and_fixed_configuration(self):
        candidates = list(grid.candidate_grid(include_catboost=True))
        self.assertEqual(len({candidate.name for candidate in candidates}), len(candidates))
        self.assertEqual(Counter(candidate.family for candidate in candidates), {
            'logistic_regression': 4, 'decision_tree': 6, 'random_forest': 12,
            'histogram_gradient_boosting': 16, 'catboost': 8, 'majority': 1,
        })
        fixed = next(candidate for candidate in candidates if candidate.name == grid.FIXED)
        self.assertEqual(grid.build(fixed).named_steps['model'].get_params(),
                         audit.build('G').named_steps['model'].get_params())
        for candidate in grid.candidate_grid(include_catboost=False):
            self.assertEqual(grid.build(candidate).named_steps['inputs'].columns, grid.COMPACT)
        self.assertTrue({'flight', 'month', 'day_of_year'}.isdisjoint(grid.COMPACT))

    def test_ranking_weights_folds_equally_and_shortlists_family_winners(self):
        rows = []
        for candidate, family, scores in [
            ('a', 'tree', [.4, .4, .7]), ('b', 'tree', [.6, .6, .2]),
            (grid.FIXED, 'boosting', [.4, .4, .4]),
            (grid.ORIGINAL, 'logistic', [.5, .5, .5]), ('lr', 'logistic', [.3, .3, .3]),
            (grid.BASELINE, 'majority', [0., 0., 0.]),
        ]:
            for fold, f1 in zip(grid.SELECTION_FOLDS, scores):
                rows.append({'candidate': candidate, 'family': family, 'settings': '{}',
                             'fold': fold, 'period': 'combined', 'n': 200 if fold == 'c' else 100,
                             **dict.fromkeys(grid.METRICS, f1)})
        rows.append({**rows[5], 'period': '2013-08', 'f1': 1.0})
        ranking = grid.rank_candidates(pd.DataFrame(rows))
        self.assertAlmostEqual(ranking.set_index('candidate').loc['a', 'mean_f1'], .5)
        self.assertGreater(ranking.set_index('candidate').loc['b', 'rank'],
                           ranking.set_index('candidate').loc['a', 'rank'])
        self.assertEqual(set(grid.shortlisted(ranking).candidate), {'a', 'lr', *grid.REFERENCES})
