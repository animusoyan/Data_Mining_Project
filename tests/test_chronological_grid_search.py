import json
import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from experiments import chronological_grid_search as grid
from experiments.independent_audit import core as audit


class ChronologicalGridTests(unittest.TestCase):
    def test_grid_coverage_and_fixed_configuration(self):
        candidates = list(grid.candidate_grid())
        self.assertEqual(len({candidate.name for candidate in candidates}), len(candidates))
        self.assertEqual(len(candidates), 39)
        self.assertEqual(Counter(candidate.family for candidate in candidates), {
            'logistic_regression': 4, 'decision_tree': 6, 'random_forest': 12,
            'histogram_gradient_boosting': 16, 'majority': 1,
        })
        fixed = next(candidate for candidate in candidates if candidate.name == grid.FIXED)
        self.assertEqual(grid.build(fixed).named_steps['model'].get_params(),
                         audit.build('G').named_steps['model'].get_params())
        for candidate in candidates:
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

    def test_resume_rejects_retired_candidates_before_loading_data(self):
        with TemporaryDirectory() as temp:
            source = Path(temp) / 'classifier.csv'
            output = Path(temp) / 'results'
            with patch.object(audit, 'load_development', side_effect=FileNotFoundError):
                with self.assertRaises(FileNotFoundError):
                    grid.run(source, output)

            protocol_path = output / 'protocol.json'
            current = protocol_path.read_bytes()
            old = json.loads(current)
            old['candidates']['retired_01'] = {'family': 'retired'}
            grid.write_json(protocol_path, old)
            old_protocol = protocol_path.read_bytes()
            checkpoint = output / 'candidate_checkpoints/a_retired_01.json'
            grid.write_json(checkpoint, {'scores': [], 'curve': []})
            with patch.object(audit, 'load_development') as load:
                with self.assertRaisesRegex(ValueError, r'incompatible grid protocol.*--output'):
                    grid.run(source, output, resume=True)
                self.assertEqual(protocol_path.read_bytes(), old_protocol)

                protocol_path.write_bytes(current)
                with self.assertRaisesRegex(ValueError, r'unexpected candidate checkpoints.*retired_01'):
                    grid.run(source, output, resume=True)
                self.assertEqual(protocol_path.read_bytes(), current)
                self.assertTrue(checkpoint.exists())

                protocol_path.unlink()
                with self.assertRaisesRegex(ValueError, r'checkpoints without a saved protocol'):
                    grid.run(source, output, resume=True)
                self.assertFalse(protocol_path.exists())
                load.assert_not_called()

    def test_checkpoint_reuses_matching_scores_and_rejects_mislabeled_scores(self):
        with TemporaryDirectory() as temp:
            directory = Path(temp)
            candidate = grid.Candidate(grid.BASELINE, 'majority', {})
            checkpoint = directory / f'a_{candidate.name}.json'
            row = {'fold': 'a', 'candidate': candidate.name, 'family': candidate.family, 'settings': '{}'}
            curve = [{'fold': 'a', 'candidate': candidate.name, 'threshold': .5}]
            grid.write_json(checkpoint, {'scores': [row], 'curve': curve})
            with patch.object(grid, 'fit_and_score') as fit:
                scores, loaded_curve = grid.checkpoint_score(directory, candidate, 'a', periods=None)
                self.assertEqual(scores, [row])
                self.assertEqual(loaded_curve.to_dict('records'), curve)
                grid.write_json(checkpoint, {'scores': [{**row, 'family': 'retired'}], 'curve': curve})
                with self.assertRaisesRegex(ValueError, r'does not match its fold and candidate'):
                    grid.checkpoint_score(directory, candidate, 'a', periods=None)
                fit.assert_not_called()
