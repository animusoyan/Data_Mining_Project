"""Checks for time boundaries, unknown labels, leakage and temporal partitions."""
import unittest
from pathlib import Path

import pandas as pd

from src.prepare import FEATURES, clock_minutes

ROOT = Path(__file__).resolve().parents[1]


class PreparationTests(unittest.TestCase):
    def test_clock_boundaries(self):
        result = clock_minutes(pd.Series([0, 59, 100, 2359, 2400, None]))
        self.assertEqual(result.iloc[:5].tolist(), [0, 59, 60, 1439, 0])
        self.assertTrue(pd.isna(result.iloc[5]))
        for invalid in [-1, 60, 1260, 2401, 2500, 123.5]:
            with self.assertRaises(ValueError):
                clock_minutes(pd.Series([invalid]))

    def test_prepared_exports(self):
        clean = pd.read_csv(ROOT / "data/processed/flights_clean.csv")
        model = pd.read_csv(ROOT / "data/processed/classifier.csv")
        unknown = pd.read_csv(ROOT / "data/processed/missing_departure_delay.csv")
        self.assertEqual(len(clean), 336776)
        self.assertEqual(len(model) + len(unknown), len(clean))
        self.assertTrue(unknown.dep_delay.isna().all())
        self.assertTrue(unknown.delayed_15.isna().all())
        self.assertFalse(model.delayed_15.isna().any())
        observed = clean.loc[clean.dep_delay.notna()].set_index("flight_id")
        self.assertTrue(observed.delayed_15.eq(observed.dep_delay.ge(15).astype(int)).all())
        self.assertEqual(set(model.columns), {"flight_id", "date", "split", "delayed_15", *FEATURES})
        pd.testing.assert_series_equal(model.set_index("flight_id").delayed_15.astype(int),
                                       observed.delayed_15.astype(int))
        dates = pd.to_datetime(model.date)
        self.assertLess(dates[model.split.eq("train")].max(), dates[model.split.eq("validation")].min())
        self.assertLess(dates[model.split.eq("validation")].max(), dates[model.split.eq("test")].min())


if __name__ == "__main__":
    unittest.main()
