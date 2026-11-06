import unittest
import numpy as np
import pandas as pd
from src.patterns import historical_flags


class HistoricalBaselineTests(unittest.TestCase):
    def setUp(self):
        self.daily = pd.DataFrame({"delay_rate": .1, "missing_rate": .01, "eligible": True},
                                  index=pd.date_range("2013-01-01", periods=80, name="date"))

    def test_warmup_and_spike(self):
        self.daily.iloc[40, 0] = .8
        result = historical_flags(self.daily)
        self.assertTrue(result.delay_rate_flag.iloc[:21].isna().all())
        self.assertFalse(result.delay_rate_flag.iloc[21:40].any())
        self.assertTrue(result.delay_rate_flag.iloc[40])
        self.assertAlmostEqual(result.delay_rate_threshold.iloc[40], .14)

    def test_future_and_current_do_not_change_baseline(self):
        original = historical_flags(self.daily)
        self.daily.loc[self.daily.index[40]:, "delay_rate"] = .9
        changed = historical_flags(self.daily)
        pd.testing.assert_series_equal(original.delay_rate_threshold.iloc[:41],
                                       changed.delay_rate_threshold.iloc[:41])

    def test_unknown_and_ineligible_are_not_normal_days(self):
        self.daily.iloc[40, 0] = np.nan
        self.daily.iloc[41, 2] = False
        result = historical_flags(self.daily)
        self.assertTrue(result.delay_rate_flag.iloc[40:42].isna().all())


if __name__ == "__main__":
    unittest.main()
