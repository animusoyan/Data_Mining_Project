import unittest

import numpy as np
import pandas as pd

from experiments.flight_delay_replay.evaluation import (
    chronological_masks, departure_resolved, make_model, select_cutoff,
)
from experiments.flight_delay_replay.features import (
    NY, arrival_features, as_of, departure_features, event_times, local_clock,
    rotation_matches, schedule_times, utc_minutes,
)


def fixture():
    flights = pd.DataFrame({
        'date': pd.to_datetime(['2013-01-01'] * 4 + ['2013-01-02'] * 4),
        'origin': ['JFK'] * 8, 'dest': ['ORD'] * 8, 'carrier': ['AA'] * 8,
        'flight': [10, 20, 30, 40] * 2, 'tailnum': ['A', 'B', 'A', 'C'] * 2,
        'sched_dep_time': [700, 900, 1100, 1300] * 2,
        'sched_arr_time': [800, 1000, 1200, 1400] * 2,
        'dep_time': [658, 922, 1103, 1335, 658, 922, 1103, 1335],
        'arr_time': [757, 1017, 1209, 1429, 757, 1017, 1209, 1429],
        'dep_delay': [-2, 22, 3, 35] * 2, 'arr_delay': [-3, 17, 9, 29] * 2,
        'distance': [740] * 8,
    })
    flights['day_of_week'] = flights.date.dt.dayofweek
    flights['month'] = flights.date.dt.month
    flights['is_weekend'] = False
    flights['route'] = flights.origin + '-' + flights.dest
    for name in ('dep', 'arr'):
        hhmm = flights[f'sched_{name}_time']
        flights[f'sched_{name}_minutes'] = (hhmm // 100) * 60 + hhmm % 100
    return flights


class FlightDelayReplayTests(unittest.TestCase):
    def setUp(self):
        self.zones = pd.Series({'JFK': NY, 'ORD': 'America/Chicago'})

    def test_clock_midnight_overnight_and_daylight_saving(self):
        date = pd.Series(pd.to_datetime(['2013-01-01', '2013-03-10', '2013-11-03']))
        local = local_clock(date, pd.Series([2400, 230, 130]))
        self.assertEqual(local.iloc[0], pd.Timestamp('2013-01-02'))
        zone = pd.Series(NY, index=date.index)
        daylight = utc_minutes(local, zone)
        standard = utc_minutes(local, zone, daylight=False)
        self.assertEqual(standard.iloc[2] - daylight.iloc[2], 60)
        self.assertEqual(pd.to_datetime(daylight.iloc[1], unit='m', utc=True).hour, 7)
        overnight = fixture().iloc[[0]].assign(sched_dep_time=2340, sched_arr_time=110)
        schedule = schedule_times(overnight, self.zones)
        self.assertEqual((schedule.arrival - schedule.departure).iloc[0], 150)
        self.assertEqual(schedule_times(overnight.assign(dep_time=200, arr_delay=100), self.zones)
                         .arrival.iloc[0], schedule.arrival.iloc[0])

    def test_confirmations_only_at_report_time(self):
        flights = fixture()
        schedule = schedule_times(flights, self.zones)
        events = event_times(flights, schedule, self.zones)
        self.assertTrue(as_of(events.departed, events.departed).all())
        self.assertFalse(as_of(events.arrived, events.arrived - 1).any())
        self.assertFalse(departure_resolved(flights, events, events.departed - 1).any())
        self.assertEqual(departure_resolved(flights, events, events.departed).tolist(),
                         [True, False, True, False] * 2)
        future_departure = events.assign(departed=schedule.arrival + 1)
        features, _ = arrival_features(flights, schedule, future_departure)
        self.assertFalse(features.departure_confirmed.any())
        self.assertTrue(features.departure_delay_reported.isna().all())
        self.assertTrue(features.minutes_since_departure.isna().all())
        self.assertTrue(features.destination_arrivals_60_count.eq(0).all())

    def test_later_outcomes_and_aircraft_assignments_cannot_change_earlier_features_or_scores(self):
        flights = fixture()
        schedule = schedule_times(flights, self.zones)
        cutoff = schedule.departure - 60

        def features(frame):
            events = event_times(frame, schedule, self.zones)
            rotations = rotation_matches(frame, schedule, events)
            return departure_features(frame, schedule, events, cutoff, rotations)

        original, blocks = features(flights)
        self.assertEqual(original.loc[6, 'prior_service_dep_delay'], -2)
        changed = flights.copy()
        changed.loc[6:, ['dep_time', 'arr_time', 'dep_delay', 'arr_delay', 'tailnum']] = [
            [1150, 1249, 50, 49, 'X'], [1400, 1510, 60, 70, 'Y'],
        ]
        replayed, _ = features(changed)
        pd.testing.assert_frame_equal(original.loc[:6], replayed.loc[:6])
        original_arrivals, _ = arrival_features(flights, schedule, event_times(flights, schedule, self.zones))
        replayed_arrivals, _ = arrival_features(changed, schedule, event_times(changed, schedule, self.zones))
        pd.testing.assert_frame_equal(original_arrivals.loc[:5], replayed_arrivals.loc[:5])
        columns = blocks['prior_service']
        model = make_model('lr', columns).fit(original.loc[:3, columns], flights.loc[:3, 'dep_delay'].ge(15))
        original_probability = model.predict_proba(original.loc[[6], columns])
        replayed_probability = model.predict_proba(replayed.loc[[6], columns])
        np.testing.assert_array_equal(original_probability, replayed_probability)

    def test_chronology_excludes_missing_labels_without_excluding_early_flights(self):
        flights = pd.DataFrame({'month': [1, 6, 7, 8, 9, 10, 11, 12]})
        flights['date'] = pd.to_datetime({'year': 2013, 'month': flights.month, 'day': 1})
        labels = pd.Series([-7, 20, 2, 3, 6, np.nan, 15, -1])
        fit, tune, assessments = chronological_masks(flights, 6, labels.notna())
        self.assertEqual(flights.month[fit].tolist(), [1, 6])
        self.assertEqual(flights.month[tune].tolist(), [7, 8])
        self.assertEqual([flights.month[mask].tolist() for mask in assessments.values()], [[9], [11, 12]])
        fit, tune, _ = chronological_masks(flights, 7, labels.notna())
        self.assertEqual(flights.month[fit].tolist(), [1, 6, 7])
        self.assertEqual(flights.month[tune].tolist(), [8])
        self.assertEqual(select_cutoff(np.array([False, True]), np.array([.3, .6]), 'accuracy'), .5)
