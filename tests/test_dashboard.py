"""Dashboard calculations, saved provenance and read-only app navigation."""

import csv
from datetime import date
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

import dashboard_data as data

APP = data.ROOT / 'app.py'


class ReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.flights = pd.DataFrame({
            'date': pd.to_datetime(['2013-01-01', '2013-01-01', '2013-01-02', '2013-01-02', '2013-01-03']),
            'carrier': ['AA', 'AA', 'AA', 'UA', 'UA'], 'origin': ['JFK', 'JFK', 'LGA', 'EWR', 'JFK'],
            'dep_delay': [15, None, -43, 1301, 0],
        })

    def test_observed_denominator_and_full_severity(self):
        summary = data.flight_summary(self.flights)
        self.assertEqual((summary['flights'], summary['observed'], summary['unknown']), (5, 4, 1))
        self.assertEqual(summary['delay_rate'], .5)
        self.assertEqual(summary['delay_statistics']['Minimum'], -43)
        self.assertEqual(summary['delay_statistics']['Maximum'], 1301)
        grouped = data.grouped_reliability(self.flights, 'carrier').set_index('carrier')
        self.assertEqual(grouped.loc['AA', 'delay_rate'], .5)
        self.assertEqual(grouped.loc['AA', 'unknown'], 1)
        bars, excluded = data.departure_histogram(self.flights)
        self.assertEqual((bars.flights.sum(), excluded), (3, 1))

    def test_filter_is_inclusive_and_combines_all_dimensions(self):
        dates = (date(2013, 1, 1), date(2013, 1, 2))
        self.assertEqual(list(data.filter_flights(self.flights, dates, 'AA', 'JFK').index), [0, 1])
        self.assertEqual(list(data.filter_flights(self.flights, dates, 'UA', 'EWR').index), [3])
        self.assertTrue(data.filter_flights(self.flights, dates, 'AA', 'EWR').empty)

    def test_no_label_and_empty_selection(self):
        empty = data.filter_flights(self.flights, (date(2013, 1, 1),))
        self.assertEqual(data.flight_summary(empty)['observed'], 0)
        self.assertIsNone(data.flight_summary(empty)['delay_rate'])
        unknown_only = self.flights.iloc[[1]]
        self.assertEqual(data.flight_summary(unknown_only)['unknown'], 1)
        self.assertIsNone(data.flight_summary(unknown_only)['delay_rate'])
        self.assertTrue(pd.isna(data.grouped_reliability(unknown_only, 'carrier').delay_rate.iloc[0]))

    def test_network_scope_preserves_unknown_flags(self):
        days = pd.DataFrame({
            'date': pd.to_datetime(['2013-01-01', '2013-01-02', '2013-01-03']),
            'cluster': pd.array([None, 0, 1], dtype='Int64'),
            'delay_rate_flag': pd.array([None, False, True], dtype='boolean'),
        })
        window = (date(2013, 1, 1), date(2013, 1, 2))
        subset = data.network_days(days, window)
        self.assertEqual(subset.delay_rate_flag.dtype.name, 'boolean')
        self.assertTrue(pd.isna(subset.cluster.iloc[0]))
        self.assertEqual(list(data.flag_labels(subset.delay_rate_flag)), ['Unknown', 'Clear'])
        self.assertIn('Full NYC network', data.NETWORK_SCOPE)

    def test_missing_artifact_names_its_path_and_remedy(self):
        with TemporaryDirectory() as root:
            for path, instructions in [(data.FLIGHTS, 'README.md'), (data.DAYS, 'reports/artifact_audit.md'),
                                       (data.ORIGINAL_AP, 'saved research artifact')]:
                with self.subTest(path=path), self.assertRaises(FileNotFoundError) as error:
                    data.artifact_path(Path(root), path)
                self.assertIn(path, str(error.exception))
                self.assertIn(instructions, str(error.exception))


class SavedResultsTests(unittest.TestCase):
    def test_original_selection_majority_and_audit_ap(self):
        if not all((data.ROOT / path).is_file() for path in (data.ORIGINAL, data.SELECTION, data.ORIGINAL_AP)):
            self.skipTest('Restore original saved tables to exercise provenance')
        comparison = pd.read_csv(data.ROOT / data.ORIGINAL)
        selection = json.loads((data.ROOT / data.SELECTION).read_text())
        ap = pd.read_csv(data.ROOT / data.ORIGINAL_AP)
        rows = data.original_results(comparison, selection, ap)
        self.assertEqual(len(rows), 8)
        by_split = {(row['model'], row['sources'][0]['row']['split']): row for row in rows}
        selected = by_split[(selection['selected_model'], 'test')]
        self.assertEqual(selected['role'], 'original_frozen_assessment')
        self.assertAlmostEqual(selected['threshold'], selection['threshold'])
        reference = by_split['majority_baseline', 'test']
        self.assertEqual(reference['role'], 'original_baseline')
        self.assertEqual(reference['metrics']['recall'], 0)
        self.assertTrue(pd.isna(reference['metrics']['precision']))
        self.assertGreater(reference['metrics']['accuracy'], selected['metrics']['accuracy'])
        for exact in ap.itertuples():
            row = by_split[exact.model, exact.split]
            self.assertAlmostEqual(row['metrics']['average_precision'], exact.average_precision)
        without_ap = data.original_results(comparison, {}, None)
        self.assertTrue(all(row['metrics']['average_precision'] is None for row in without_ap))
        self.assertFalse(any(row['role'] == 'original_frozen_assessment' for row in without_ap))

    def test_canonical_rows_match_exact_source_keys_and_selection(self):
        if not (data.ROOT / data.RESULTS).is_file():
            self.skipTest('Restore canonical saved results to exercise provenance')
        canonical = json.loads((data.ROOT / data.RESULTS).read_text())
        sources = {}

        def source_row(source):
            name = source['file']
            if name not in sources:
                with data.artifact_path(data.ROOT, name).open(newline='') as handle:
                    sources[name] = list(csv.DictReader(handle))
            matches = [row for row in sources[name]
                       if all(row[key] == str(value) for key, value in source['row'].items())]
            self.assertEqual(len(matches), 1, source)
            return matches[0]

        try:
            for row in canonical['results']:
                saved = source_row(row['sources'][0])
                self.assertEqual(row['n'], int(saved['n']))
                self.assertAlmostEqual(row['threshold'], float(saved['threshold']))
                supplement = {source['field']: source_row(source)[source['field']] for source in row['sources'][1:]}
                for name, value in row['metrics'].items():
                    archived = supplement.get(name, saved.get(name))
                    if name == 'accuracy' and archived is None:
                        archived = (int(saved['tn']) + int(saved['tp'])) / int(saved['n'])
                    if value is None:
                        self.assertIn(archived, (None, ''))
                    else:
                        self.assertAlmostEqual(value, float(archived))
                if row['group'] == 'operational_one_hour' and row['model'] in ('grid_16', 'full_lr'):
                    selected = row['role'] == 'selected_before_assessment'
                    self.assertEqual(saved['winner'] == 'True', selected)
                    self.assertEqual(source_row(row['threshold_selection_source'])['winner'] == 'True', selected)
                    self.assertIn('Selected before assessment' if selected else 'Highest observed historical F1',
                                  data.model_table([row]).Role.tolist())
        except FileNotFoundError as error:
            self.skipTest(str(error))


@unittest.skipUnless(importlib.util.find_spec('streamlit'), 'Install requirements-dashboard.txt for AppTest')
class DashboardAppTests(unittest.TestCase):
    def launch(self):
        from streamlit.testing.v1 import AppTest

        return AppTest.from_file(str(APP), default_timeout=30).run()

    def test_all_sections_and_scoped_filters(self):
        from streamlit.dataframe_util import convert_arrow_bytes_to_pandas_df

        paths = (data.FLIGHTS, data.DAYS, data.PROFILES, data.CASES, data.RESULTS, data.ORIGINAL, data.SELECTION)
        if any(not (data.ROOT / path).is_file() for path in paths):
            self.skipTest('Restore the dashboard artifacts listed in README.md')
        app = self.launch()
        self.assertFalse(app.exception)
        self.assertEqual(app.metric[0].value, '336,776')
        self.assertEqual([metric.label for metric in app.metric[1:3]],
                         ['Known departure delays', 'Unknown departure delays'])
        self.assertEqual([table.value.columns[0] for table in app.dataframe], ['Airline', 'Origin'])
        airline = app.dataframe[0]
        shown = convert_arrow_bytes_to_pandas_df(airline.proto.arrow_data.styler.display_values)
        self.assertEqual(shown['Flights'].iloc[0], f"{airline.value['Flights'].iloc[0]:,}")
        self.assertEqual(shown['Delay rate'].iloc[0], f"{airline.value['Delay rate'].iloc[0]:.2%}")
        self.assertEqual(len(app.get('vega_lite_chart')), 2)
        app.date_input(key='flight_dates').set_value((date(2013, 1, 1), date(2013, 1, 2)))
        app.selectbox(key='flight_carrier').set_value('UA')
        app.selectbox(key='flight_origin').set_value('EWR').run()
        self.assertFalse(app.exception)
        self.assertLess(int(app.metric[0].value.replace(',', '')), 1000)

        app.radio(key='section').set_value('Daily disruption').run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.get('vega_lite_chart')), 2)
        self.assertEqual(len(app.selectbox), 0)
        self.assertIn(data.NETWORK_SCOPE, ' '.join(element.value for element in app.markdown))
        profiles = next(element for element in app.dataframe if 'Days' in element.value.columns)
        self.assertEqual(profiles.value.Days.tolist(), [279, 84])
        shown = convert_arrow_bytes_to_pandas_df(profiles.proto.arrow_data.styler.display_values)
        self.assertEqual(shown['Mean flights per day'].tolist(), ['911.5', '962.5'])
        self.assertEqual(shown['Mean delay rate'].tolist(), ['16.68%', '40.39%'])
        app.date_input(key='network_dates').set_value((date(2013, 1, 1), date(2013, 1, 3))).run()
        self.assertFalse(app.exception)
        captions = ' '.join(element.value for element in app.caption)
        self.assertIn('Cluster 0 groups lower-disruption days. Cluster 1 groups higher-disruption days.', captions)
        self.assertIn('363 of 365 dates were eligible and 2 were unassigned', captions)
        self.assertIn('at least 500 known departure delays covering at least 50% of its flights', captions)
        day_status = next(element.value for element in app.dataframe if 'Delay flag' in element.value.columns)
        self.assertEqual(list(day_status['Delay flag']), ['Unknown'] * 3)

        app.radio(key='section').set_value('Models and findings').run()
        self.assertFalse(app.exception)
        comparisons = [element.value for element in app.dataframe if 'Model' in element.value.columns]
        model_columns = ['Model', 'Assessment', 'Population', 'F1', 'ROC-AUC', 'Precision', 'Recall',
                         'Average precision', 'Accuracy', 'Role', 'Prediction time', 'n', 'Cutoff', 'Fit',
                         'Cutoff selection']
        self.assertTrue(all(list(table.columns) == model_columns for table in comparisons))
        self.assertTrue(any('Majority baseline' in table.Model.tolist() for table in comparisons))
        self.assertTrue(any('Waiting rule' in table.Model.tolist() for table in comparisons))
        self.assertTrue(any('Unresolved' in table.Population.tolist() for table in comparisons))
        app.selectbox(key='original_period').set_value('September–October · selection period')
        app.selectbox(key='departure_checkpoint').set_value('Scheduled departure').run()
        self.assertFalse(app.exception)
        comparisons = [element.value for element in app.dataframe if 'Model' in element.value.columns]
        self.assertTrue(any('Selected on validation' in table.Role.tolist() for table in comparisons))
        self.assertFalse(any('Waiting rule' in table.Model.tolist() for table in comparisons))

    def test_empty_and_missing_artifacts_render_useful_messages(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / data.FLIGHTS).parent.mkdir(parents=True)
            (root / data.FLIGHTS).write_text('date,carrier,origin,dep_delay\n2013-01-01,UA,EWR,0\n2013-01-02,AA,JFK,\n')
            with patch.object(data, 'ROOT', root):
                app = self.launch()
                app.selectbox(key='flight_carrier').set_value('UA')
                app.selectbox(key='flight_origin').set_value('JFK').run()
                self.assertFalse(app.exception)
                self.assertEqual(app.metric[0].value, '0')
                self.assertTrue(any('No flights match' in item.value for item in app.info))
                app.radio(key='section').set_value('Daily disruption').run()
                self.assertFalse(app.exception)
                self.assertTrue(any(data.DAYS in item.value and 'artifact_audit.md' in item.value for item in app.warning))
                app.radio(key='section').set_value('Models and findings').run()
                self.assertFalse(app.exception)
                self.assertTrue(any(data.RESULTS in item.value for item in app.warning))

    def test_missing_optional_ap_keeps_other_results_visible(self):
        from streamlit.dataframe_util import convert_arrow_bytes_to_pandas_df

        if any(not (data.ROOT / path).is_file() for path in (data.ORIGINAL, data.RESULTS, data.SELECTION)):
            self.skipTest('Restore saved model comparisons to exercise optional artifacts')
        with patch.object(data, 'ORIGINAL_AP', 'outputs/optional_ap_missing.csv'):
            app = self.launch()
            app.radio(key='section').set_value('Models and findings').run()
        self.assertFalse(app.exception)
        self.assertTrue(any('outputs/optional_ap_missing.csv' in item.value for item in app.warning))
        comparisons = [element for element in app.dataframe if 'Model' in element.value.columns]
        original = next(table for table in comparisons if 'Majority baseline' in table.value.Model.tolist())
        self.assertTrue(original.value['Average precision'].isna().all())
        shown = convert_arrow_bytes_to_pandas_df(original.proto.arrow_data.styler.display_values).set_index('Model')
        self.assertTrue(shown['Average precision'].eq('—').all())
        self.assertEqual(shown.loc['Majority baseline', 'Precision'], '—')
        self.assertTrue(any('Recent-arrivals boosting' in table.value.Model.tolist() for table in comparisons))


if __name__ == '__main__':
    unittest.main()
