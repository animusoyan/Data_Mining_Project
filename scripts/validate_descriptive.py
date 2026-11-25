"""Read-only checks for the descriptive recovery exports."""

import argparse
from difflib import unified_diff
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def read_table(run, name, keys):
    return pd.read_csv(run / f'tables/{name}.csv', index_col=keys, float_precision='round_trip')


def same(saved, expected):
    pd.testing.assert_frame_equal(saved, expected, check_dtype=False, check_exact=False,
                                  check_index_type=False, rtol=1e-12, atol=1e-12)


def counts(data, keys):
    return data.groupby(keys, observed=True).agg(
        flights=('flight_id', 'size'), observed_labels=('dep_delay', 'count'),
        delayed=('is_delayed', 'sum'))


def check_group(run, name, data, keys):
    saved = read_table(run, name, keys)
    expected = counts(data, keys)
    if isinstance(expected.index, pd.DatetimeIndex):
        expected.index = expected.index.strftime('%Y-%m-%d')
    labels = expected.observed_labels.replace(0, np.nan)
    expected['delay_rate'] = expected.delayed / labels
    expected['missing_labels'] = expected.flights - expected.observed_labels
    expected['missing_rate'] = expected.missing_labels / expected.flights
    for field, calculator in [
        ('mean_delay', 'mean'), ('median_delay', 'median'), ('p90_delay', lambda x: x.quantile(.9)),
    ]:
        if field in saved:
            expected[field] = data.groupby(keys, observed=True).dep_delay.agg(calculator).to_numpy()
    available = [field for field in saved if field in expected]
    same(saved[available].sort_index(), expected[available].sort_index())
    return saved


def check_eda(run, flights):
    groups = {
        'monthly': 'month',
        'airlines': 'carrier', 'airports': 'origin', 'routes': 'route',
        'hourly': 'hour', 'daily': 'date', 'weekday_hour': ['day_of_week', 'hour'],
        'airport_month': ['origin', 'month'], 'carrier_month': ['carrier', 'month'],
    }
    full = {name: check_group(run, name, flights, keys) for name, keys in groups.items()}
    # Weekly uses calendar grouping rather than grouping on the source date directly.
    weekly = flights.assign(week_ending=flights.date.dt.to_period('W-SUN').dt.end_time.dt.normalize())
    check_group(run, 'weekly', weekly.rename(columns={'date': 'departure_date', 'week_ending': 'date'}), 'date')
    daily, routes = full['daily'], full['routes']

    for name, keys, expected in [
        ('selected_routes', 'route', routes.nlargest(10, 'flights').sort_values('delay_rate')),
        ('july_top_days', 'date', daily.loc[daily.index.str.startswith('2013-07')].nlargest(5, 'delay_rate')),
        ('worst_days', 'date', daily.loc[daily.observed_labels.ge(500)].nlargest(10, 'delay_rate')),
    ]:
        same(read_table(run, name, keys), expected)
    focus_date = read_table(run, 'worst_days', 'date').index[0]
    focus = flights.loc[flights.date.eq(pd.Timestamp(focus_date))]
    for name, key, base, minimum in [
        ('worst_day_origins', 'origin', 'airports', 0),
        ('worst_day_carriers', 'carrier', 'airlines', 30),
    ]:
        observed = focus.groupby(key).dep_delay.count()
        table = check_group(run, name, focus.loc[focus[key].isin(observed[observed.ge(minimum)].index)], key)
        np.testing.assert_allclose(table.annual_rate, full[base].delay_rate.loc[table.index], rtol=1e-12)

    morning = flights.hour.between(5, 9)
    evening = flights.hour.between(17, 21)
    bands = flights.assign(period=np.where(morning, 'morning', 'evening')).loc[morning | evening]
    table = check_group(run, 'airport_month_time_bands', bands, ['origin', 'month', 'period'])
    rates, labels = table.delay_rate.unstack(), table.observed_labels.unstack()
    contrasts = rates.loc[labels.ge(100).all(axis=1)].copy()
    contrasts['evening_minus_morning_pp'] = (contrasts.evening - contrasts.morning) * 100
    same(read_table(run, 'evening_morning_contrasts', ['origin', 'month']), contrasts.rename_axis(columns=None))

    matched = read_table(run, 'matched_airline_comparisons', ['route', 'carrier'])
    observed = flights.loc[flights.dep_delay.notna()].assign(time_band=lambda d: d.hour // 4 * 4)
    for route, rows in matched.groupby(level='route', sort=False):
        carriers = rows.index.get_level_values('carrier')
        assert len(carriers) == 2 and carriers.is_unique
        group = observed.loc[observed.route.eq(route) & observed.carrier.isin(carriers)]
        cells = group.groupby(['month', 'time_band', 'carrier']).is_delayed.agg(['size', 'mean'])
        n = cells['size'].unstack().reindex(columns=carriers)
        shared = n.ge(20).all(axis=1)
        common = n.loc[shared]
        shared_rates = cells['mean'].unstack().loc[shared]
        weights = common.sum(axis=1) / common.to_numpy().sum()
        for carrier, row in rows.droplevel('route').iterrows():
            carrier_rows = group.loc[group.carrier.eq(carrier)]
            shared_n = int(common[carrier].sum())
            np.testing.assert_allclose(
                row[['route_observed', 'shared_observed', 'coverage', 'shared_cells',
                     'raw_route_rate', 'raw_shared_rate', 'standardized_rate']],
                [len(carrier_rows), shared_n, shared_n / len(carrier_rows), len(common),
                 carrier_rows.is_delayed.mean(), (shared_rates[carrier] * common[carrier]).sum() / shared_n,
                 (shared_rates[carrier] * weights).sum()], rtol=1e-12)
    return full


def preceding_only(values, window, minimum, multiplier):
    """Recompute from explicit slices so the validation does not reuse the notebook's rolling function."""
    output = []
    for position, value in enumerate(values):
        past = values.iloc[max(0, position - window):position].dropna()
        if len(past) < minimum:
            output.append((np.nan, np.nan, np.nan, pd.NA))
            continue
        center = past.mean()
        spread = max(past.std(ddof=1), .02)
        threshold = center + multiplier * spread
        output.append((center, threshold, (value - center) / spread,
                       value > threshold if pd.notna(value) else pd.NA))
    result = pd.DataFrame(output, columns=['baseline', 'threshold', 'score', 'flag'], index=values.index)
    result['flag'] = result.flag.astype('boolean')
    return result


def check_patterns(run, daily, source_sha256):
    schema = {'cluster': 'Int64', 'delay_rate_flag': 'boolean', 'missing_rate_flag': 'boolean'}
    data = pd.read_csv(run / 'tables/daily_patterns.csv', index_col='date', dtype=schema, float_precision='round_trip')
    features = ['delay_rate', 'median_delay', 'p90_delay', 'flights']
    eligible = daily.observed_labels.ge(500) & daily.missing_rate.le(.5) & daily[features].notna().all(axis=1)
    expected = daily.drop(columns=['mean_delay', 'missing_labels']).assign(eligible=eligible)
    same(data[expected.columns], expected)
    same(read_table(run, 'pattern_daily_summary', 'date')[expected.columns], expected)
    assert data.cluster.notna().equals(eligible)
    assert data.index.equals(pd.date_range('2013-01-01', '2013-12-31', name='date').strftime('%Y-%m-%d'))

    selection = read_table(run, 'cluster_selection', 'k')
    assert list(selection.index) == list(range(2, 7))
    selected = int(selection.loc[selection.smallest_cluster.ge(10)].silhouette.idxmax())
    bundle = joblib.load(run / 'models/daily_clusters.joblib')
    model, scaler = bundle['kmeans'], bundle['scaler']
    config = json.loads((run / 'models/daily_patterns_config.json').read_text(encoding='utf-8'))
    assert config['source_sha256'] == source_sha256
    assert config['features'] == bundle['features'] == list(scaler.feature_names_in_) == features
    assert selected == model.n_clusters == config['selected_k']
    assert model.random_state == config['random_state'] == 42 and model.n_init == config['n_init'] == 20
    assert int(scaler.n_samples_seen_) == int(eligible.sum())
    np.testing.assert_allclose(scaler.mean_, data.loc[eligible, features].mean(), rtol=1e-12)
    np.testing.assert_array_equal(data.loc[eligible, 'cluster'], [bundle['ordered_cluster_ids'][x] for x in model.labels_])
    profiles = data.groupby('cluster').agg(days=('flights', 'size'), flights_mean=('flights', 'mean'),
                                          delay_rate_mean=('delay_rate', 'mean'), median_delay_mean=('median_delay', 'mean'),
                                          p90_delay_mean=('p90_delay', 'mean'), missing_rate_mean=('missing_rate', 'mean'))
    same(read_table(run, 'cluster_profiles', 'cluster'), profiles)
    assert profiles.delay_rate_mean.is_monotonic_increasing
    stability = read_table(run, 'cluster_stability', 'seed')
    assert list(stability.index) == list(range(5))

    delay = data.delay_rate.where(eligible)
    summaries = []
    primary = None
    for window, minimum, multiplier in [(28, 21, 2.), (28, 21, 3.), (56, 42, 2.)]:
        delayed = preceding_only(delay, window, minimum, multiplier)
        missing = preceding_only(data.missing_rate, window, minimum, multiplier)
        if primary is None:
            primary = delayed.flag
            for name, flags, metric in [('delay_anomalies', delayed, 'delay_rate'),
                                        ('missing_label_anomalies', missing, 'missing_rate')]:
                expected_flags = flags.add_prefix(metric + '_')
                same(data[expected_flags.columns], expected_flags)
                rows = data.loc[flags.flag.fillna(False)].sort_values(metric + '_score', ascending=False)
                saved = read_table(run, name, 'date').astype({key: dtype for key, dtype in schema.items()})
                same(saved, rows)
        common = primary.notna() & delayed.flag.notna()
        summaries.append([window, minimum, multiplier, int(delayed.flag.notna().sum()),
                          int(delayed.flag.sum()), int(missing.flag.sum()), int(common.sum()),
                          int(primary[common].sum()), int(delayed.flag[common].sum())])
    sensitivity = pd.read_csv(run / 'tables/anomaly_sensitivity.csv')
    np.testing.assert_array_equal(sensitivity.to_numpy(), summaries)
    assert data.loc[~eligible, 'delay_rate_flag'].isna().all()
    assert data[['delay_rate_flag', 'missing_rate_flag']].iloc[:21].isna().all().all()
    np.testing.assert_allclose(data.delay_excess_pp, (data.delay_rate - data.delay_rate_threshold) * 100, atol=1e-12)
    overlap = data.groupby('cluster').agg(days=('flights', 'size'), scorable_days=('delay_rate_flag', 'count'),
                                         delay_flags=('delay_rate_flag', 'sum'), missing_flags=('missing_rate_flag', 'sum'))
    same(read_table(run, 'cluster_anomaly_overlap', 'cluster'), overlap)
    cases = read_table(run, 'pattern_case_studies', 0).rename_axis('date')
    assert list(cases.index) == ['2013-03-08', '2013-07-01', '2013-07-23', '2013-02-09']
    same(cases.astype(schema), data.loc[cases.index, cases.columns])
    return {'daily_dates': len(data), 'eligible_dates': int(eligible.sum()), 'selected_k': selected,
            'cluster_sizes': [int(n) for n in profiles.days], 'min_adjusted_rand_index': float(stability.adjusted_rand_index.min()),
            'silhouette_k2': float(selection.loc[2, 'silhouette']), 'silhouette_k3': float(selection.loc[3, 'silhouette']),
            'delay_scorable_dates': int(data.delay_rate_flag.notna().sum()),
            'delay_flags': int(data.delay_rate_flag.sum()), 'missing_label_flags': int(data.missing_rate_flag.sum()),
            'past_only_windows_verified': [28, 56]}


def validate_exports(root, run, hashes):
    raw_manifest = json.loads((root / 'data/raw/source.json').read_text(encoding='utf-8'))
    assert hashes['data/raw/flights.csv'] == raw_manifest['sha256']
    fields = ['year', 'month', 'day', 'carrier', 'origin', 'dest', 'hour', 'dep_delay']
    raw = pd.read_csv(root / 'data/raw/flights.csv', usecols=['rownames', *fields]).set_index('rownames')
    flights = pd.read_csv(root / 'data/processed/flights_clean.csv', parse_dates=['date'])
    observed = flights.dep_delay.notna()
    assert (len(flights), int(observed.sum()), int((~observed).sum())) == (336776, 328521, 8255)
    assert raw.index.is_unique and flights.flight_id.is_unique
    same(flights.set_index('flight_id')[fields].sort_index(), raw[fields].sort_index().rename_axis('flight_id'))
    assert flights.date.eq(pd.to_datetime(flights[['year', 'month', 'day']])).all()
    assert flights.route.eq(flights.origin + '-' + flights.dest).all()
    assert flights.day_of_week.eq(flights.date.dt.dayofweek).all()
    assert flights.loc[~observed, 'delayed_15'].isna().all()
    assert flights.loc[observed, 'delayed_15'].eq(flights.loc[observed, 'dep_delay'].ge(15)).all()
    flights['is_delayed'] = flights.dep_delay.ge(15)
    groups = check_eda(run, flights)
    patterns = check_patterns(run, groups['daily'], hashes['data/processed/flights_clean.csv'])

    comparisons = []
    for path in sorted((run / 'reports').glob('*.md')):
        archived = root / 'reports' / path.name
        previous, generated = archived.read_text(encoding='utf-8'), path.read_text(encoding='utf-8')
        comparisons.append({'archive': archived.relative_to(root).as_posix(),
                            'regenerated': path.relative_to(run).as_posix(),
                            'byte_identical': archived.read_bytes() == path.read_bytes(),
                            'line_differences': list(unified_diff(previous.splitlines(), generated.splitlines(), lineterm=''))})
    delays = flights.dep_delay.dropna()
    return {
        'source_checks': {'flights': len(flights), 'observed_departure_labels': int(observed.sum()),
                          'unknown_departure_labels': int((~observed).sum()), 'unique_flight_ids': True,
                          'raw_values_reconciled_by_flight_id': True,
                          'early_departures': int(delays.lt(0).sum()), 'min_delay_minutes': float(delays.min()),
                          'max_delay_minutes': float(delays.max()), 'delays_outside_histogram_view': int((delays.lt(-60) | delays.gt(240)).sum()),
                          'observed_delay_rate': float(flights.is_delayed.sum() / observed.sum())},
        'eda_checks': {'grouped_counts_and_rates': True, 'mean_median_and_p90_include_observed_extremes': True,
                       'highest_month': int(groups['monthly'].delay_rate.idxmax()),
                       'highest_month_rate': float(groups['monthly'].delay_rate.max()),
                       'worst_eligible_day': read_table(run, 'worst_days', 'date').index[0]},
        'daily_patterns_checks': patterns,
        'schema_notes': {'tables/pattern_case_studies.csv': 'The first, unnamed CSV column is the date index.'},
        'archive_comparisons': comparisons,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path, help='existing run to validate without regenerating exports')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    provenance = json.loads((args.run / 'provenance.json').read_text(encoding='utf-8'))
    print(json.dumps(validate_exports(root, args.run, provenance['input_hashes']), indent=2, allow_nan=False))
