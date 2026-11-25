"""Read-only data contracts and calculations for the flight-delay dashboard."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
FLIGHTS = 'data/processed/flights_clean.csv'
RECOVERY = 'outputs/experiments/descriptive_recovery/descriptive_recovery_001/tables'
DAYS = f'{RECOVERY}/daily_patterns.csv'
PROFILES = f'{RECOVERY}/cluster_profiles.csv'
CASES = f'{RECOVERY}/pattern_case_studies.csv'
DELAY_EVENTS = f'{RECOVERY}/delay_anomalies.csv'
MISSING_EVENTS = f'{RECOVERY}/missing_label_anomalies.csv'
RESULTS = 'reports/current_results.json'
ORIGINAL = 'outputs/tables/prediction_comparison.csv'
SELECTION = 'outputs/models/prediction_selection.json'
ORIGINAL_AP = 'outputs/experiments/independent_audit/stage1_audit_001/historical_recomputed.csv'

NETWORK_SCOPE = 'Full NYC network · JFK, LGA and EWR'
FLIGHT_COLUMNS = ['date', 'carrier', 'origin', 'dep_delay']
DAILY_DTYPES = {'eligible': 'boolean', 'cluster': 'Int64', 'delay_rate_flag': 'boolean', 'missing_rate_flag': 'boolean'}
METRICS = {
    'f1': 'F1', 'roc_auc': 'ROC-AUC', 'precision': 'Precision', 'recall': 'Recall',
    'average_precision': 'Average precision', 'accuracy': 'Accuracy',
}
MODEL_NAMES = {
    'majority_baseline': 'Majority baseline',
    'logistic_regression': 'Logistic regression', 'decision_tree': 'Decision tree', 'random_forest': 'Random forest',
    'forest_interactions_d14': 'Interaction forest',
    'F': 'Audit F · compact logistic', 'G': 'Audit G · compact boosting',
    'decision_tree_02': 'Grid decision tree', 'fixed_compact_boosting': 'Fixed compact boosting',
    'schedule_hgb': 'Schedule-only boosting', 'grid_16': 'Operational boosting',
    'full_lr': 'Operational logistic regression', 'prior_service_average': 'Prior-service model average',
    'prior_service_deeper_hgb': 'Prior-service boosting', 'waiting_shallow_hgb': 'Waiting-feature boosting',
    'prior_service_lr': 'Prior-service logistic regression', 'waiting_rule': 'Waiting rule',
    'recent_arrivals_hgb': 'Recent-arrivals boosting',
}
ROLE_NAMES = {
    'frozen_model_selection': 'Selected on validation',
    'original_frozen_assessment': 'Validation-selected model',
    'original_comparison': 'Original comparison',
    'original_baseline': 'Majority reference',
    'improvement_internal_champion_validation_reused_for_cutoff': 'Internal champion · cutoff reused validation',
    'audit_compact_comparison': 'Audit comparison',
    'grid_development_rank_1': 'Selected on folds A–C',
    'predeclared_reference': 'Fixed reference',
    'schedule_candidate_selected_by_august_f1': 'Schedule-only reference · August F1',
    'selected_before_assessment': 'Selected before assessment',
    'highest_observed_historical_f1_among_carried_forward_candidates': 'Highest observed historical F1',
    'predeclared_status_rule': 'Simple status reference',
}
ORIGINAL_SPLITS = {'validation': ['2013-09', '2013-10'], 'test': ['2013-11', '2013-12']}


def artifact_path(root, relative):
    path = root / relative
    if path.is_file():
        return path
    if relative == FLIGHTS:
        remedy = 'Restore the prepared dataset or follow the data preparation instructions in README.md.'
    elif relative.startswith(RECOVERY):
        remedy = 'Restore the saved descriptive_recovery_001 exports listed in reports/artifact_audit.md.'
    else:
        remedy = 'Restore the saved research artifact at this path from the project outputs described in README.md.'
    raise FileNotFoundError(f'Missing `{relative}`. {remedy}')


def select_dates(frame, dates):
    """Inclusive scheduled departure dates, with an incomplete range left empty."""
    if len(dates) != 2:
        return frame.iloc[:0]
    start, end = map(pd.Timestamp, dates)
    return frame.loc[frame.date.between(start, end)]


def filter_flights(frame, dates, carrier=None, origin=None):
    filtered = select_dates(frame, dates)
    if carrier is not None:
        filtered = filtered.loc[filtered.carrier.eq(carrier)]
    if origin is not None:
        filtered = filtered.loc[filtered.origin.eq(origin)]
    return filtered


def network_days(frame, dates):
    """Saved city-wide assignments have one display filter: the date range."""
    return select_dates(frame, dates)


def flight_summary(frame):
    delays = frame.dep_delay.dropna()
    observed = len(delays)
    return {
        'flights': len(frame), 'observed': observed, 'unknown': len(frame) - observed,
        'delay_rate': float(delays.ge(15).mean()) if observed else None,
        'delay_statistics': {
            'Mean': delays.mean(), 'Median': delays.median(), '90th percentile': delays.quantile(.9),
            'Minimum': delays.min(), 'Maximum': delays.max(),
        },
    }


def grouped_reliability(frame, key):
    group = frame.assign(late=frame.dep_delay.ge(15)).groupby(key, observed=True)
    grouped = group.agg(flights=('dep_delay', 'size'), observed=('dep_delay', 'count'), delayed=('late', 'sum'))
    grouped['unknown'] = grouped.flights - grouped.observed
    grouped['delay_rate'] = grouped.delayed / grouped.observed.where(grouped.observed.ne(0))
    return grouped.reset_index()


def departure_histogram(frame, low=-60, high=240, step=10):
    observed = frame.dep_delay.dropna()
    visible = observed.loc[observed.between(low, high)]
    edges = list(range(low, high + 1, step))
    bins = pd.cut(visible, edges, include_lowest=True).value_counts(sort=False)
    return pd.DataFrame({'lower': edges[:-1], 'upper': edges[1:], 'flights': bins.to_numpy()}), len(observed) - len(visible)


def flag_labels(series):
    return series.map({True: 'Flagged', False: 'Clear'}).fillna('Unknown')


def format_period(period):
    if not period:
        return '—'
    first, last = period
    return first if first == last else f'{first} → {last}'


def original_results(comparison, selection, average_precision):
    """Join saved original rows to the audit AP by model and split."""
    chosen = selection.get('selected_model')
    ap_rows = {}
    if average_precision is not None:
        ap_rows = average_precision.set_index(['model', 'split']).average_precision.to_dict()
    results = []
    for row in comparison.to_dict('records'):
        model, split = row['model'], row['split']
        if split not in ORIGINAL_SPLITS:
            continue
        keys = {'model': model, 'split': split}
        sources = [{'file': ORIGINAL, 'row': keys}]
        if (model, split) in ap_rows:
            sources.append({'file': ORIGINAL_AP, 'row': keys, 'field': 'average_precision'})
        if model == chosen:
            role = 'frozen_model_selection' if split == 'validation' else 'original_frozen_assessment'
        else:
            role = 'original_baseline' if model == 'majority_baseline' else 'original_comparison'
        results.append({
            'model': model, 'role': role, 'prediction_time': 'Pre-departure schedule only',
            'assessment_period': ORIGINAL_SPLITS[split], 'fit_period': ['2013-01', '2013-08'],
            'threshold_selection_period': None if model == 'majority_baseline' else ORIGINAL_SPLITS['validation'],
            'population': 'all flights with an observed target label', 'n': row['n'], 'threshold': row['threshold'],
            'metrics': {**{name: row.get(name) for name in METRICS},
                        'accuracy': (row['tn'] + row['tp']) / row['n'] if row['n'] else None,
                        'average_precision': ap_rows.get((model, split))},
            'sources': sources,
        })
    return results


def model_table(rows):
    records = []
    for row in rows:
        source_row = row['sources'][0]['row']
        name = MODEL_NAMES.get(row['model'], row['model'])
        if 'fold' in source_row:
            name = f"{name} · fold {source_row['fold'].upper()}"
        subset = source_row.get('subset')
        population = 'Unresolved' if subset == 'unresolved' else 'All labeled'
        records.append({
            'Model': name, 'Assessment': format_period(row['assessment_period']), 'Population': population,
            **{label: row['metrics'].get(key) for key, label in METRICS.items()},
            'Role': ROLE_NAMES.get(row['role'], row['role']), 'Prediction time': row['prediction_time'],
            'n': row['n'], 'Cutoff': row['threshold'],
            'Fit': format_period(row['fit_period']), 'Cutoff selection': format_period(row['threshold_selection_period']),
        })
    return pd.DataFrame(records)


def source_table(rows):
    import json

    sources = {}
    for row in rows:
        for source in [*row['sources'], row.get('threshold_selection_source')]:
            if source:
                key = (source['file'], json.dumps(source['row'], ensure_ascii=False), source.get('field', ''))
                sources[key] = {'Saved file': key[0], 'Row key': key[1], 'Field': key[2]}
    return pd.DataFrame(sources.values())
