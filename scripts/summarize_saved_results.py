"""Build the current results JSON from saved research tables without model execution."""

import argparse
import csv
from itertools import product
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = Path('outputs')
EXPERIMENTS = OUTPUTS / 'experiments'
ORIGINAL = OUTPUTS / 'tables/prediction_comparison.csv'
IMPROVEMENT = EXPERIMENTS / 'modeling_study'
AUDIT = EXPERIMENTS / 'independent_audit/stage1_audit_001'
GRID = EXPERIMENTS / 'chronological_grid_search'
REPLAY = EXPERIMENTS / 'flight_delay_replay'
SUMMARY = ROOT / 'reports/current_results.json'
METRICS = ('accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'average_precision')
ALL = 'all flights with an observed target label'
PERIODS = {'September–October': (9, 10), 'November–December': (11, 12)}
REPLAY_GROUPS = {'before_day': 'schedule_development', 'departure_f1': 'operational_one_hour',
                 'departure_accuracy': 'status_assisted_departure', 'arrival_accuracy': 'scheduled_arrival'}


def read_rows(path):
    with (ROOT / path).open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def row_for(rows, **keys):
    matches = [row for row in rows if all(row[key] == str(value) for key, value in keys.items())]
    if len(matches) != 1:
        raise ValueError(f'Expected one saved row for {keys}, found {len(matches)}')
    return matches[0]


def months(first, last=None):
    return [f'2013-{first:02d}', f'2013-{(last or first):02d}']


def result(group, path, row, keys, *, role, model, prediction_time, fit, tuning, assessment,
           population=ALL, target='departure_delay_minutes >= 15', supplement=None):
    source = {'file': path.as_posix(), 'row': {key: row[key] for key in keys}}
    values = row if supplement is None else {**supplement[1], **row}
    metrics = {name: float(values[name]) if values.get(name) else None for name in METRICS}
    counts = {name: int(row[name]) for name in ('tn', 'fp', 'fn', 'tp')}
    n = int(row['n'])
    metrics['accuracy'] = (counts['tn'] + counts['tp']) / n
    sources = [source]
    if supplement is not None:
        filespec, _, field = supplement
        sources.append({'file': filespec.as_posix(), 'row': source['row'], 'field': field})
    return {
        'group': group,
        'role': role,
        'model': model,
        'target': target,
        'prediction_time': prediction_time,
        'fit_period': months(*fit),
        'threshold_selection_period': months(*tuning),
        'assessment_period': months(*assessment),
        'population': population,
        'n': n,
        'threshold': float(row['threshold']),
        'metrics': metrics,
        'confusion_counts': counts,
        'sources': sources,
    }


def original_results():
    model = read_json(OUTPUTS / 'models/prediction_selection.json')['selected_model']
    original = read_rows(ORIGINAL)
    ap_path = AUDIT / 'historical_recomputed.csv'
    audit_ap = read_rows(ap_path)
    for split, period, role in [('validation', (9, 10), 'frozen_model_selection'),
                                ('test', (11, 12), 'original_frozen_assessment')]:
        keys = {'model': model, 'split': split}
        yield result('original_historical', ORIGINAL, row_for(original, **keys), keys,
                     role=role, model=model, prediction_time='Pre-departure schedule only, no fixed lead time',
                     fit=(1, 8), tuning=(9, 10), assessment=period,
                     supplement=(ap_path, row_for(audit_ap, **keys), 'average_precision'))


def schedule_results(grid_winner):
    time = 'Pre-departure schedule only, no fixed lead time'
    shortlist = read_json(IMPROVEMENT / 'shortlist.json')
    path = IMPROVEMENT / 'validation_metrics.csv'
    model = shortlist['internal_champion']
    keys = {'candidate': model}
    yield result('schedule_development', path, row_for(read_rows(path), **keys), keys,
                 role='improvement_internal_champion_validation_reused_for_cutoff',
                 model=model, prediction_time=time, fit=(1, 8), tuning=(9, 10), assessment=(9, 10))

    path = AUDIT / 'fold_metrics.csv'
    audit_rows = read_rows(path)
    for model in ('F', 'G'):
        keys = {'candidate': model, 'fold': 'd', 'grid': 'primary'}
        yield result('schedule_development', path, row_for(audit_rows, **keys), keys,
                     role='audit_compact_comparison', model=model, prediction_time=time,
                     fit=(1, 7), tuning=(8,), assessment=(9, 10))

    folds = read_json(GRID / 'protocol.json')['folds']
    for fold, model, role in [
        *((fold, grid_winner, 'grid_development_rank_1') for fold in ('a', 'b', 'c', 'd')),
        ('d', 'fixed_compact_boosting', 'predeclared_reference'),
    ]:
        path = GRID / ('september_october_metrics.csv' if fold == 'd' else 'selection_fold_metrics.csv')
        keys = {'candidate': model, 'fold': fold, 'period': 'combined'}
        fit_end, tune, assessment = folds[fold]
        yield result('schedule_development', path, row_for(read_rows(path), **keys), keys,
                     role=role, model=model, prediction_time=time, fit=(1, fit_end),
                     tuning=(tune,), assessment=(assessment[0], assessment[-1]))


def replay_results():
    path = REPLAY / 'historical_metrics.csv'
    assessment_rows = read_rows(path)
    selection_path = REPLAY / 'cutoff_selection.csv'
    selection = read_rows(selection_path)
    checkpoints = dict.fromkeys((row['experiment'], row['checkpoint']) for row in selection)
    for experiment, time in checkpoints:
        tuning_rows = [row for row in selection if (row['experiment'], row['checkpoint']) == (experiment, time)]
        chosen = [(row_for(tuning_rows, winner='True')['id'], 'selected_before_assessment')]
        if experiment in ('before_day', 'departure_f1'):
            schedule = max((row for row in tuning_rows if row['features'] == 'schedule'),
                           key=lambda row: (float(row['f1']), row['id']))['id']
            chosen = [(schedule, 'schedule_candidate_selected_by_august_f1')]
            if experiment == 'departure_f1':
                chosen.extend([
                    (row_for(tuning_rows, winner='True')['id'], 'selected_before_assessment'),
                    ('full_lr', 'highest_observed_historical_f1_among_carried_forward_candidates'),
                ])
        if experiment == 'departure_accuracy' and time.startswith('10 minutes after'):
            chosen.append(('waiting_rule', 'predeclared_status_rule'))

        for (model, role), (period, assessment) in product(chosen, PERIODS.items()):
            status_aware = experiment.endswith('_accuracy')
            for subset in ('all', 'unresolved') if status_aware else ('all',):
                keys = {'experiment': experiment, 'checkpoint': time, 'id': model,
                        'period': period, 'subset': subset}
                row = row_for(assessment_rows, **keys)
                fit_end = int(row_for(tuning_rows, id=model)['fit_through_month'])
                arrival = experiment == 'arrival_accuracy'
                if subset == 'all':
                    population = ALL
                elif arrival:
                    population = 'labeled flights without a reported arrival by the checkpoint'
                else:
                    population = 'labeled flights with no reported departure under 15 minutes late by the checkpoint'
                if not status_aware:
                    population += ' (also the unresolved population in this replay)'
                record = result(REPLAY_GROUPS[experiment], path, row, keys, role=role, model=model,
                                prediction_time=time, fit=(1, fit_end), tuning=(fit_end + 1, 8),
                                assessment=assessment, population=population,
                                target='arrival_delay_minutes >= 15' if arrival else 'departure_delay_minutes >= 15')
                if model == 'waiting_rule':
                    record.update(fit_period=None, threshold_selection_period=None)
                else:
                    record['threshold_selection_source'] = {
                        'file': selection_path.as_posix(),
                        'row': {'experiment': experiment, 'checkpoint': time, 'id': model},
                    }
                yield record


def summary():
    ranking_path = GRID / 'ranking.csv'
    winner = row_for(read_rows(ranking_path), rank=1)
    agreement_path = REPLAY / 'waiting_rule_agreement_10.csv'
    return {
        'period_format': 'Inclusive first and last calendar month in YYYY-MM',
        'later_assessment_context': 'Exploratory historical comparisons on previously examined periods',
        'grid_selection': {
            'source': {'file': ranking_path.as_posix(), 'row': {'rank': 1, 'candidate': winner['candidate']}},
            'mean_f1': float(winner['mean_f1']),
            'folds': ['a', 'b', 'c'],
        },
        'results': [*original_results(), *schedule_results(winner['candidate']), *replay_results()],
        'waiting_rule_agreement_at_plus_10_minutes': [
            {'source': {'file': agreement_path.as_posix(), 'row': {'period': row['period'], 'model': row['model']}},
             'n': int(row['n']), 'different_decisions': int(row['different_decisions'])}
            for row in read_rows(agreement_path)
        ],
        'historical_fixed_references': [
            {'file': (REPLAY / 'historical_metrics.csv').as_posix(),
             'row_filter': {'experiment': experiment, 'id': model},
             'threshold': threshold, 'original_selection': 'undocumented'}
            for experiment, model, threshold in [
                ('departure_f1', 'ensemble_reference_0265', 0.265),
                ('arrival_accuracy', 'arrival_reference_0510', 0.510),
            ]
        ],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='check the tracked JSON against saved tables')
    args = parser.parse_args()
    output = json.dumps(summary(), indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    if args.check:
        if SUMMARY.read_text(encoding='utf-8') != output:
            raise SystemExit('Current results JSON differs from the saved research tables')
        print('Current results JSON matches the saved research tables.')
    else:
        print(output, end='')
