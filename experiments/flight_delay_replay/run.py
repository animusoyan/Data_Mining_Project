"""Run all three historical replay studies against the cached NYC Flights snapshot."""

import argparse
import json
from pathlib import Path
import platform

import pandas as pd
import sklearn
from sklearn.model_selection import ParameterGrid
from threadpoolctl import threadpool_limits

from src.predict import ROOT

from .evaluation import ACCURACY_GRID, F1_GRID, FIXED_BOOSTING, Comparison, Spec, departure_resolved
from .features import (arrival_features, as_of, departure_features, event_times, load_snapshot,
                       rotation_matches, schedule_times)
from .report import render


CHECKPOINTS = {60: '60 minutes before departure', 15: '15 minutes before departure',
               5: '5 minutes before departure', 0: 'Scheduled departure',
               -5: '5 minutes after scheduled departure', -10: '10 minutes after scheduled departure'}
TIMING_BLOCKS = ['waiting', 'prior_service_compact', 'prior_service']
SHALLOW_GRID = {'max_leaf_nodes': [15, 31], 'min_samples_leaf': [50, 100],
                'learning_rate': [.05, .1], 'l2_regularization': [1, 10]}
ALWAYS_ON_TIME = Spec('always_on_time', 'none', 'always_on_time', cutoff=.5)
WAITING_RULE = Spec('waiting_rule', 'none', 'waiting_rule', cutoff=.5)


def departure_comparison(name, checkpoint, lead, flights, schedule, events, output, rotations,
                         metric, fit_end, status_aware):
    cutoff = schedule.midnight if lead is None else schedule.departure - lead
    features, blocks = departure_features(flights, schedule, events, cutoff, rotations)
    service = [column for column in blocks['prior_service'] if column.startswith('prior_service_')]
    blocks['prior_service_compact'] = blocks['waiting'] + service
    resolved = departure_resolved(flights, events, cutoff) if status_aware else cutoff.ne(cutoff)
    engine = Comparison(name, checkpoint, lead, flights, output, cutoff, resolved, flights.dep_delay,
                        features, blocks, metric, fit_end, rule=~as_of(events.departed, cutoff))
    engine.check_label_reporting(events.departed)
    return engine


def f1_specs():
    specs = [ALWAYS_ON_TIME, Spec('schedule_lr', 'schedule', 'lr')]
    specs.extend(Spec(f'{block}_hgb', block, 'hgb') for block in
                 ['schedule', 'prior_day_and_traffic', 'recent_reports', 'waiting',
                  'route_and_airline', 'prior_service'])
    specs.extend(Spec(f'full_{model}', 'prior_service', model) for model in ['lr', 'rf', 'mlp', 'regression'])
    specs.extend([Spec('ensemble_tuned', 'prior_service', 'rf+hgb'),
                  Spec('ensemble_reference_0265', 'prior_service', 'rf+hgb', cutoff=.265),
                  Spec('schedule_mlp', 'schedule', 'mlp'),
                  Spec('schedule_regression', 'schedule', 'regression')])
    specs.extend(Spec(f'grid_{i:02d}', 'prior_service', 'hgb', params)
                 for i, params in enumerate(ParameterGrid(SHALLOW_GRID), 1))
    return specs


def timing_specs(lead):
    specs = [ALWAYS_ON_TIME]
    for block in TIMING_BLOCKS:
        specs.extend([Spec(f'{block}_lr', block, 'lr'),
                      Spec(f'{block}_shallow_hgb', block, 'hgb'),
                      Spec(f'{block}_deeper_hgb', block, 'hgb', {'max_leaf_nodes': 31}),
                      Spec(f'{block}_average', block, 'lr+hgb')])
    if lead < 0:
        specs.append(WAITING_RULE)
    return specs


def run(root, output):
    flights, zones, checksum = load_snapshot(root)
    output.mkdir(parents=True, exist_ok=False)
    schedule = schedule_times(flights, zones)
    events = event_times(flights, schedule, zones)
    rotations = rotation_matches(flights, schedule, events)
    data = {
        'source_sha256': checksum, 'python': platform.python_version(), 'sklearn': sklearn.__version__,
        'rows': len(flights), 'observed_departure_labels': int(flights.dep_delay.notna().sum()),
        'observed_arrival_labels': int(flights.arr_delay.notna().sum()),
        'early_departures': int(flights.dep_delay.lt(0).sum()), 'early_arrivals': int(flights.arr_delay.lt(0).sum()),
        'one_hour_fit_through_month': 7, 'accuracy_and_arrival_fit_through_month': 6,
        'f1_cutoffs': F1_GRID.tolist(), 'accuracy_cutoffs': ACCURACY_GRID.tolist(),
        'fixed_boosting': FIXED_BOOSTING, 'shallow_grid': SHALLOW_GRID,
        'checkpoints': CHECKPOINTS, 'arrival_cutoff': 'scheduled arrival in destination local time, converted to UTC',
        'rollover': 'earliest destination-local occurrence strictly after scheduled departure',
        'reports': 'assumed available when the local actual clock event occurs',
        'arrival_near_midnight_note': 'period keys use the NYC scheduled departure calendar',
        'threads': 4,
    }
    (output / 'protocol.json').write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    metrics, selections = [], []

    def collect(engine, specs, **kwargs):
        rows, selection = engine.run(specs, **kwargs)
        metrics.extend(rows)
        selections.append(selection)
        pd.DataFrame(metrics).to_csv(output / 'historical_metrics.csv', index=False)
        pd.concat(selections, ignore_index=True).to_csv(output / 'cutoff_selection.csv', index=False)
        return selection

    with threadpool_limits(limits=4):
        preday = departure_comparison('before_day', 'Start of scheduled departure day', None, flights,
                                      schedule, events, output, rotations, 'f1', 7, False)
        collect(preday, [ALWAYS_ON_TIME, Spec('schedule_lr', 'schedule', 'lr'),
                         Spec('schedule_hgb', 'schedule', 'hgb'),
                         Spec('prior_day_and_traffic_hgb', 'prior_day_and_traffic', 'hgb')])
        del preday
        hourly = departure_comparison('departure_f1', CHECKPOINTS[60], 60, flights, schedule, events,
                                      output, rotations, 'f1', 7, False)
        collect(hourly, f1_specs(), grid_prefix='grid_')
        del hourly
        for lead, checkpoint in CHECKPOINTS.items():
            timing = departure_comparison('departure_accuracy', checkpoint, lead, flights, schedule,
                                          events, output, rotations, 'accuracy', 6, True)
            specs = timing_specs(lead)
            selection = collect(timing, specs)
            if lead < 0:
                winner = selection.loc[selection.winner].iloc[0]
                spec = next(spec for spec in specs if spec.name == winner.id)
                agreement = []
                for period, mask in timing.assessments.items():
                    predicted = timing.score(spec, mask) >= winner.cutoff
                    waiting = timing.score(WAITING_RULE, mask) >= WAITING_RULE.cutoff
                    agreement.append({'checkpoint': checkpoint, 'lead_minutes': lead, 'period': period,
                                      'model': spec.name, 'n': len(predicted),
                                      'different_decisions': int((predicted != waiting).sum())})
                pd.DataFrame(agreement).to_csv(output / f'waiting_rule_agreement_{abs(lead)}.csv', index=False)
            del timing

        features, blocks = arrival_features(flights, schedule, events)
        arrived = as_of(events.arrived, schedule.arrival)
        departed = as_of(events.departed, schedule.arrival)
        late_departure_rule = ~arrived & (~departed | flights.dep_delay.ge(15))
        arrival = Comparison('arrival_accuracy', 'Scheduled arrival', 0, flights, output, schedule.arrival,
                             arrived, flights.arr_delay, features, blocks, 'accuracy', 6,
                             extra_rules={'late_departure_rule': late_departure_rule})
        arrival.check_label_reporting(events.arrived)
        arrival_specs = [WAITING_RULE, Spec('late_departure_rule', 'none', 'late_departure_rule', cutoff=.5),
                         Spec('arrival_reference_0510', 'recent_arrivals', 'hgb', cutoff=.510)]
        arrival_specs.extend(Spec(f'{block}_{model}', block, model)
                             for block in blocks for model in ['lr', 'hgb'])
        collect(arrival, arrival_specs)
    render(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/experiments/flight_delay_replay')
    args = parser.parse_args()
    run(ROOT, args.output)
