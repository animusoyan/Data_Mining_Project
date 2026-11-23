"""Render a report and one timing chart from saved historical metrics."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from src.predict import markdown_table


TIMING_REFERENCE = {
    'September–October': [.8696, .8713, .8747, .8866, .9217, .9675],
    'November–December': [.8116, .8141, .8188, .8417, .8970, .9576],
}
LEADS = [60, 15, 5, 0, -5, -10]
METRIC_COLUMNS = ['checkpoint', 'period', 'id', 'subset', 'threshold', 'n', 'prevalence',
                  'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'average_precision',
                  'tn', 'fp', 'fn', 'tp']


def timing_chart(timing, output):
    fig, ax = plt.subplots(figsize=(8, 4.5), layout='constrained')
    for (period, group), color in zip(timing.groupby('period', sort=False), ['#006d77', '#ab471c']):
        ax.plot(group.lead_minutes, group.accuracy * 100, 'o-', color=color, label=period)
    ax.axvline(0, color='#6b7280', linewidth=1, linestyle=':')
    ax.set(xlabel='Minutes before scheduled departure (negative means after)',
           ylabel='Accuracy (%)', title='Historical departure-delay accuracy by checkpoint')
    ax.set_xticks(LEADS)
    ax.invert_xaxis()
    ax.grid(axis='y', alpha=.25)
    ax.legend(frameon=False)
    fig.savefig(output / 'departure_timing_accuracy.png', dpi=170)
    plt.close(fig)


def render(output):
    protocol = json.loads((output / 'protocol.json').read_text(encoding='utf-8'))
    rows = pd.read_csv(output / 'historical_metrics.csv')
    all_rows = rows.loc[rows.subset.eq('all')]
    timing = all_rows.loc[all_rows.experiment.eq('departure_accuracy') & all_rows.winner].copy()
    timing['reference_accuracy'] = [
        TIMING_REFERENCE[row.period][LEADS.index(row.lead_minutes)] for row in timing.itertuples()]
    timing['accuracy_difference'] = timing.accuracy - timing.reference_accuracy
    timing_chart(timing, output)
    departure = all_rows.loc[all_rows.experiment.eq('departure_f1')]
    sequence = ['always_on_time', 'schedule_lr', 'schedule_hgb', 'prior_day_and_traffic_hgb',
                'recent_reports_hgb', 'waiting_hgb', 'route_and_airline_hgb', 'prior_service_hgb']
    progression = departure.loc[departure.id.isin(sequence)]
    progression = progression.pivot(index='id', columns='period', values='f1').reindex(sequence).reset_index()
    progression.insert(0, 'checkpoint', '60 minutes before departure')
    progression['Sep–Oct F1 change vs prior row'] = progression['September–October'].diff()
    progression['Nov–Dec F1 change vs prior row'] = progression['November–December'].diff()
    reference = departure.loc[departure.id.eq('ensemble_reference_0265')].copy()
    reference['reference_f1'] = reference.period.map({'September–October': .4171, 'November–December': .5034})
    reference['f1_difference'] = reference.f1 - reference.reference_f1
    reference['reference_accuracy'] = reference.period.map({'November–December': .7651})
    reference['accuracy_difference'] = reference.accuracy - reference.reference_accuracy
    arrival = rows.loc[rows.experiment.eq('arrival_accuracy')]
    arrival_reference = arrival.loc[arrival.id.eq('arrival_reference_0510')].copy()
    arrival_reference['reference_accuracy'] = arrival_reference.period.map(
        {'September–October': .9397, 'November–December': .9069})
    arrival_reference.loc[arrival_reference.subset.eq('unresolved'), 'reference_accuracy'] = float('nan')
    late_arrival = arrival_reference.period.eq('November–December') & arrival_reference.subset.eq('unresolved')
    arrival_reference.loc[late_arrival, 'reference_accuracy'] = .791
    arrival_reference['accuracy_difference'] = arrival_reference.accuracy - arrival_reference.reference_accuracy
    november_arrival = arrival_reference.loc[arrival_reference.period.eq('November–December')
                                            & arrival_reference.subset.eq('all')].iloc[0]
    rule_agreement = pd.concat([pd.read_csv(output / f'waiting_rule_agreement_{lead}.csv')
                                for lead in (5, 10)], ignore_index=True)
    report = [
        '# NYC flight-delay historical replays',
        'The two assessment periods are September–October and November–December. '
        'Both periods have been examined previously. The comparison uses observed labels for the '
        'relevant target and retains early flights. Training and cutoff selection use earlier months.',
        f'The snapshot contains {protocol["rows"]:,} flights, with '
        f'{protocol["observed_departure_labels"]:,} known departure labels and '
        f'{protocol["observed_arrival_labels"]:,} known arrival labels. '
        f'There are {protocol["early_departures"]:,} early departures and '
        f'{protocol["early_arrivals"]:,} early arrivals.',
        '## Before the departure day',
        markdown_table(all_rows.loc[all_rows.experiment.eq('before_day'), METRIC_COLUMNS]),
        '## One hour before departure',
        'The boosting configuration is held fixed across the feature additions. The first logistic '
        'regression row supplies a separate schedule-only reference.',
        markdown_table(progression),
        'The 0.265 ensemble is the named operating-point reconstruction. The full CSV also includes '
        'the tuned ensemble, the highest-ranked shallow-grid candidate, regression, MLP, and other models.',
        markdown_table(reference[['checkpoint', 'period', 'threshold', 'f1', 'reference_f1',
                                  'f1_difference', 'accuracy', 'reference_accuracy', 'accuracy_difference']]),
        markdown_table(departure.loc[departure.winner
                                     | departure.id.isin(['schedule_mlp', 'schedule_regression']), METRIC_COLUMNS]),
        '## Accuracy across departure checkpoints',
        'At ten minutes after scheduled departure, only five minutes remain until the target delay '
        'threshold. Confirmed departures below 15 minutes resolve as negative once reported.',
        '![Accuracy by departure checkpoint](departure_timing_accuracy.png)',
        markdown_table(timing[['checkpoint', 'lead_minutes', 'period', 'id', 'accuracy',
                               'reference_accuracy', 'accuracy_difference']]),
        markdown_table(timing[METRIC_COLUMNS]),
        'The table counts rows where the selected model and the waiting-only rule made different decisions.',
        markdown_table(rule_agreement),
        '## Arrival delay at scheduled arrival',
        'Arrival status and confirmed departure status are evaluated at the destination-local scheduled '
        'arrival converted to UTC. The unresolved subset isolates the flights where the model is used.',
        markdown_table(arrival.loc[arrival.winner | arrival.model.str.endswith('_rule'), METRIC_COLUMNS]),
        markdown_table(arrival_reference[['checkpoint', 'period', 'subset', 'n', 'accuracy',
                                          'reference_accuracy', 'accuracy_difference', 'precision',
                                          'recall', 'f1']]),
        'The supplied November–December 0.510 reference had precision 0.858, recall 0.767, '
        f'and F1 0.810. This replay measured {november_arrival.precision:.4f}, '
        f'{november_arrival.recall:.4f}, and {november_arrival.f1:.4f} at that cutoff.',
        '## Replay assumptions',
        'Historical departure and arrival reports are assumed available at their reconstructed event '
        'times. Schedule dates use the NYC departure calendar. Eligible aircraft rotation pairs '
        'come from completed prior days. Historical tail numbers establish service relationships, '
        'and the corresponding service on the flight day is found by its published identity. '
        'Observed arrival clocks locate events in the replay. The unresolved arrival model uses only '
        'the schedule, known departure facts, and other flights’ already available arrival reports.',
        'Origin history uses a seven-day window ending before the scheduled day or at the prediction '
        'cutoff, whichever comes first. Route and carrier history use 14 days at the same boundary. '
        'Recent origin departures use 30-, 60-, and 120-minute windows, and waiting counts '
        'use the previous 120 minutes of scheduled flights. Recent arrivals use 60- and 180-minute '
        'windows. The boosted-delay regressor converts minutes to a monotonic screening score around '
        '15 minutes so it can share the classification tables and threshold selection.',
        'The earlier reference settings and feature definitions were supplied as summary values. '
        'The results here use the feature windows, model defaults, and cutoff grids recorded for this replay.',
        'Complete candidate results, confusion counts, and selected cutoffs are in historical_metrics.csv. '
        'The cutoff-selection results include every shallow-grid candidate.',
    ]
    (output / 'report.md').write_text('\n\n'.join(report) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    render(parser.parse_args().output)
