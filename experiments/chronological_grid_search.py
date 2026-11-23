"""Run the departure-delay development grid with python -m experiments.chronological_grid_search."""

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import platform
import time

import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from experiments.independent_audit import core as audit
from src.predict import ROOT, candidates as original_candidates, markdown_table


SELECTION_FOLDS = ('a', 'b', 'c')
COMPARISON_FOLD = 'd'
FIXED = 'fixed_compact_boosting'
ORIGINAL = 'original_logistic_compact'
BASELINE = 'majority_baseline'
REFERENCES = (FIXED, ORIGINAL, BASELINE)
THREADS = 4
METRICS = ['f1', 'precision', 'recall', 'average_precision', 'roc_auc']
NUMERIC, CATEGORICAL = audit.roles('F')
COMPACT = [column for column in audit.FEATURES if column in NUMERIC + CATEGORICAL]
GRIDS = {
    'logistic_regression': {'C': [0.1, 1, 10]},
    'decision_tree': {'max_depth': [6, 10, 14], 'min_samples_leaf': [100, 300]},
    'random_forest': {'max_depth': [10, 14, 18], 'min_samples_leaf': [50, 150],
                      'max_features': ['sqrt', 0.5]},
    'histogram_gradient_boosting': {'max_leaf_nodes': [15, 31], 'min_samples_leaf': [50, 100],
                                   'learning_rate': [0.05, 0.10], 'l2_regularization': [1, 10]},
}


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    settings: dict


def candidate_grid():
    fixed = audit.build('G').named_steps['model'].get_params()
    for family, grid in GRIDS.items():
        for index, settings in enumerate(ParameterGrid(grid), start=1):
            is_fixed = family == 'histogram_gradient_boosting' and all(
                fixed[key] == value for key, value in settings.items())
            name = FIXED if is_fixed else f'{family}_{index:02d}'
            yield Candidate(name, family, settings)
    yield Candidate(ORIGINAL, 'logistic_regression', {'C': 1.0})
    yield Candidate(BASELINE, 'majority', {})


def build(candidate):
    if candidate.family == 'histogram_gradient_boosting':
        return audit.build('G').set_params(**{f'model__{k}': v for k, v in candidate.settings.items()})

    pipeline = audit.build('F')
    if candidate.name == ORIGINAL:
        estimator = original_candidates()['logistic_regression']
    elif candidate.name == BASELINE:
        estimator = original_candidates()[BASELINE]
    elif candidate.family == 'decision_tree':
        estimator = DecisionTreeClassifier(random_state=audit.SEED, **candidate.settings)
    elif candidate.family == 'random_forest':
        estimator = RandomForestClassifier(n_estimators=100, n_jobs=THREADS,
                                           random_state=audit.SEED, **candidate.settings)
    else:
        return pipeline.set_params(model__C=candidate.settings['C'])
    return pipeline.set_params(model=estimator)


def fit_and_score(candidate, fold, periods):
    fit, tuning, assessment = periods
    model = build(candidate)
    start = time.monotonic()
    model.fit(fit[COMPACT], fit.delayed_15)
    fit_seconds = time.monotonic() - start

    tuning_scores = model.predict_proba(tuning[COMPACT])[:, list(model.classes_).index(1)]
    curve = audit.threshold_curve(tuning.delayed_15, tuning_scores)
    threshold = audit.choose(curve)
    curve = curve.assign(candidate=candidate.name, fold=fold, selected=curve.threshold.eq(threshold))
    scores = pd.Series(model.predict_proba(assessment[COMPACT])[:, list(model.classes_).index(1)],
                       index=assessment.index)
    settings = json.dumps(candidate.settings, sort_keys=True)
    common = {'candidate': candidate.name, 'family': candidate.family, 'settings': settings,
              'fold': fold, 'threshold': threshold, 'n_fit': len(fit), 'n_threshold': len(tuning),
              'fit_seconds': fit_seconds}
    assessment_metrics = [{'period': 'combined', **common,
                           **audit.metrics(assessment.delayed_15, scores, threshold)}]
    if assessment.month.nunique() > 1:
        assessment_metrics.extend(
            {'period': f'2013-{month:02d}', **common,
             **audit.metrics(group.delayed_15, scores.loc[group.index], threshold)}
            for month, group in assessment.groupby('month'))
    return assessment_metrics, curve


def rank_candidates(fold_results):
    combined = fold_results.loc[fold_results.period.eq('combined')]
    ranking = (combined.groupby(['candidate', 'family', 'settings'], as_index=False)[METRICS]
               .mean().rename(columns={name: f'mean_{name}' for name in METRICS})
               .sort_values(['mean_f1', 'candidate'], ascending=[False, True], ignore_index=True))
    ranking.insert(0, 'rank', ranking.index + 1)
    return ranking


def shortlisted(ranking):
    family_winners = ranking.loc[~ranking.candidate.isin([ORIGINAL, BASELINE])].drop_duplicates('family')
    return ranking.loc[ranking.candidate.isin([*family_winners.candidate, *REFERENCES])]


def winner_differences(results, winner):
    fields = ['threshold', *METRICS]
    reference = results.loc[results.candidate.eq(FIXED)].set_index(['fold', 'period'])[fields]
    selected = results.loc[results.candidate.eq(winner)].set_index(['fold', 'period'])[fields]
    return (selected - reference).add_prefix('delta_').reset_index()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def checkpoint_score(directory, candidate, fold, periods):
    path = directory / f'{fold}_{candidate.name}.json'
    if path.exists():
        saved = json.loads(path.read_text(encoding='utf-8'))
        scores, curve = saved['scores'], saved['curve']
        settings = json.dumps(candidate.settings, sort_keys=True)
        if (not scores or not curve
                or any(row.get('candidate') != candidate.name or row.get('fold') != fold
                       for row in scores + curve)
                or any(row.get('family') != candidate.family or row.get('settings') != settings
                       for row in scores)):
            raise ValueError(f'Checkpoint {path.name} does not match its fold and candidate. '
                             'Use --output with a new directory.')
        return scores, pd.DataFrame(curve)
    print(f'{fold}: {candidate.name} {candidate.settings}', flush=True)
    rows, curve = fit_and_score(candidate, fold, periods)
    rows = [{key: None if pd.isna(value) else value for key, value in row.items()} for row in rows]
    pending = path.with_suffix(f'.{os.getpid()}.tmp')
    write_json(pending, {'scores': rows, 'curve': curve.to_dict('records')})
    pending.replace(path)
    return rows, curve


def write_report(out, ranking, shortlist, selection, comparison, differences):
    winner = ranking.iloc[0]
    fixed = ranking.loc[ranking.candidate.eq(FIXED)].iloc[0]
    periods = pd.concat([selection, comparison], ignore_index=True)
    shown = periods.loc[periods.candidate.isin(shortlist.candidate)]
    columns = ['candidate', 'fold', 'period', 'threshold', *METRICS]
    report = [
        '# Chronological departure-delay grid search',
        f'**Winner** `{winner.candidate}` with `{winner.settings}`. Its mean assessment F1 was '
        f'{winner.mean_f1:.4f}, compared with {fixed.mean_f1:.4f} for the fixed compact boosting '
        f'reference ({winner.mean_f1 - fixed.mean_f1:+.4f}).',
        f'The grid scored {len(ranking)} scikit-learn candidates. '
        'Every model used the audit’s compact schedule features. The original logistic regression '
        'reference reuses the original estimator settings with compact preprocessing. The majority '
        'reference uses the audit grid and tie rule, which selects 0.500 for its constant negative prediction.',
        'Models and preprocessing were fitted through March, April, and May in folds A–C. '
        'The cutoff was selected in April, May, and June respectively, then frozen for May, June, '
        'and July–August assessment. The mean gives each fold equal weight. Setting ties are resolved '
        'alphabetically by candidate ID. The best grid setting per family and all references were '
        'saved before fold D. Their fold D models were fitted through July and their cutoffs chosen '
        'in August before scoring September–October. These previously examined periods provide '
        'development evidence only. The experiment excludes November–December.',
        '## Selection folds for shortlisted candidates',
        markdown_table(shown.loc[shown.fold.isin(SELECTION_FOLDS) & shown.period.eq('combined'), columns]),
        '## September–October comparison',
        markdown_table(comparison[columns]),
        '## Winner minus fixed compact boosting',
        markdown_table(differences),
        'The full assessment tables, cutoff curves, ranking, and frozen shortlist are saved with the run.',
    ]
    (out / 'report.md').write_text('\n\n'.join(report) + '\n', encoding='utf-8')


def run(source, out, resume=False, selection_fold=None):
    out.mkdir(parents=True, exist_ok=resume)
    with threadpool_limits(limits=THREADS):
        specs = {candidate.name: candidate for candidate in candidate_grid()}
        protocol = {
            'source': str(source), 'python': platform.python_version(), 'scikit_learn': sklearn.__version__,
            'pandas': pd.__version__, 'threads': THREADS,
            'features': COMPACT, 'categorical': CATEGORICAL, 'folds': audit.FOLDS,
            'selection_folds': SELECTION_FOLDS, 'comparison_fold': COMPARISON_FOLD,
            'threshold_grid': audit.GRID.tolist(), 'threshold_tie': 'closest to 0.5, then higher',
            'rank': 'descending unweighted mean A–C assessment F1, then candidate ID',
            'shortlist': 'best grid setting per family plus all references',
            'candidates': {name: {'family': spec.family, 'grid_settings': spec.settings,
                                  'parameters': build(spec).named_steps['model'].get_params()}
                           for name, spec in specs.items()},
        }
        protocol_path = out / 'protocol.json'
        checkpoints = out / 'candidate_checkpoints'
        saved_names = {path.name for path in checkpoints.glob('*.json')}
        if protocol_path.exists():
            previous = json.loads(protocol_path.read_text(encoding='utf-8'))
            if previous != json.loads(json.dumps(protocol)):
                raise ValueError('Cannot resume an incompatible grid protocol. Use --output with a new directory.')
        elif saved_names:
            raise ValueError('Cannot resume checkpoints without a saved protocol. Use --output with a new directory.')
        else:
            write_json(protocol_path, protocol)
        expected_names = {f'{fold}_{name}.json' for fold in (*SELECTION_FOLDS, COMPARISON_FOLD) for name in specs}
        unexpected_names = sorted(saved_names - expected_names)
        if unexpected_names:
            raise ValueError(f'Cannot resume unexpected candidate checkpoints: {", ".join(unexpected_names)}. '
                             'Use --output with a new directory.')
        checkpoints.mkdir(exist_ok=True)
        data = audit.load_development(source)
        rows, curves = [], []
        selection_folds = (selection_fold,) if selection_fold else SELECTION_FOLDS
        for fold in selection_folds:
            periods = audit.partitions(data, fold)
            for candidate in specs.values():
                scores, curve = checkpoint_score(checkpoints, candidate, fold, periods)
                rows.extend(scores)
                curves.append(curve)
        if selection_fold:
            print(f'Completed selection fold {selection_fold}: {len(specs)} candidates', flush=True)
            return
        selection = pd.DataFrame(rows)
        ranking = rank_candidates(selection)
        shortlist = shortlisted(ranking)
        selection.to_csv(out / 'selection_fold_metrics.csv', index=False)
        pd.concat(curves, ignore_index=True).to_csv(out / 'selection_threshold_curves.csv', index=False)
        ranking.to_csv(out / 'ranking.csv', index=False)
        write_json(out / 'shortlist.json', [
            {'candidate': name, 'rank': int(rank), 'mean_f1': float(mean_f1),
             'family': specs[name].family, 'settings': specs[name].settings,
             'parameters': build(specs[name]).named_steps['model'].get_params()}
            for name, rank, mean_f1 in shortlist[['candidate', 'rank', 'mean_f1']].itertuples(index=False, name=None)
        ])

        frozen = json.loads((out / 'shortlist.json').read_text(encoding='utf-8'))
        periods = audit.partitions(data, COMPARISON_FOLD)
        rows, curves = [], []
        for entry in frozen:
            candidate = Candidate(entry['candidate'], entry['family'], entry['settings'])
            scores, curve = checkpoint_score(checkpoints, candidate, COMPARISON_FOLD, periods)
            rows.extend(scores)
            curves.append(curve)
        comparison = pd.DataFrame(rows)
        comparison.to_csv(out / 'september_october_metrics.csv', index=False)
        pd.concat(curves, ignore_index=True).to_csv(out / 'august_threshold_curves.csv', index=False)

    differences = winner_differences(pd.concat([selection, comparison]), ranking.iloc[0].candidate)
    differences.to_csv(out / 'winner_vs_fixed_boosting.csv', index=False)
    write_report(out, ranking, shortlist, selection, comparison, differences)
    print(f'Winner: {ranking.iloc[0].candidate}. Results: {out}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'data/processed/classifier.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/experiments/chronological_grid_search')
    parser.add_argument('--resume', action='store_true', help='reuse completed candidate/fold checkpoints')
    parser.add_argument('--selection-fold', choices=SELECTION_FOLDS,
                        help='write checkpoints for one selection fold and exit')
    arguments = parser.parse_args()
    run(arguments.source, arguments.output, arguments.resume, arguments.selection_fold)
