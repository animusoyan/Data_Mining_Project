"""Chronological pre-departure classification. Run once with python -m src.predict."""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.prepare import FEATURES

ROOT = Path(__file__).resolve().parents[1]
CATEGORICAL = ['carrier', 'flight', 'origin', 'dest']
NUMERIC = [name for name in FEATURES if name not in CATEGORICAL]
THRESHOLDS = np.round(np.arange(.10, .501, .025), 3)


def validate_data(frame):
    """Reject schema drift, invalid labels and incorrectly assigned temporal splits."""
    expected = {'flight_id', 'date', 'split', 'delayed_15', *FEATURES}
    if set(frame.columns) != expected:
        raise ValueError('Classifier schema must match the approved feature allowlist and metadata')
    dates = pd.to_datetime(frame.date, errors='raise')
    if dates.isna().any() or not dates.dt.year.eq(2013).all():
        raise ValueError('Expected valid 2013 dates')
    if not frame.month.eq(dates.dt.month).all():
        raise ValueError('Feature month disagrees with date')
    expected_split = dates.dt.month.map(lambda m: 'train' if m <= 8 else 'validation' if m <= 10 else 'test')
    if not frame.split.eq(expected_split).all():
        raise ValueError('Chronological split differs from Jan-Aug / Sep-Oct / Nov-Dec')
    if frame.flight_id.isna().any() or not frame.flight_id.is_unique:
        raise ValueError('Flight IDs must be present and unique')
    if not frame.delayed_15.isin([0, 1]).all():
        raise ValueError('Classification requires observed binary labels')
    if set(frame.split) != {'train', 'validation', 'test'}:
        raise ValueError('All three partitions are required')


def build_pipeline(estimator):
    numeric = Pipeline([('impute', SimpleImputer(strategy='median')),
                        ('scale', StandardScaler())])
    categorical = Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                            ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=True))])
    preprocessing = ColumnTransformer([('numeric', numeric, NUMERIC),
                                       ('categorical', categorical, CATEGORICAL)], remainder='drop')
    return Pipeline([('preprocess', preprocessing), ('model', estimator)])


def candidates():
    return {
        'majority_baseline': DummyClassifier(strategy='most_frequent'),
        'logistic_regression': LogisticRegression(solver='liblinear', C=1., max_iter=500, random_state=42),
        'decision_tree': DecisionTreeClassifier(max_depth=12, min_samples_leaf=100, random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=100, max_depth=18,
            min_samples_leaf=30, max_features='sqrt', n_jobs=2, random_state=42),
    }


def positive_probability(pipeline, frame):
    """Pass only approved predictors, regardless of additional inference metadata."""
    classes = pipeline.named_steps['model'].classes_
    return pipeline.predict_proba(frame.loc[:, FEATURES])[:, list(classes).index(1)]


def evaluate(y, probability, threshold):
    y = np.asarray(y, dtype=int)
    predicted = np.asarray(probability) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    precision = float(tp / (tp + fp)) if tp + fp else np.nan
    recall = float(tp / (tp + fn)) if tp + fn else np.nan
    f1 = float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else np.nan
    return {'n': len(y), 'negative_count': int((y == 0).sum()), 'positive_count': int(y.sum()),
            'prevalence': float(y.mean()) if len(y) else np.nan,
            'predicted_positive_count': int(predicted.sum()), 'precision': precision,
            'recall': recall, 'f1': f1,
            'roc_auc': float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else np.nan,
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}


def select_threshold(y, probability):
    """Fixed modest grid; ties prefer the threshold closest to 0.5, then higher."""
    rows = [{'threshold': float(t), 'f1': f1_score(y, probability >= t, zero_division=0)} for t in THRESHOLDS]
    chosen = max(rows, key=lambda row: (row['f1'], -abs(row['threshold'] - .5), row['threshold']))
    return chosen['threshold'], rows


def markdown_table(frame):
    # No optional tabulate dependency required.
    def value(v):
        if pd.isna(v): return 'undefined'
        if isinstance(v, (float, np.floating)): return f'{v:.4f}'
        return str(v)
    return '\n'.join(['| ' + ' | '.join(frame.columns) + ' |',
                      '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |'] +
                     ['| ' + ' | '.join(value(v) for v in row) + ' |' for row in frame.itertuples(index=False, name=None)])


def run(root=ROOT):
    root = Path(root)
    tables, models = root/'outputs/tables', root/'outputs/models'
    tables.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    lock = models/'prediction_selection.json'
    # Protect the first holdout assessment from accidental repeated training/evaluation.
    if lock.exists():
        raise FileExistsError('Prediction selection already exists; inspect saved results instead of rerunning the holdout.')
    source = root/'data/processed/classifier.csv'
    data = pd.read_csv(source, parse_dates=['date'])
    validate_data(data)
    train = data.loc[data.split.eq('train')]
    validation = data.loc[data.split.eq('validation')]
    fitted, thresholds, comparison, monthly, predictions, search = {}, {}, [], [], [], []
    for name, estimator in candidates().items():
        print('Training', name, flush=True)
        start = time.monotonic()
        pipeline = build_pipeline(estimator)
        pipeline.fit(train[FEATURES], train.delayed_15)
        probability = positive_probability(pipeline, validation)
        threshold, rows = (.5, []) if name == 'majority_baseline' else select_threshold(validation.delayed_15, probability)
        search.extend({'model': name, **row} for row in rows)
        fitted[name], thresholds[name] = pipeline, threshold
        comparison.append({'model': name, 'split': 'validation', 'threshold': threshold,
                           'fit_seconds': time.monotonic() - start,
                           **evaluate(validation.delayed_15, probability, threshold)})
        predictions.append(prediction_frame(name, validation, probability, threshold))
        monthly.extend(monthly_metrics(name, validation, probability, threshold))
        print(name, comparison[-1], flush=True)
    validation_metrics = pd.DataFrame(comparison)
    selected = validation_metrics.sort_values(['f1', 'model'], ascending=[False, True]).iloc[0].model
    metadata = {'selected_model': selected, 'threshold': thresholds[selected],
        'thresholds': thresholds, 'features': FEATURES, 'categorical_features': CATEGORICAL,
        'numeric_features': NUMERIC, 'target': 'delayed_15', 'positive_class': 1,
        'target_definition': 'departure delay >=15 minutes; unknown labels excluded',
        'split': {'train': '2013-01 through 2013-08', 'validation': '2013-09 through 2013-10', 'test': '2013-11 through 2013-12'},
        'selection_metric': 'validation F1', 'model_tie_break': 'alphabetical model name',
        'threshold_grid': THRESHOLDS.tolist(), 'threshold_tie_break': 'closest to 0.5, then higher',
        'fit_scope': 'January-August only; no refit on validation',
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'python_version': platform.python_version(), 'sklearn_version': sklearn.__version__,
        'pandas_version': pd.__version__, 'numpy_version': np.__version__,
        'parameters': {name: estimator.get_params() for name, estimator in candidates().items()},
        'test_policy': 'All four models and thresholds frozen before one test comparison; winner selected exclusively on validation'}
    # Persist the decision BEFORE computing any test predictions or test metrics.
    lock.write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    validation_metrics.to_csv(tables/'prediction_validation.csv', index=False)
    pd.DataFrame(search).to_csv(tables/'prediction_threshold_search.csv', index=False)
    for name, pipeline in fitted.items():
        joblib.dump(pipeline, models/f'prediction_{name}.joblib')
    joblib.dump(fitted[selected], models/'flight_delay_pipeline.joblib')
    print('Frozen selection:', selected, thresholds[selected], flush=True)
    test = data.loc[data.split.eq('test')]
    for name, pipeline in fitted.items():
        probability = positive_probability(pipeline, test)
        comparison.append({'model': name, 'split': 'test', 'threshold': thresholds[name],
                           'fit_seconds': np.nan, **evaluate(test.delayed_15, probability, thresholds[name])})
        predictions.append(prediction_frame(name, test, probability, thresholds[name]))
        monthly.extend(monthly_metrics(name, test, probability, thresholds[name]))
    metrics = pd.DataFrame(comparison)
    metrics.to_csv(tables/'prediction_comparison.csv', index=False)
    pd.DataFrame(monthly).to_csv(tables/'prediction_monthly.csv', index=False)
    pd.concat(predictions, ignore_index=True).to_csv(tables/'prediction_predictions.csv', index=False)
    metrics[['model', 'split', 'tn', 'fp', 'fn', 'tp']].to_csv(tables/'prediction_confusion_matrices.csv', index=False)
    coverage = data.groupby('split').delayed_15.agg(n='size', positive_count='sum', prevalence='mean').reset_index()
    coverage['negative_count'] = coverage.n - coverage.positive_count
    coverage.to_csv(tables/'prediction_split_coverage.csv', index=False)
    write_report(root, metrics, pd.DataFrame(monthly), coverage, metadata)
    print(metrics.to_string(index=False), flush=True)
    return metrics


def prediction_frame(name, frame, probability, threshold):
    result = frame[['flight_id', 'date', 'split', 'month', 'delayed_15']].copy()
    result['model'], result['probability'], result['threshold'] = name, probability, threshold
    result['prediction'] = (probability >= threshold).astype(int)
    return result


def monthly_metrics(name, frame, probability, threshold):
    probability = pd.Series(probability, index=frame.index)
    return [{'model': name, 'split': group.split.iloc[0], 'month': int(month), 'threshold': threshold,
             **evaluate(group.delayed_15, probability.loc[group.index], threshold)}
            for month, group in frame.groupby('month')]


def write_report(root, metrics, monthly, coverage, metadata):
    selected = metadata['selected_model']
    columns = ['model', 'split', 'threshold', 'precision', 'recall', 'f1', 'roc_auc']
    selected_rows = metrics.loc[metrics.model.eq(selected)].set_index('split')
    validation, test = selected_rows.loc['validation'], selected_rows.loc['test']
    report = ['# Flight departure-delay prediction', '',
        'Target: departure delay >=15 minutes. Only observed labels are evaluated. All four models use the existing scheduled/calendar feature allowlist.', '',
        '## Protocol',
        'Training: January-August. Validation: September-October. Test: November-December. No shuffling or refitting on validation. All preprocessing is fitted on training only. Flight number is categorical; numeric variables are median-imputed and standardized, and categorical variables are mode-imputed and one-hot encoded with unseen categories ignored.',
        'The majority baseline always predicts the training majority class at threshold 0.5. Logistic regression provides an additive reference; a bounded decision tree captures interactions; the random forest averages 100 bounded trees. Hyperparameters were fixed before fitting; there was no hyperparameter search.',
        'For each learned model, thresholds 0.10 through 0.50 in steps of 0.025 were compared using validation F1 only. Ties favor the threshold closest to 0.5, then higher. Model ties are broken alphabetically. All model thresholds and the winner were saved before any test prediction. The four frozen models were each assessed once on test for the required comparison; test results do not revise the winner.', '',
        '## Class coverage', markdown_table(coverage), '',
        '## Model comparison', markdown_table(metrics[columns]), '',
        f'Selected model: **{selected}**, threshold **{metadata["threshold"]:.3f}**, because it has the highest validation F1 under the declared protocol. The saved dashboard pipeline remains fitted exclusively on January-August.', '',
        f'Validation F1 is {metrics.loc[(metrics.model == selected) & (metrics.split == "validation"), "f1"].iloc[0]:.4f}; test F1 is {metrics.loc[(metrics.model == selected) & (metrics.split == "test"), "f1"].iloc[0]:.4f}. The selected model trades many false alarms for delay recall; inspect the confusion counts below. Any higher test score from another model does not change the validation-frozen selection.', '',
        f'The selected model has test precision {test.precision:.2%} and recall {test.recall:.2%}: it detects {int(test.tp):,} of {int(test.positive_count):,} delayed flights, while producing {int(test.fp):,} false alarms. The validation F1 lead over the runner-up is only {metrics.loc[metrics.split.eq("validation"), "f1"].nlargest(2).diff().abs().iloc[-1]:.4f}, so the comparison does not establish a decisive performance advantage.',
        f'Positive prevalence changes from {validation.prevalence:.2%} in validation to {test.prevalence:.2%} in test. Test F1 is higher, while ROC-AUC changes from {validation.roc_auc:.4f} to {test.roc_auc:.4f}; higher F1 alone therefore does not imply stronger ranking or generalization. No significance test or probability calibration was performed.', '',
        '## Monthly performance',
        markdown_table(monthly[['model', 'split', 'month', 'n', 'negative_count', 'positive_count', 'prevalence', 'precision', 'recall', 'f1', 'roc_auc']]), '',
        '## Confusion matrices',
        'Rows are actual classes (0, 1); columns are predicted classes (0, 1). Each matrix is [[TN, FP], [FN, TP]].',
        markdown_table(metrics[['model', 'split', 'tn', 'fp', 'fn', 'tp']]), '',
        '## Interpretation and limitations',
        'Precision measures how many flagged flights were delayed; recall measures how many delayed flights were found. F1 balances these two at the chosen threshold. ROC-AUC uses positive-class probabilities rather than hard predictions and measures ranking, not probability calibration.',
        'Undefined precision (no predicted positives), recall (no actual positives), F1 (zero denominator), or ROC-AUC (one observed class) is stored as an empty CSV value and shown as undefined here. Zero recall/F1 for the majority baseline is real; its precision is undefined. Confusion matrices and class counts make these cases explicit.',
        'Compare validation and test metrics and the individual months rather than treating validation as a promise of future accuracy. Calendar shifts and changing flight mixes can affect generalization. Threshold selection on two validation months can overfit those months, and the grid is deliberately modest. Tree calendar features cannot extrapolate seasonal trends beyond their training range.',
        'Only 2013 NYC departures are represented. Unknown delay labels are excluded, so results do not estimate cancellation risk or performance among missing outcomes. Schedule associations do not establish causal airline quality. Probabilities are not calibrated operational guarantees. No weather or aircraft history is included.',
        'Full-year EDA was previously available for historical description; no full-year outcome aggregates, clusters, anomaly flags, or actual-time fields are used here. Feature choices and candidate specifications follow the approved scope, not final test results.', '',
        '## Artifacts and reproducibility',
        'outputs/models/flight_delay_pipeline.joblib stores the selected fitted pipeline. prediction_selection.json stores features, thresholds, parameters, versions, and source checksum. Per-model pipelines, validation threshold scores, comparison metrics, monthly metrics, confusion counts and row-level holdout predictions are also saved under outputs/. Generated artifacts are Git-ignored.',
        'Run python -m src.predict once. An existing selection file blocks accidental repeat holdout evaluation; the prediction notebook loads saved artifacts and never retrains. Use the same scikit-learn environment when loading joblib files.']
    (root/'reports/prediction_findings.md').write_text('\n\n'.join(report) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    run()
