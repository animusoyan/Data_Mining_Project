"""Chronological selection, unresolved-only model fitting and historical evaluation."""

from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from experiments.independent_audit.core import GRID, SEED, choose, metrics, threshold_curve

from .features import CATEGORICAL, as_of


HISTORICAL = {'September–October': (9, 10), 'November–December': (11, 12)}
ACCURACY_GRID = np.round(np.arange(.1, .951, .005), 3)
F1_GRID = np.sort(np.r_[GRID, .265])
FIXED_BOOSTING = dict(max_leaf_nodes=15, min_samples_leaf=100, learning_rate=.1,
                     l2_regularization=10, max_iter=100, early_stopping=False, random_state=SEED)


@dataclass(frozen=True)
class Spec:
    name: str
    block: str
    model: str
    parameters: dict | None = None
    cutoff: float | None = None


def make_model(model, columns, parameters=None):
    categorical = [column for column in CATEGORICAL if column in columns]
    numeric = [column for column in columns if column not in categorical]
    parameters = parameters or {}
    if model in ('hgb', 'regression'):
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
        preprocessing = ColumnTransformer([('cats', encoder, categorical), ('nums', 'passthrough', numeric)])
        estimator = HistGradientBoostingClassifier if model == 'hgb' else HistGradientBoostingRegressor
        model = estimator(**{**FIXED_BOOSTING, **parameters}, categorical_features=list(range(len(categorical))))
    else:
        preprocessing = ColumnTransformer([
            ('cats', OneHotEncoder(handle_unknown='ignore'), categorical),
            ('nums', Pipeline([('fill', SimpleImputer(strategy='median', add_indicator=True)),
                               ('scale', StandardScaler())]), numeric),
        ])
        models = {
            'lr': LogisticRegression(solver='liblinear', max_iter=500, random_state=SEED),
            'rf': RandomForestClassifier(n_estimators=100, max_depth=14, min_samples_leaf=150,
                                         max_features='sqrt', n_jobs=4, random_state=SEED),
            'mlp': MLPClassifier(hidden_layer_sizes=(16,), max_iter=40, early_stopping=False,
                                 random_state=SEED),
        }
        model = models[model].set_params(**parameters)
    return Pipeline([('preprocess', preprocessing), ('model', model)])


def chronological_masks(flights, fit_end, observed):
    if fit_end not in (6, 7):
        raise ValueError('Fit must end in June or July')
    if not flights.date.dt.year.eq(2013).all() or not flights.month.eq(flights.date.dt.month).all():
        raise ValueError('Replay chronology requires the original 2013 departure calendar')
    fit = observed & flights.month.le(fit_end)
    validation = observed & flights.month.between(fit_end + 1, 8)
    assessments = {period: observed & flights.month.between(*months) for period, months in HISTORICAL.items()}
    return fit, validation, assessments


def departure_resolved(flights, events, cutoff):
    return as_of(events.departed, cutoff) & flights.dep_delay.lt(15)


def select_cutoff(y, p, metric, forced=None):
    if forced is not None:
        return forced
    if metric == 'f1':
        return choose(threshold_curve(y, p, F1_GRID))
    return float(max(ACCURACY_GRID, key=lambda t: (np.mean((p >= t) == y), -abs(t - .5), t)))


def evaluate_rows(y, scores, unresolved, cutoff, common):
    for subset, keep in [('all', np.ones(len(y), dtype=bool)), ('unresolved', unresolved)]:
        if keep.any():
            yield {**common, 'subset': subset, 'threshold': cutoff, **metrics(y[keep], scores[keep], cutoff)}


class Comparison:
    def __init__(self, name, checkpoint, lead, flights, output, cutoff, resolved, minutes,
                 features, blocks, weight, fit_end, rule=None, extra_rules=None):
        self.name, self.checkpoint, self.lead = name, checkpoint, lead
        self.output, self.features, self.blocks = output, features, blocks
        self.weight, self.fit_end = weight, fit_end
        self.cutoff = cutoff
        self.unresolved = ~resolved
        self.minutes = minutes
        self.y = minutes.ge(15)
        self.fit, self.tune, self.assessments = chronological_masks(flights, fit_end, minutes.notna())
        # A scheduled-month label is not available for fitting if its report
        # crossed into the first cutoff-selection month (or was never reported).
        self.rule = rule if rule is not None else self.unresolved
        self.extra_rules = extra_rules or {}
        self.models = {}

    def check_label_reporting(self, report):
        tune_start = self.cutoff.loc[self.tune].min()
        if pd.isna(tune_start) or not as_of(report, tune_start - 1).loc[self.fit].any():
            raise ValueError('Empty chronological fit or cutoff-selection period')
        self.fit &= self.cutoff.lt(tune_start) & as_of(report, tune_start - 1)
        # August labels received after the first September prediction cutoff
        # cannot contribute to the frozen historical settings.
        assessment_start = self.cutoff.loc[next(iter(self.assessments.values()))].min()
        self.tune &= self.cutoff.lt(assessment_start) & as_of(report, assessment_start - 1)
        if not self.fit.any() or not self.tune.any():
            raise ValueError('Chronological reporting constraints left an empty selection partition')

    def model(self, block, name, parameters):
        key = (block, name, json.dumps(parameters, sort_keys=True))
        if key not in self.models:
            cols = self.blocks[block]
            eligible = self.fit & self.unresolved
            y = self.minutes if name == 'regression' else self.y
            estimator = make_model(name, cols, parameters)
            estimator.fit(self.features.loc[eligible, cols], y.loc[eligible])
            self.models[key] = estimator
        return self.models[key]

    def score(self, spec, mask):
        if spec.model == 'always_on_time':
            return np.zeros(mask.sum())
        if spec.model == 'waiting_rule':
            return self.rule.loc[mask].to_numpy(dtype=float)
        if spec.model in self.extra_rules:
            return self.extra_rules[spec.model].loc[mask].to_numpy(dtype=float)
        names = spec.model.split('+')
        active = self.unresolved.loc[mask].to_numpy(dtype=bool)
        result = np.zeros(len(active))
        if not active.any():
            return result
        indices = self.features.index[mask & self.unresolved]
        inputs = self.features.loc[indices, self.blocks[spec.block]]
        scores = []
        for name in names:
            parameters = spec.parameters if len(names) == 1 else None
            model = self.model(spec.block, name, parameters)
            if name == 'regression':
                scores.append(expit((model.predict(inputs) - 15) / 10))
            else:
                scores.append(model.predict_proba(inputs)[:, list(model.classes_).index(True)])
        result[active] = np.mean(scores, axis=0)
        return result

    def run(self, specs, *, grid_prefix=None):
        tune = []
        y = self.y.loc[self.tune].to_numpy()
        for spec in specs:
            print(f'{self.name} | {self.checkpoint} | {spec.name}', flush=True)
            score = self.score(spec, self.tune)
            cutoff = select_cutoff(y, score, self.weight, spec.cutoff)
            tune.append({'experiment': self.name, 'checkpoint': self.checkpoint, 'lead_minutes': self.lead,
                         'id': spec.name, 'features': spec.block, 'model': spec.model,
                         'parameters': spec.parameters, 'cutoff': cutoff, 'selection_metric': self.weight,
                         'fit_through_month': self.fit_end, **metrics(y, score, cutoff)})
        score_models = [row for row in tune if row['features'] != 'none']
        winner = max(score_models or tune, key=lambda row: (row[self.weight], row['id']))
        grid_winner = None
        if grid_prefix:
            grid_winner = max((row for row in tune if row['id'].startswith(grid_prefix)),
                              key=lambda row: (row[self.weight], row['id']))['id']
        selected = {row['id']: row for row in tune
                    if not grid_prefix or not row['id'].startswith(grid_prefix) or row['id'] == grid_winner}
        frozen = pd.DataFrame(tune).assign(winner=lambda d: d.id.eq(winner['id']),
                                          historical=lambda d: d.id.isin(selected))
        path = self.output / f'{self.name}_{self.lead if self.lead is not None else "day"}_selection.csv'
        frozen.to_csv(path, index=False)
        rows = []
        for spec in specs:
            if spec.name not in selected:
                continue
            common = {key: selected[spec.name][key] for key in
                      ['experiment', 'checkpoint', 'lead_minutes', 'id', 'features', 'model', 'parameters']}
            common['winner'] = spec.name == winner['id']
            for period, mask in self.assessments.items():
                probability = self.score(spec, mask)
                rows.extend(evaluate_rows(self.y.loc[mask].to_numpy(), probability,
                                          self.unresolved.loc[mask].to_numpy(),
                                          selected[spec.name]['cutoff'], {**common, 'period': period}))
        return rows, frozen
