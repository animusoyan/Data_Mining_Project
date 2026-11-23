# NYC flight delays

Version 1 uses NYC Flights 2013. Setup, data preparation, exploratory analysis,
daily clustering and historical anomaly detection are implemented.
Flight-delay classification and evaluation are implemented; the dashboard is the next stage. Weather is excluded.

## Setup and run (PowerShell)

Use Python 3.13. Create a local environment if needed:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

Alternatively run `python -m src.prepare` using that environment. Paths are
resolved relative to the project, not the working directory. In PyCharm select
the existing interpreter `.venv\Scripts\python.exe`. Install the requirements in
the interpreter used by the notebook kernel as well if it differs from this one.

## Data and outputs

- `data/raw/flights.csv`: cached, unchanged source CSV from the Rdatasets mirror
  of `nycflights13/flights`. The source's `rownames` column becomes `flight_id`.
- `data/raw/source.json`: source URL, documentation URL and SHA-256 checksum.
- `data/processed/flights_clean.csv`: all 336,776 records, including unknown labels.
- `data/processed/classifier.csv`: observed labels, explicit scheduled/calendar
  feature allowlist, identifiers and split metadata. IDs/date/split are metadata,
  not additional model predictors.
- `data/processed/missing_departure_delay.csv`: records excluded from classification.
- `reports/data_dictionary.csv`: column definitions, types and prediction eligibility.
- `reports/data_quality.md` and `.json`: validation and missing-value results.
- `reports/monthly_coverage.csv`: observed/missing label counts by month.

Generated datasets and the local environment are ignored by Git. Quality reports
are small reviewable deliverables. Re-running uses the cached source, verifies its
checksum against the manifest, and creates only missing processed CSVs. Existing
processed CSVs are skipped; delete a CSV explicitly if you want to regenerate it
after changing preparation rules. Quality reports are refreshed on each run.
Keep the raw CSV and
manifest together to preserve the exact snapshot; a fresh download can change if
the upstream mirror changes. The quality report also records the checksum.

Dataset documentation: https://nycflights13.tidyverse.org/reference/flights.html

CSV mirror: https://vincentarelbundock.github.io/Rdatasets/csv/nycflights13/flights.csv

## Preparation rules

The target is departure delay >=15 minutes. Missing departure delay stays unknown
and is excluded only from the classifier; it is not automatically a cancellation.
Negative delays are retained. No imputation or outlier removal is performed.
HHMM values are validated, including 2400 as midnight. Destination arrival clocks
and origin departure clocks are local times, not directly comparable durations.
The original `time_hour` string is preserved without timezone conversion.

The clean descriptive dataset retains actual times and outcomes. The classifier
export excludes those fields and tail number. Prediction inputs are declared in
`src/prepare.py::FEATURES`; preprocessing must be fitted on training data later.
Train = January-August, validation = September-October, test = November-December.
Preparation only declares these partitions; the separate prediction workflow fits models.

Only 2013 is covered, so findings cannot establish recurring annual trends or
causal explanations. No weather data is downloaded or joined.

## Exploratory analysis

Open `notebooks/analysis.ipynb` with the project-local environment and run all
cells in order. It reads the prepared CSV without changing it. Charts and tables
are regenerated in `outputs/figures/` and `outputs/tables/`; the findings summary
is generated at `reports/eda_findings.md`.

The notebook covers weekly/monthly trends, delay distributions, airline/origin
comparisons, the ten busiest routes, weekday/hour heatmaps and missing-label
coverage. Rates use observed departure-delay labels as their denominator.
Daily summaries are exported for subsequent clustering and anomaly detection.
All-year historical exploration includes the held-out months; it must not be
used to tune predictive models against final test outcomes.

The notebook also follows up on July's high delay rate, the most disrupted
days, airline comparisons within shared routes/months/departure-time bands,
and morning versus evening rates within every airport and month. It exports
the supporting tables and `reports/eda_followups.md`. Thresholds, overlap
coverage and limitations are documented beside the comparisons.

## Daily clustering and anomaly detection

Run all cells in `notebooks/daily_patterns.ipynb`. It rebuilds daily summaries
directly from `flights_clean.csv`, so the EDA notebook need not run first.
Reusable calculations are in `src/patterns.py`.

K-means standardizes daily delay rate, median delay, 90th-percentile delay and
flight count. It compares k=2..6, selecting the best silhouette among solutions
with at least ten days in every cluster, and checks five additional seeds.
Dates need at least 500 observed labels and 50% label coverage. Cluster IDs are
ordered by mean daily delay rate; they are descriptive, not predictive labels.

Delay and missing-label anomalies use separate rolling baselines of prior days.
The primary threshold is the prior 28-calendar-day mean plus twice the prior
standard deviation (floor: 0.02), requiring 21 historical observations. Current
and future days cannot enter their own baseline. Warm-up and ineligible delay
scores remain unknown. Sensitivity checks use a stricter threshold and a longer
window; these are heuristic flags, not significance tests or confirmed events.

Outputs include `outputs/tables/daily_patterns.csv`, cluster profiles and
selection/stability scores, anomaly lists, sensitivity tables, and case studies.
Figures are in `outputs/figures/`; the fitted scaler/model and configuration are
in `outputs/models/`; the interpretation is in `reports/daily_patterns_findings.md`.
Rerunning this notebook refreshes these generated outputs. Prepared data is unchanged.
All-year clusters and same-day anomaly outcomes must not be used as pre-departure
classifier inputs. Only 2013 is covered and weather remains excluded.

Run checks with `python -m unittest discover -s tests -v` in the project environment.

## Flight-delay prediction

Use the verified Python 3.13 environment with `requirements.txt`. On this checkout,
that existing interpreter is
`/Users/animusoyan/PycharmProjects/PythonProject4/.venv/bin/python`;
there is no repository-local `.venv`. Select the same environment for notebooks.
For a portable new setup, follow the environment instructions above (on macOS/Linux,
use `.venv/bin/python` instead of `.venv\Scripts\python.exe`).

Run `python -m src.predict` once after preparation. It trains the four required
models: majority-class baseline, logistic regression, decision tree and random
forest. All use only `src.prepare.FEATURES`. Flight number is categorical; numeric
inputs are scaled, and unseen categories are handled. Preprocessing is fitted on
January-August only. September-October validation F1 selects thresholds and the
winner. The decision is saved before all four frozen models are assessed once
on November-December. No validation refit or test-driven retuning is performed.

`outputs/models/prediction_selection.json` records the frozen choice, feature
roles, parameters, versions and input checksum. The selected complete pipeline is
`outputs/models/flight_delay_pipeline.joblib`; load it with joblib, supply the
approved columns, and compare class-1 probabilities against the saved threshold.
Its `predict()` method uses the estimator's default threshold, so use
`src.predict.positive_probability()` and the saved threshold for project decisions.
No training is needed for future dashboard inference.

Tables under `outputs/tables/prediction_*.csv` contain validation/test metrics,
monthly sample sizes and class counts, confusion matrices, threshold search and
row-level predictions. Undefined metrics are blank, not silently replaced by zero.
`reports/prediction_findings.md` documents results and limitations.

Run `notebooks/prediction.ipynb` to inspect saved results and refresh charts. It
never retrains or repeats holdout inference. An existing selection file blocks
accidental rerunning of the training command; preserve the original artifacts and
holdout assessment. Generated artifacts remain Git-ignored. Tests use
`python -m unittest discover -s tests -v` and include saved-pipeline inference once
artifacts exist. The dashboard remains a later stage.

## Independent flight-delay historical replays

From the project environment, with the cached NYC Flights 2013 snapshot and
its manifest in `data/raw/`, run:

```powershell
.\.venv\Scripts\python.exe -B -m experiments.flight_delay_replay.run
```

On macOS or Linux, substitute `.venv/bin/python`. An existing output directory
is protected. Use `--output outputs/experiments/flight_delay_replay_repeat`
for a new run. The command writes the protocol, cutoff-selection results,
September–October and November–December historical metrics, a report, and a
timing chart under `outputs/experiments/flight_delay_replay/`. To regenerate
only the report and chart from those tables, run
`python -m experiments.flight_delay_replay.report outputs/experiments/flight_delay_replay`.

The replays compare departure-delay alerts before the day and one hour before
departure, accuracy at six departure checkpoints, and arrival delay at
scheduled arrival. A positive label is a delay of at least 15 minutes.
Training and cutoff selection use earlier months. Early flights are retained
and missing labels are excluded independently for each target. The two
assessment periods are historical comparisons.

The replay assumes reports were available at their reconstructed event times.
The departure schedule is localized to the origin and the arrival schedule to
the destination. The arrival date uses the earliest destination-local
occurrence after departure. Times in a spring-forward gap move to the first
valid local time. An ambiguous scheduled time starts with the daylight
occurrence, while observed times use the reported delay to disambiguate.
The lookup in `experiments/flight_delay_replay/airport_time_zones.csv` comes
from the companion airport data and includes Puerto Rico and the U.S. Virgin
Islands for airports omitted there. [Airport metadata source](https://nycflights13.tidyverse.org/reference/airports.html)

The separate compact schedule-only development grid runs with
`python -B -m experiments.chronological_grid_search` after data preparation.
It searches on the audit's first three folds and compares the frozen
shortlist in September–October. Add `--resume` to reuse completed fits from
a compatible run. For an archived or incompatible run, use
`--output outputs/experiments/chronological_grid_search_sklearn` to start
in a new directory.
