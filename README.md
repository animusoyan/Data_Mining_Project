# NYC flight-delay research

This project describes daily disruption at JFK, LGA and EWR and evaluates when a flight’s delay of **at least 15 minutes** can be predicted. It uses the cached NYC Flights 2013 snapshot, with 336,776 flights. The original classifier predicts departure delay from the published schedule. The later replays add operating information available at specific checkpoints and use a separate arrival-delay target at scheduled arrival. Missing outcome labels are excluded for each target. Early flights remain in the labeled population. Weather was not incorporated. [Dataset documentation](https://nycflights13.tidyverse.org/reference/flights.html)

Data preparation, exploratory analysis, daily clustering and anomaly detection, and the six prediction research stages are complete. The first Streamlit dashboard is available for review. Start with the [current findings](reports/current_findings.md) and the [artifact audit](reports/artifact_audit.md).

## Research reading order

| Stage | What it covers | Where to read |
| --- | --- | --- |
| Data preparation | Cached source verification, HHMM handling, missing labels, scheduled features and chronological partitions. | [Data quality](reports/data_quality.md), [dictionary](reports/data_dictionary.csv), [preparation code](src/prepare.py) |
| EDA and daily patterns | All-year descriptive EDA, K-means daily profiles and anomaly flags based on preceding days. The descriptive outcomes stay outside the prediction features. | [EDA](reports/eda_findings.md), [follow-ups](reports/eda_followups.md), [daily findings](reports/daily_patterns_findings.md), [EDA notebook](notebooks/analysis.ipynb), [daily notebook](notebooks/daily_patterns.ipynb), [pattern code](src/patterns.py) |
| 1. Original four-model experiment | Majority baseline, logistic regression, decision tree and random forest. January–August fit, September–October F1 selection and a frozen November–December assessment. | [Original findings](reports/prediction_findings.md), [notebook](notebooks/prediction.ipynb), [code](src/predict.py) |
| 2. Fourteen-configuration improvement study | Schedule encodings and model settings ranked on three chronological folds. The frozen shortlist was compared in September–October, a period also used to choose cutoffs for new models. | [Study report](reports/modeling_improvement_study.md), [notebook](notebooks/modeling_improvement_study.ipynb), [code](experiments/modeling_study.py) |
| 3. Independent seven-candidate audit | Checks the original results and prior study, then compares candidates A–G, including matched compact logistic and boosting models. The new-model folds end in October. | [Audit report](reports/independent_model_audit.md), [notebook](notebooks/independent_model_audit.ipynb), [code](experiments/independent_audit/) |
| 4. Chronological calibration study | Evaluates intercept-only recalibration of frozen audit F/G models on later development months. Calibration was not adopted. | [Calibration report](reports/chronological_calibration_study.md), [notebook](notebooks/chronological_calibration_study.ipynb), [code](experiments/calibration_study/) |
| 5. Chronological schedule-only grid | Ranks 39 configurations and references by mean F1 across three earlier folds, then compares a frozen shortlist in September–October. | [Grid report](outputs/experiments/chronological_grid_search/report.md), [ranking](outputs/experiments/chronological_grid_search/ranking.csv), [code](experiments/chronological_grid_search.py) |
| 6. Departure and arrival replay | Compares start-of-day and one-hour departure predictions, status-assisted departure checkpoints and predictions at scheduled arrival. Both historical assessment windows are exploratory. | [Replay report](outputs/experiments/flight_delay_replay/report.md), [metrics](outputs/experiments/flight_delay_replay/historical_metrics.csv), [code](experiments/flight_delay_replay/) |

## Streamlit dashboard

Use Python 3.13 and a dedicated dashboard environment. Run these commands from the project root. The existing research environment can continue to use its original requirements.

**macOS and Linux**

```sh
python3.13 -m venv ~/.venvs/nyc-flight-dashboard
~/.venvs/nyc-flight-dashboard/bin/python -m pip install -r requirements-dashboard.txt
~/.venvs/nyc-flight-dashboard/bin/python -B -m streamlit run app.py
```

**Windows PowerShell**

```powershell
$dashboardPython = "$env:LOCALAPPDATA\venvs\nyc-flight-dashboard\Scripts\python.exe"
py -3.13 -m venv "$env:LOCALAPPDATA\venvs\nyc-flight-dashboard"
& $dashboardPython -m pip install -r requirements-dashboard.txt
& $dashboardPython -B -m streamlit run app.py
```

| Section | Saved inputs and filter scope |
| --- | --- |
| Flight reliability | [Prepared flight records](data/processed/flights_clean.csv). The date range, airline and origin filters all act on these flight-level records. |
| Daily disruption | `daily_patterns.csv`, `cluster_profiles.csv`, `delay_anomalies.csv`, `missing_label_anomalies.csv` and `pattern_case_studies.csv` from the explicitly configured [descriptive recovery run](outputs/experiments/descriptive_recovery/descriptive_recovery_001/tables/). Its own date filter limits which days are displayed. Cluster profiles, assignments and flags describe the full NYC network. |
| Models and findings | [Canonical results](reports/current_results.json), [original comparison](outputs/tables/prediction_comparison.csv), [selection record](outputs/models/prediction_selection.json) and [audit AP supplement](outputs/experiments/independent_audit/stage1_audit_001/historical_recomputed.csv). The assessment and checkpoint selectors change which saved rows are displayed. |

Restore missing files from the shared artifacts at these same project-relative paths. The dashboard displays a message for missing data and reads CSV and JSON only. Its scope is historical presentation. Operational live inference is not implemented. To check the dashboard with the dashboard environment, run `python -B -m unittest discover -s tests -p test_dashboard.py -v`.

## Local setup

Use Python 3.13. The verification environment used Python 3.13.13 on macOS arm64. The experiment requirements include the existing pins and declare the other direct research and notebook dependencies at their verified versions. On Windows, `py -3.13` requires the Python launcher. [Verified dependency record](reports/artifact_audit.md#verified-environment)

**macOS and Linux**

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements-experiments.txt
.venv/bin/python -B -m unittest discover -s tests -v
.venv/bin/python -B scripts/summarize_saved_results.py --check
```

**Windows PowerShell**

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-experiments.txt
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -B scripts\summarize_saved_results.py --check
```

For the full checks and saved-result notebooks, place the shared data and outputs at their original relative paths first. Select the same environment as the notebook kernel. The portable commands use the local project environment you create. Historical reports retain the original interpreter paths and one-off run prerequisites.

## Reading saved results and reproducing runs

The examples below use `python` from the selected environment.

| Action or command | What it does |
| --- | --- |
| Open the reports, CSVs, JSON or rendered notebook cells | Reads saved evidence. Existing notebook outputs can be viewed without running cells. |
| `python -B scripts/summarize_saved_results.py --check` | Reads saved CSV/JSON and verifies the [machine-readable summary](reports/current_results.json). Omitting `--check` prints the generated JSON. |
| `python -m notebook` | Opens the notebooks. Executing `prediction.ipynb` or `modeling_improvement_study.ipynb` reads saved metrics and refreshes figures. Executing the audit or calibration notebook displays tables and plots from saved files. |
| `python -m experiments.study_report` or `python -m experiments.flight_delay_replay.report PATH_TO_REPLAY` | Reads saved tables and rewrites the study report or replay report and chart at their configured destinations. Read the existing files to preserve that historical rendering. |
| `python -m src.prepare` or `python main.py` | Verifies or fetches the raw snapshot, creates missing processed data and refreshes data-quality reports. |
| Execute `analysis.ipynb` or `daily_patterns.ipynb` | Recomputes descriptive exports. The daily notebook fits clustering again and regenerates anomaly exports. |
| `python -m src.predict` | Trains the original four models and runs the historical assessment. The saved selection record protects an existing run. |
| `python -m experiments.modeling_study` | Fits the improvement configurations and experimental calibrator. The default study output directory is protected. |
| `python -m experiments.independent_audit.run` and `python -m experiments.calibration_study.run` | Historical phased workflows. The audit’s `experiment` phase fits models and its `historical` phase can run development inference. Calibration `execute` can score frozen bases and fit intercepts. Use the archived reports for exact run context, inputs and phase locks. |
| `python -B -m experiments.chronological_grid_search` and `python -B -m experiments.flight_delay_replay.run` | Fit and assess the corresponding experiments. A deliberately authorized repeat needs a fresh `--output` directory. Use grid `--resume` only for a compatible incomplete run. |

The replay and grid provide saved metrics. The [artifact audit](reports/artifact_audit.md) records gaps for flight-level replay and operational inference.

## Files and sharing

- `src/` contains preparation, pattern analysis and the original classifier. `experiments/` contains later study code. Preserve module paths for loading saved pipelines.
- `reports/` contains current findings, the artifact audit and historical write-ups. `notebooks/` retains the displayed outputs.
- `data/raw/` holds the cached CSV and source manifest. `data/processed/` holds the clean flight data, the classifier export and the missing-label records.
- `outputs/tables/`, `outputs/figures/`, `outputs/models/` and `outputs/experiments/` hold generated evidence. **Git ignores the generated data and outputs.** Share these separately at the same relative paths, including manifests, intermediate results and checkpoints.
- For a shared export, include the project source, scripts, tests, documentation, requirements, notebooks and intended research data/output directories. Omit `.git/`, `.idea/` (including files tracked in the repository), virtual environments, caches, OS metadata and packaging debris.
