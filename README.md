# NYC flight delays

Version 1 uses NYC Flights 2013. Setup and data preparation are implemented;
exploratory analysis, daily clustering, anomaly detection, prediction and a
dashboard are the next stages. Weather is excluded.

## Setup and run (PowerShell)

Use Python 3.13. Create a local environment if needed:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

Alternatively run `python -m src.prepare` using that environment. Paths are
resolved relative to the project, not the working directory. In PyCharm select
the existing interpreter `.venv\Scripts\python.exe`. The IDE previously pointed
to another project's environment; that environment has not been modified.

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
These are partitions only; no models have been trained.

Only 2013 is covered, so findings cannot establish recurring annual trends or
causal explanations. No weather data is downloaded or joined.
