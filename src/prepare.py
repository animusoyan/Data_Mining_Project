"""Download and prepare NYC Flights 2013. Run with python -m src.prepare."""
from pathlib import Path
import hashlib
import json
from urllib.request import urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
URL = "https://vincentarelbundock.github.io/Rdatasets/csv/nycflights13/flights.csv"
SOURCE_DOCS = "https://nycflights13.tidyverse.org/reference/flights.html"
FEATURES = ["month", "day_of_week", "day_of_year", "is_weekend", "sched_dep_minutes",
            "sched_arr_minutes", "carrier", "flight", "origin", "dest", "distance"]
DESCRIPTIONS = {
    "year": "Scheduled departure year", "month": "Scheduled departure month",
    "day": "Scheduled departure day of month",
    "dep_time": "Actual departure, local HHMM; 2400 is midnight",
    "sched_dep_time": "Scheduled departure, origin local HHMM",
    "dep_delay": "Departure delay in minutes; negative means early",
    "arr_time": "Actual arrival, destination local HHMM; 2400 is midnight",
    "sched_arr_time": "Scheduled arrival, destination local HHMM",
    "arr_delay": "Arrival delay in minutes; negative means early",
    "carrier": "Two-letter airline code", "flight": "Flight number",
    "tailnum": "Aircraft registration; excluded from initial predictors",
    "origin": "Origin airport code", "dest": "Destination airport code",
    "air_time": "Actual airborne duration in minutes", "distance": "Distance in miles",
    "hour": "Scheduled departure hour", "minute": "Scheduled departure minute",
    "time_hour": "Source scheduled departure hour timestamp; retained unchanged",
    "flight_id": "Stable source row identifier for this downloaded snapshot",
    "date": "Scheduled departure date (origin local calendar)",
    "day_of_week": "Monday=0 through Sunday=6", "day_of_year": "Calendar day 1-365",
    "is_weekend": "Saturday or Sunday",
    "sched_dep_minutes": "Scheduled departure minutes since midnight",
    "sched_arr_minutes": "Scheduled arrival local minutes since midnight; not a duration",
    "route": "Origin-destination pair", "label_observed": "Departure delay is observed",
    "delayed_15": "1 if departure delay >=15 minutes, 0 otherwise; missing if unknown",
    "split": "train Jan-Aug; validation Sep-Oct; test Nov-Dec",
}


def clock_minutes(values):
    """Accept integer HHMM in 0000..2359 or 2400; reject malformed values."""
    hour, minute = values // 100, values % 100
    valid = values.isna() | ((values == values.round()) &
                            (((hour >= 0) & (hour <= 23) & (minute < 60)) | (values == 2400)))
    if not valid.all():
        raise ValueError(f"Invalid HHMM values: {values[~valid].head().tolist()}")
    return ((hour * 60 + minute) % 1440).astype("Int64")


def prepare(raw):
    df = raw.rename(columns={"rownames": "flight_id"}).copy()
    expected = set(DESCRIPTIONS) - {"date", "day_of_week", "day_of_year", "is_weekend",
        "sched_dep_minutes", "sched_arr_minutes", "route", "label_observed", "delayed_15", "split"}
    if set(df.columns) != expected:
        raise ValueError(f"Unexpected source schema: {sorted(df.columns)}")
    if len(df) != 336776 or not df.year.eq(2013).all() or set(df.origin) != {"JFK", "LGA", "EWR"}:
        raise ValueError("Unexpected NYC Flights 2013 coverage")
    if not df.flight_id.is_unique or df.duplicated(subset=df.columns.drop("flight_id")).any():
        raise ValueError("Duplicate source rows")
    df["date"] = pd.to_datetime(df[["year", "month", "day"]], errors="raise")
    for column in ["dep_time", "arr_time", "sched_dep_time", "sched_arr_time"]:
        minutes = clock_minutes(df[column])
        if column.startswith("sched_"):
            if minutes.isna().any():
                raise ValueError(f"Missing schedule: {column}")
            df[column.replace("_time", "_minutes")] = minutes
    if not (df.hour.eq(df.sched_dep_time // 100) & df.minute.eq(df.sched_dep_time % 100)).all():
        raise ValueError("Scheduled departure hour/minute mismatch")
    if (df.distance <= 0).any() or (df.air_time.dropna() <= 0).any():
        raise ValueError("Nonpositive distance or air time")
    df["day_of_week"] = df.date.dt.dayofweek
    df["day_of_year"] = df.date.dt.dayofyear
    df["is_weekend"] = df.day_of_week.ge(5)
    df["route"] = df.origin + "-" + df.dest
    df["label_observed"] = df.dep_delay.notna()
    df["delayed_15"] = df.dep_delay.ge(15).astype("Int64").where(df.label_observed)
    df["split"] = df.month.map(lambda m: "train" if m <= 8 else "validation" if m <= 10 else "test")
    return df


def main():
    raw_dir, processed = ROOT / "data/raw", ROOT / "data/processed"
    reports = ROOT / "reports"
    for folder in (raw_dir, processed, reports):
        folder.mkdir(parents=True, exist_ok=True)
    path = raw_dir / "flights.csv"
    if not path.exists():
        with urlopen(URL, timeout=120) as response:
            content = response.read()
        # Validate before caching a download (e.g. reject an HTML error response).
        from io import BytesIO
        prepare(pd.read_csv(BytesIO(content)))
        path.write_bytes(content)
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path = raw_dir / "source.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text())["sha256"] != checksum:
        raise ValueError("Raw data checksum differs from saved manifest")
    raw = pd.read_csv(path)
    df = prepare(raw)
    manifest = {"dataset": "nycflights13/flights", "download_url": URL,
                "documentation": SOURCE_DOCS, "sha256": checksum,
                "rows": len(df), "weather_included": False}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    clean_path = processed / "flights_clean.csv"
    if not clean_path.exists():
        df.to_csv(clean_path, index=False, date_format="%Y-%m-%d")
    else:
        print(f"Skipping existing file: {clean_path.name}")
    classifier = df.loc[df.label_observed, ["flight_id", "date", "split", *FEATURES, "delayed_15"]]
    classifier_path = processed / "classifier.csv"
    if not classifier_path.exists():
        classifier.to_csv(classifier_path, index=False, date_format="%Y-%m-%d")
    else:
        print(f"Skipping existing file: {classifier_path.name}")
    missing_path = processed / "missing_departure_delay.csv"
    if not missing_path.exists():
        df.loc[~df.label_observed].to_csv(missing_path, index=False)
    else:
        print(f"Skipping existing file: {missing_path.name}")
    dictionary = pd.DataFrame([{"column": c, "dtype_in_memory": str(df[c].dtype),
        "description": DESCRIPTIONS[c], "prediction_input": c in FEATURES} for c in df.columns])
    dictionary.to_csv(reports / "data_dictionary.csv", index=False)
    missing = df.isna().sum()
    monthly = df.groupby("month").agg(flights=("flight_id", "size"),
        observed_labels=("label_observed", "sum"), delayed=("delayed_15", "sum"))
    monthly["missing_labels"] = monthly.flights - monthly.observed_labels
    monthly.to_csv(reports / "monthly_coverage.csv")
    quality = {**manifest, "classifier_rows": len(classifier),
        "missing_departure_delay": int((~df.label_observed).sum()),
        "missing_values": {c: int(n) for c, n in missing.items()},
        "split_counts": classifier.groupby("split").size().to_dict(),
        "prediction_features": FEATURES}
    (reports / "data_quality.json").write_text(json.dumps(quality, indent=2) + "\n", encoding="utf-8")
    lines = ["# Data preparation quality report", "", f"Flights retained: {len(df):,}.",
        f"Known departure-delay labels: {len(classifier):,}.",
        f"Unknown departure-delay labels: {len(df) - len(classifier):,}.", "",
        "Checks passed: expected schema/row count, 2013 coverage, three NYC origins, unique rows,",
        "valid calendar dates and HHMM fields, schedule hour/minute consistency, positive distances and observed air times.", "",
        "Missing labels remain unknown; they are not classified as on-time or automatically called cancellations.",
        "Negative delays are valid early departures/arrivals. No outliers were removed or values imputed.",
        "2400 is accepted as midnight for clock features. Arrival clocks use destination local time;",
        "they are not subtracted from departure clocks to calculate durations.", "",
        "Classifier inputs contain only scheduled/calendar information. Outcomes remain in the descriptive dataset.",
        "Tail number is excluded because its availability before departure is not established.",
        "No encoders, scalers, or imputers are fitted here; fit them on the training partition later.",
        "Split: January-August train, September-October validation, November-December test.",
        "Weather is excluded. One year cannot establish recurring annual trends.", "",
        "See data_quality.json for missing values and split counts; monthly_coverage.csv for monthly label coverage.",
        f"Raw SHA-256: {checksum}"]
    (reports / "data_quality.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: quality[k] for k in ["rows", "classifier_rows", "missing_departure_delay", "split_counts"]}, indent=2))


if __name__ == "__main__":
    main()
