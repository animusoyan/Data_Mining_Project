# Data preparation quality report

Flights retained: 336,776.
Known departure-delay labels: 328,521.
Unknown departure-delay labels: 8,255.

Checks passed: expected schema/row count, 2013 coverage, three NYC origins, unique rows,
valid calendar dates and HHMM fields, schedule hour/minute consistency, positive distances and observed air times.

Missing labels remain unknown; they are not classified as on-time or automatically called cancellations.
Negative delays are valid early departures/arrivals. No outliers were removed or values imputed.
2400 is accepted as midnight for clock features. Arrival clocks use destination local time;
they are not subtracted from departure clocks to calculate durations.

Classifier inputs contain only scheduled/calendar information. Outcomes remain in the descriptive dataset.
Tail number is excluded because its availability before departure is not established.
No encoders, scalers, or imputers are fitted here; fit them on the training partition later.
Split: January-August train, September-October validation, November-December test.
Weather is excluded. One year cannot establish recurring annual trends.

See data_quality.json for missing values and split counts; monthly_coverage.csv for monthly label coverage.
Raw SHA-256: b791a2590c9167aa0bd3cb45af3f64101180878cf088c183eae4011e3a654f42
