# Exploratory findings: NYC Flights 2013

Coverage: 336,776 flights; 328,521 observed departure delays; 8,255 missing (2.45%).
Overall departure delay rate (>=15 minutes): 22.19% among observed labels.
Highest monthly rate: month 7, 30.33%; lowest: month 9, 14.41%.
Median delay: -2.0 min; mean: 12.6; 90th percentile: 49.0.
Origin rates: EWR: 25.32% (observed n=117,596); JFK: 21.34% (observed n=109,416); LGA: 19.50% (observed n=101,509).
Highest airline rate: EV (31.14%); lowest: HA (7.02%). See airlines.csv for flight counts.
Highest scheduled-hour rate: 21:00 (37.85%, observed n=10,524). Sparse hours can be unstable; inspect the count heatmap.

## Interpretation and limitations

Route comparisons use the ten busiest routes by total flights, not extreme delay rates.
Missing labels are unknown, not on-time flights. Rates exclude missing labels; counts remain visible.
Charts that hide outliers or limit the display range retain all observed values in summary statistics.
Weekly buckets end Sunday; first and last buckets cover partial weeks.
Airline, route and hour differences are unadjusted descriptive associations, not causal effects.
All-year EDA includes held-out months for historical description only. Future model selection must use training/validation data; final test metrics remain reserved for final evaluation.
One year of NYC departures cannot establish recurring annual trends. Weather is excluded.
Daily summaries are exported for later clustering and anomaly detection. Neither is performed here.
