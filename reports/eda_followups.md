# EDA follow-up findings

## July concentration
July rate: 30.33%. Its five highest-rate days account for 25.2% of July delayed flights. Excluding those five days, the remaining rate is 26.92%, versus 22.19% annually. July daily median: 30.25%. The exclusion is a sensitivity check, not a replacement for the full-month result. July rates exceed each origin annual rate at 3/3 origins. Together with the remaining-days result, this suggests the elevated monthly rate extends beyond a few extreme dates.

## Disrupted days
The highest-rate eligible day is 2013-03-08: 69.09% delayed, with 799 observed labels and 18.39% missing. 3 of 3 origins exceed their annual rates that day. The highest missing-label share occurs on 2013-02-09 (57.46%); this is a separate signal, not a confirmed cancellation count. These tables locate disruption but cannot establish its cause.

## Comparable airline schedules
  route carrier  route_observed  shared_observed  coverage  shared_cells  raw_route_rate  raw_shared_rate  standardized_rate
JFK-LAX      AA            3196             3189  0.997810            57        0.165832         0.166196           0.177964
JFK-LAX      DL            2492             2405  0.965088            57        0.140449         0.142620           0.134495
LGA-ATL      DL            5486             4794  0.873861            46        0.155669         0.142887           0.145641
LGA-ATL      FL            2289             2287  0.999126            46        0.277851         0.277656           0.266701
LGA-ORD      AA            5515             5370  0.973708            53        0.178604         0.176536           0.176490
LGA-ORD      UA            3032             2901  0.956794            53        0.227243         0.220958           0.221614
JFK-LAX: AA 17.80% versus DL 13.45%, a signed difference of 4.3 percentage points; overlap coverage 96.5% to 99.8%. LGA-ATL: DL 14.56% versus FL 26.67%, a signed difference of -12.1 percentage points; overlap coverage 87.4% to 99.9%. LGA-ORD: AA 17.65% versus UA 22.16%, a signed difference of -4.5 percentage points; overlap coverage 95.7% to 97.4%. These comparisons hold route fixed and align month/time-band weights on shared support. Coverage and raw-shared rates expose selection effects from excluding non-overlapping schedules. Remaining differences may reflect flight mix, day-specific conditions, or other unmeasured factors; they are not causal airline effects.

## Morning versus evening
Evening delay rates exceed morning rates in 36/36 supported airport-month groups. The median within-group difference is 22.0 percentage points (range 9.8 to 39.3). This is an association within airport and month, not proof of cascading delays. Route and carrier mix may still differ.

All analyses are descriptive. No weather, causal attribution, clustering, or formal anomaly detection is included. See the notebook for thresholds, plots and coverage checks.
