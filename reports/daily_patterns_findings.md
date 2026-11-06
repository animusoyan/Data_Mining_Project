# Daily clustering and anomaly findings

Selected k=2. Minimum seed agreement (adjusted Rand index): 1.000. Profiles report day-average rates, not flight-weighted rates. IDs increase with average delay rate; flight count also influences membership. Silhouette assesses geometry, not causal explanations or forecast accuracy.

## Cluster profiles (day-average statistics)
         days  flights_mean  delay_rate_mean  median_delay_mean  p90_delay_mean  missing_rate_mean
cluster                                                                                           
0         279    911.512545          0.16679          -2.177419       30.190681           0.008399
1          84    962.500000          0.40387           6.166667      107.732143           0.064962

## Past-only anomaly baseline
29 delay flags across 342 scorable days; 21 separate missing-label flags. Early dates without adequate history remain unscored. Threshold scores are descriptive deviations, not statistical significance tests. High absolute delays can be unflagged during an already disrupted period.

Threshold: previous 28 calendar-day mean + 2 * max(previous standard deviation, 0.02), requiring 21 prior valid days. Delay scoring requires >=500 labels and >=50% coverage; missing-label scoring uses all days.

## Sensitivity checks
 window_days  minimum_days  multiplier  scorable_days  delay_flags  missing_flags  common_days  primary_flags_on_common_days  alternative_flags_on_common_days
          28            21         2.0            342           29             21          342                            29                                29
          28            21         3.0            342            6             13          342                            29                                 6
          56            42         2.0            321           21             17          321                            26                                21

## Relationship between clusters and anomalies
         days  scorable_days  delay_flags  missing_flags
cluster                                                 
0         279            259            2              1
1          84             83           27             18

Clusters describe absolute daily behaviour; anomaly flags describe deviation from recent history. They need not agree.

## Previously investigated dates
            eligible  cluster  delay_rate  missing_rate  delay_rate_threshold  delay_rate_flag  missing_rate_flag
2013-03-08      True        1    0.690864      0.183861              0.407474             True              False
2013-07-01      True        1    0.600454      0.087992              0.563381             True              False
2013-07-23      True        1    0.521064      0.095286              0.572295            False              False
2013-02-09     False     <NA>    0.316151      0.574561              0.374891             <NA>               True

## Limitations and next step
K-means imposes groups on continuous data and depends on scaling and chosen features. Seed agreement is not evidence of causal or temporal validity.
The recent baseline can adapt upward during disruption and does not adjust for weekdays, routes, carriers or seasons. Flags are investigation prompts, not confirmed events.
Missing outcomes are not automatically cancellations. Warm-up/ineligible days are unknown, not normal.
Full-year descriptive outcomes are not pre-departure prediction features. Next: chronological flight-delay classification using the existing train/validation/test partitions. Weather remains excluded.

## What these results tell us

Cluster 0 contains 279 days: average daily delay rate 16.7%, average daily median delay -2.2 minutes, and average daily 90th-percentile delay 30.2 minutes.
Cluster 1 contains 84 days: average daily delay rate 40.4%, average daily median delay 6.2 minutes, and average daily 90th-percentile delay 107.7 minutes.
Two and three clusters have similar silhouette scores (0.526 and 0.521). The selected partition is a useful coarse summary, not evidence of two natural types of day.
29 scorable delay spikes overlap the clusters; 27 occur in the highest-delay cluster.
March 8 and July 1 are both high-delay cluster members and past-baseline delay flags. July 23 is a high-delay cluster member but not a delay flag: its threshold has risen after preceding disruptions.
February 8 and 9 have too few known departure delays for delay clustering/scoring, but both remain visible through the separate missing-label investigation. Unknown delay flags must not be interpreted as evidence that these days were normal.
The primary threshold flags 29 days; using three standard deviations flags 6. Flag counts are threshold-sensitive, and the longer-window comparison also has a longer warm-up.
For a dashboard, show cluster profile, raw rate, baseline, missing-label share and scoring status together. A single red/green normality label would hide important distinctions.
Next, evaluate pre-departure flight classification chronologically. Do not feed these same-day, full-year outcome summaries into those models.
