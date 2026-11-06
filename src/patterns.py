"""Daily descriptive clusters and past-only disruption flags."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

FEATURES = ["delay_rate", "median_delay", "p90_delay", "flights"]
MIN_LABELS = 500
MIN_COVERAGE = 0.5


def daily_summary(flights):
    result = flights.groupby("date").agg(
        flights=("flight_id", "size"), observed_labels=("dep_delay", "count"),
        delayed=("delayed_15", "sum"), median_delay=("dep_delay", "median"),
        p90_delay=("dep_delay", lambda x: x.quantile(.9)))
    result = result.sort_index()
    result["delay_rate"] = result.delayed / result.observed_labels.replace(0, np.nan)
    result["missing_rate"] = 1 - result.observed_labels / result.flights
    result["eligible"] = ((result.observed_labels >= MIN_LABELS) &
                          (1 - result.missing_rate >= MIN_COVERAGE) &
                          result[FEATURES].notna().all(axis=1))
    return result


def cluster_days(daily):
    """Full-year descriptive fitting; never use these outcomes as flight predictors."""
    data = daily.loc[daily.eligible, FEATURES]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    records, models = [], {}
    for k in range(2, 7):
        model = KMeans(n_clusters=k, random_state=42, n_init=20).fit(scaled)
        models[k] = model
        sizes = np.bincount(model.labels_, minlength=k)
        records.append({"k": k, "silhouette": silhouette_score(scaled, model.labels_),
                        "inertia": model.inertia_, "smallest_cluster": int(sizes.min())})
    scores = pd.DataFrame(records).set_index("k")
    # Avoid a numerically attractive solution built around a handful of days.
    supported = scores.loc[scores.smallest_cluster >= 10]
    if supported.empty:
        raise ValueError("No candidate has at least 10 days in every cluster")
    k = int(supported.silhouette.idxmax())
    model = models[k]
    means = data.assign(raw_cluster=model.labels_).groupby("raw_cluster").delay_rate.mean()
    order = {int(label): rank for rank, label in enumerate(means.sort_values().index)}
    labels = pd.Series([order[int(x)] for x in model.labels_], index=data.index, name="cluster").astype("Int64")
    stability = []
    for seed in [0, 1, 2, 3, 4]:
        alternate = KMeans(n_clusters=k, random_state=seed, n_init=20).fit_predict(scaled)
        stability.append({"seed": seed, "adjusted_rand_index": adjusted_rand_score(model.labels_, alternate)})
    return labels, scores, pd.DataFrame(stability), scaler, model, order


def historical_flags(daily, window=28, min_periods=21, multiplier=2.0):
    """Threshold = prior-day rolling mean + multiplier * max(prior std, .02)."""
    if not daily.index.is_unique or not daily.index.is_monotonic_increasing:
        raise ValueError("Daily index must be unique and chronological")
    if not daily.index.equals(pd.date_range(daily.index.min(), daily.index.max(), freq="D", name=daily.index.name)):
        raise ValueError("Daily summary must contain consecutive calendar days")
    result = pd.DataFrame(index=daily.index)
    for metric in ["delay_rate", "missing_rate"]:
        values = daily[metric].where(daily.eligible) if metric == "delay_rate" else daily[metric]
        history = values.shift(1).rolling(window, min_periods=min_periods)
        center = history.mean()
        spread = history.std(ddof=1).clip(lower=.02)
        threshold = center + multiplier * spread
        ready = threshold.notna() & values.notna()
        result[f"{metric}_baseline"] = center
        result[f"{metric}_threshold"] = threshold
        result[f"{metric}_score"] = (values - center) / spread
        result[f"{metric}_flag"] = values.gt(threshold).astype("boolean").where(ready)
    return result
