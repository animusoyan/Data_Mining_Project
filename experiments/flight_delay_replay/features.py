"""Schedule and point-in-time features for the historical replays."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.independent_audit.core import roles
from src.prepare import ROOT, clock_minutes, prepare


METADATA = Path(__file__).with_name('airport_time_zones.csv')
NY = 'America/New_York'
MINUTE = pd.Timedelta(minutes=1)
DAY = pd.Timedelta(days=1)
NUMERIC, CATEGORICAL = roles('F')
SCHEDULE = NUMERIC + CATEGORICAL
SERVICE = ['carrier', 'flight', 'origin', 'dest']


def load_snapshot(root=ROOT):
    source = root / 'data/raw/flights.csv'
    manifest = json.loads((source.parent / 'source.json').read_text(encoding='utf-8'))
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    if checksum != manifest['sha256']:
        raise ValueError('Cached flights.csv does not match the repository snapshot manifest')
    zones = pd.read_csv(METADATA).set_index('airport').zone
    return prepare(pd.read_csv(source)), zones, checksum


def local_clock(date, hhmm):
    """Represent 2400 as the next day's midnight."""
    offset = clock_minutes(hhmm).astype(float) + hhmm.eq(2400) * 1440
    return date + pd.to_timedelta(offset, unit='min')


def utc_minutes(local, zones, *, daylight=True):
    """Localize each destination separately before converting to epoch minutes."""
    if zones.isna().any():
        raise ValueError('Every scheduled airport requires an IANA time zone')
    result = pd.Series(np.nan, index=local.index)
    for zone, group in zones.groupby(zones):
        stamps = local.loc[group.index].dt.tz_localize(
            zone, ambiguous=daylight, nonexistent='shift_forward').dt.tz_convert('UTC')
        result.loc[group.index] = (stamps - pd.Timestamp('1970-01-01', tz='UTC')) / MINUTE
    return result


def schedule_times(flights, zones):
    origins, destinations = flights.origin.map(zones), flights.dest.map(zones)
    departure_local = local_clock(flights.date, flights.sched_dep_time)
    departure = utc_minutes(departure_local, origins)
    arrival_local = local_clock(flights.date, flights.sched_arr_time)
    arrival = utc_minutes(arrival_local, destinations)
    # The earliest destination-local occurrence after the scheduled departure.
    # A repeated fall-back clock can have a second occurrence on the same date.
    repeated = utc_minutes(arrival_local, destinations, daylight=False)
    arrival = arrival.where(arrival.gt(departure), repeated)
    overnight = arrival.le(departure)
    arrival.loc[overnight] = utc_minutes(arrival_local.loc[overnight] + DAY,
                                        destinations.loc[overnight])
    if arrival.le(departure).any():
        raise ValueError('Scheduled arrival must follow scheduled departure')
    midnight = utc_minutes(flights.date, origins)
    return pd.DataFrame({'departure': departure, 'arrival': arrival, 'midnight': midnight})


def report_time(local_schedule, observed_clock, delay, zones, scheduled):
    """Use the reported local delay to recover the observation's local date.

    The local clock anchors the timestamp. On the repeated DST hour choose
    the occurrence closest to schedule plus the reported delay.
    """
    local_event = local_clock((local_schedule + pd.to_timedelta(delay, unit='min')).dt.normalize(),
                              observed_clock.mask(observed_clock.eq(2400), 0))
    expected = local_schedule + pd.to_timedelta(delay, unit='min')
    delta = (local_event - expected) / MINUTE
    local_event = local_event + DAY * delta.lt(-720) - DAY * delta.gt(720)
    daylight = utc_minutes(local_event, zones)
    standard = utc_minutes(local_event, zones, daylight=False)
    estimate = scheduled + delay
    return daylight.where((daylight - estimate).abs().le((standard - estimate).abs()), standard)


def event_times(flights, schedule, zones):
    origins, destinations = flights.origin.map(zones), flights.dest.map(zones)
    departure_local = local_clock(flights.date, flights.sched_dep_time)
    arrival_local = pd.Series(pd.NaT, index=flights.index, dtype='datetime64[ns]')
    for zone, group in destinations.groupby(destinations):
        arrival_local.loc[group.index] = (pd.to_datetime(schedule.arrival.loc[group.index], unit='m', utc=True)
                                          .dt.tz_convert(zone).dt.tz_localize(None))
    return pd.DataFrame({
        'departed': report_time(departure_local, flights.dep_time, flights.dep_delay,
                               origins, schedule.departure),
        'arrived': report_time(arrival_local, flights.arr_time, flights.arr_delay,
                              destinations, schedule.arrival),
    })


def as_of(report, cutoff):
    return report.notna() & report.le(cutoff)


def window_stats(groups, report, start, end, values=None, exclude_self=True):
    """Query matching observed events in (start, end], optionally excluding the query flight."""
    result = np.zeros((len(groups), 3))
    target = None if values is None else values.to_numpy(dtype=float)
    report, start, end = (column.to_numpy(dtype=float) for column in (report, start, end))
    for positions in groups.groupby(groups, sort=False).indices.values():
        indices = positions[np.isfinite(report[positions])]
        if target is not None:
            indices = indices[np.isfinite(target[indices])]
        indices = indices[np.argsort(report[indices], kind='stable')]
        timestamps = report[indices]
        left = np.searchsorted(timestamps, start[positions], side='right')
        right = np.searchsorted(timestamps, end[positions], side='right')
        own = (report[positions] > start[positions]) & (report[positions] <= end[positions])
        if target is not None:
            own &= np.isfinite(target[positions])
        own &= exclude_self
        count = right - left - own
        result[positions, 0] = count
        if target is not None:
            for column, attribute in enumerate((target, target >= 15), 1):
                prefix = np.r_[0., np.cumsum(attribute[indices])]
                totals = prefix[right] - prefix[left] - np.where(own, attribute[positions], 0.)
                result[positions, column] = np.divide(totals, count, out=np.full(len(count), np.nan),
                                                      where=count > 0)
    return pd.DataFrame(result, index=groups.index, columns=['count', 'mean_delay', 'late_rate'])


def waiting_counts(groups, scheduled, departed, cutoff, window=120):
    """Count scheduled flights within the last window whose reports have yet to arrive."""
    enters = scheduled.where(departed.isna() | departed.gt(scheduled))
    exits = pd.concat([scheduled + window, departed], axis=1).min(axis=1).where(enters.notna())
    lower = pd.Series(-np.inf, index=groups.index)
    entered = window_stats(groups, enters, lower, cutoff)['count']
    exited = window_stats(groups, exits, lower, cutoff)['count']
    return entered - exited


def rotation_matches(flights, schedule, events):
    """Learn the preceding NYC service by completed historical tail-number pairs."""
    services, _ = pd.factorize(pd.MultiIndex.from_frame(flights[SERVICE]))
    next_midnight = utc_minutes(flights.date + DAY, pd.Series(NY, index=flights.index))
    same_day = events.departed.ge(schedule.midnight) & events.arrived.lt(next_midnight)
    history = pd.DataFrame({'tail': flights.tailnum, 'service': services, 'day': flights.date,
                            'dep': events.departed, 'arr': events.arrived})
    history = history.loc[same_day].dropna().sort_values(['dep', 'service'], kind='stable')
    previous = history.groupby(['day', 'tail'])[['service', 'arr']].shift()
    complete = history.arr.ge(history.dep) & previous.arr.le(history.dep)
    eligible = history.loc[complete, 'day'] + DAY
    pairs = pd.DataFrame({'available': eligible, 'service': history.loc[complete, 'service'],
                          'earlier': previous.loc[complete, 'service'].astype(int)})
    changes = pairs.groupby(['available', 'service', 'earlier']).size()
    updates = defaultdict(list)
    for (day, main, earlier), count in changes.items():
        updates[day].append((main, earlier, count))

    counts = defaultdict(Counter)
    best = {}
    matches = np.full(len(flights), -1, dtype=int)
    scheduled = schedule.departure.to_numpy()
    for day, positions in flights.groupby('date', sort=True).indices.items():
        for main, earlier, count in updates[day]:
            choices = counts[main]
            choices[earlier] += count
            old = best.get(main, earlier)
            best[main] = max((old, earlier), key=lambda option: (choices[option], -option))
        today = {}
        for pos in positions[np.argsort(scheduled[positions], kind='stable')]:
            current = services[pos]
            earlier = today.get(best.get(current))
            if earlier is not None and scheduled[earlier] < scheduled[pos]:
                matches[pos] = earlier
            today[current] = pos
    return matches


def add_stats(frame, prefix, stats):
    for column in stats:
        frame[f'{prefix}_{column}'] = stats[column]


def departure_features(flights, schedule, events, cutoff, rotations):
    features = flights[SCHEDULE].copy()
    blocks = {'schedule': list(features)}
    origin, route, airline = flights.origin, flights.route, flights.carrier
    prior = (schedule.midnight - 1).clip(upper=cutoff)
    add_stats(features, 'prior_origin', window_stats(origin, events.departed, prior - 7 * 1440,
                                                      prior, flights.dep_delay))
    features['scheduled_origin_count'] = window_stats(
        origin, schedule.departure, schedule.departure - 60, schedule.departure + 60)['count']
    blocks['prior_day_and_traffic'] = list(features)
    for window in (30, 60, 120):
        add_stats(features, f'recent_origin_{window}', window_stats(
            origin, events.departed, cutoff - window, cutoff, flights.dep_delay))
    blocks['recent_reports'] = list(features)
    features['waiting_origin'] = waiting_counts(origin, schedule.departure, events.departed, cutoff)
    blocks['waiting'] = list(features)
    for name, group in [('route', route), ('carrier', airline)]:
        add_stats(features, f'prior_{name}', window_stats(group, events.departed, prior - 14 * 1440,
                                                           prior, flights.dep_delay))
        add_stats(features, f'recent_{name}', window_stats(group, events.departed, cutoff - 120,
                                                            cutoff, flights.dep_delay))
        features[f'waiting_{name}'] = waiting_counts(group, schedule.departure, events.departed, cutoff)
        features[f'scheduled_{name}_count'] = window_stats(
            group, schedule.departure, schedule.departure - 60, schedule.departure + 60)['count']
    blocks['route_and_airline'] = list(features)
    earlier = rotations.clip(min=0)
    earlier_schedule = pd.Series(schedule.departure.to_numpy()[earlier], index=flights.index)
    earlier_dep = pd.Series(events.departed.to_numpy()[earlier], index=flights.index)
    earlier_arr = pd.Series(events.arrived.to_numpy()[earlier], index=flights.index)
    known = (rotations >= 0) & cutoff.ge(schedule.midnight)
    dep_seen = known & as_of(earlier_dep, cutoff)
    arr_seen = known & as_of(earlier_arr, cutoff)
    features['prior_service_known'] = known.astype(int)
    features['prior_service_scheduled_gap'] = (schedule.departure - earlier_schedule).where(known)
    features['prior_service_departed'] = dep_seen.astype(int)
    features['prior_service_dep_delay'] = pd.Series(flights.dep_delay.to_numpy()[earlier],
                                                   index=flights.index).where(dep_seen)
    features['prior_service_arrived'] = arr_seen.astype(int)
    features['prior_service_arr_delay'] = pd.Series(flights.arr_delay.to_numpy()[earlier],
                                                   index=flights.index).where(arr_seen)
    blocks['prior_service'] = list(features)
    return features, blocks


def arrival_features(flights, schedule, events):
    cutoff = schedule.arrival
    frame = flights[SCHEDULE].copy()
    frame['scheduled_duration'] = schedule.arrival - schedule.departure
    departed = as_of(events.departed, cutoff)
    frame['departure_confirmed'] = departed.astype(int)
    frame['departure_delay_reported'] = flights.dep_delay.where(departed)
    frame['minutes_since_departure'] = (cutoff - events.departed).where(departed)
    core = list(frame)
    for name, group in [('destination', flights.dest), ('route', flights.route), ('carrier', flights.carrier)]:
        for window in (60, 180):
            add_stats(frame, f'{name}_arrivals_{window}', window_stats(
                group, events.arrived, cutoff - window, cutoff, flights.arr_delay))
    return frame, {'schedule_and_departure': core, 'recent_arrivals': list(frame)}
