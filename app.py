"""Streamlit presentation of the saved NYC Flights 2013 research."""

import json

import altair as alt
import pandas as pd
import streamlit as st

import dashboard_data as data

TEAL = '#006D70'
BLUE = '#315E87'
ORANGE = '#AB4B21'
GRAY = '#64788A'
MISSING_VALUE = '—'
KNOWN_DELAYS = 'Known departure delays'
UNKNOWN_DELAYS = 'Unknown departure delays'
ORIGINS = {'EWR': 'Newark (EWR)', 'JFK': 'John F. Kennedy (JFK)', 'LGA': 'LaGuardia (LGA)'}
DATE = alt.X('date:T', title='Scheduled departure date', axis=alt.Axis(format='%b %d'))


@st.cache_data(show_spinner=False)
def cached_csv(path, signature, **options):
    return pd.read_csv(path, **options)


@st.cache_data(show_spinner=False)
def cached_json(path, signature):
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def read_artifact(relative, **options):
    try:
        path = data.artifact_path(data.ROOT, relative)
        stats = path.stat()
        signature = (stats.st_mtime_ns, stats.st_size)
        if path.suffix == '.json':
            return cached_json(str(path), signature)
        return cached_csv(str(path), signature, **options)
    except (OSError, ValueError) as error:
        st.warning(str(error) if isinstance(error, FileNotFoundError) else
                   f'Could not read `{relative}`. Check the saved file against the artifact instructions in README.md. {error}')
        return None


def date_range(frame, label, key):
    if frame.empty:
        return ()
    earliest, latest = frame.date.min().date(), frame.date.max().date()
    return st.date_input(label, (earliest, latest), min_value=earliest, max_value=latest,
                         format='YYYY-MM-DD', key=key)


def rate(value):
    return MISSING_VALUE if pd.isna(value) else f'{value:.2%}'


def draw(chart):
    st.altair_chart(chart, width='stretch')


def formatted_table(frame, formats, **options):
    st.dataframe(frame.style.format(formats, na_rep=MISSING_VALUE), hide_index=True, width='stretch', **options)


def reliability_table(frame, key, label):
    groups = data.grouped_reliability(frame, key).sort_values('flights', ascending=False)
    if key == 'origin':
        groups[key] = groups[key].map(ORIGINS).fillna(groups[key])
    columns = {key: label, 'flights': 'Flights', 'observed': KNOWN_DELAYS, 'unknown': UNKNOWN_DELAYS,
               'delayed': '15+ min late', 'delay_rate': 'Delay rate'}
    formats = dict.fromkeys(('Flights', KNOWN_DELAYS, UNKNOWN_DELAYS, '15+ min late'), '{:,.0f}')
    formatted_table(groups.rename(columns=columns), {**formats, 'Delay rate': '{:.2%}'})


def reliability():
    st.header('Flight and airline reliability')
    st.caption('Filters apply to this section’s flight-level summaries and charts.')
    flights = read_artifact(data.FLIGHTS, usecols=data.FLIGHT_COLUMNS, parse_dates=['date'])
    if flights is None:
        return
    if flights.empty:
        st.info('The prepared flight dataset has no rows.')
        return
    left, center, right = st.columns([2, 1, 1])
    with left:
        dates = date_range(flights, 'Flight date range', 'flight_dates')
    with center:
        carrier = st.selectbox('Airline', [None, *sorted(flights.carrier.dropna().unique())],
                               format_func=lambda value: value or 'All airlines', key='flight_carrier')
    with right:
        origin = st.selectbox('Origin airport', [None, *sorted(flights.origin.dropna().unique())],
                               format_func=lambda value: ORIGINS.get(value, value or 'All airports'), key='flight_origin')
    selected = data.filter_flights(flights, dates, carrier, origin)
    summary = data.flight_summary(selected)
    for column, label, value in zip(st.columns(4),
                                   ['Total flights', KNOWN_DELAYS, UNKNOWN_DELAYS, 'Departed 15+ min late'],
                                   [f"{summary['flights']:,}", f"{summary['observed']:,}",
                                    f"{summary['unknown']:,}", rate(summary['delay_rate'])]):
        column.metric(label, value, border=True)
    st.caption('Delay rate = flights departing at least 15 minutes late ÷ flights with a known departure delay. '
               'Unknown means the dataset has no departure-delay value for that flight.')
    if selected.empty:
        st.info('No flights match these filters. Choose a complete date range or broaden the airline or airport selection.')
        return

    st.subheader('Departure-delay summary')
    if summary['observed']:
        statistics = summary['delay_statistics']
        for column, (label, value) in zip(st.columns(len(statistics)), statistics.items()):
            column.metric(label, f'{value:,.1f} min')
        st.caption('Summary uses all observed values in the selection. Negative delays are departures ahead of schedule.')
    else:
        st.info('This selection has no observed departure delays. Delay rates and severity are unavailable.')

    st.subheader('Departure delay over time')
    interval = st.radio('Group by', ['Day', 'Week', 'Month'], index=1, horizontal=True, key='trend_interval')
    periods = selected.date.dt.to_period({'Day': 'D', 'Week': 'W-SUN', 'Month': 'M'}[interval])
    trend = data.grouped_reliability(selected.assign(date=periods.dt.start_time), 'date')
    if trend.delay_rate.notna().any():
        draw(alt.Chart(trend).mark_line(color=TEAL, point=True, invalid='break-paths-show-domains').encode(
            DATE, alt.Y('delay_rate:Q', title='Departed 15+ min late', axis=alt.Axis(format='%')),
            tooltip=[alt.Tooltip('date:T', title='Period starting', format='%Y-%m-%d'),
                     alt.Tooltip('delay_rate:Q', title='Delay rate', format='.2%'),
                     alt.Tooltip('observed:Q', title=KNOWN_DELAYS, format=','),
                     alt.Tooltip('unknown:Q', title=UNKNOWN_DELAYS, format=',')]).properties(height=265))
        st.caption('Counts use only selected flights. Weeks run Monday–Sunday and may be partial at the filter boundaries.')

    st.subheader('Airline and origin comparisons')
    st.caption('These observed differences can reflect schedule, route, and flight mix.')
    st.markdown('**By airline**')
    reliability_table(selected, 'carrier', 'Airline')
    st.markdown('**By origin airport**')
    reliability_table(selected, 'origin', 'Origin')

    st.subheader('Departure-delay distribution')
    if summary['observed']:
        histogram, excluded = data.departure_histogram(selected)
        draw(alt.Chart(histogram).mark_bar(color=BLUE).encode(
            x=alt.X('lower:Q', title='Departure delay (minutes)', bin='binned'), x2='upper:Q',
            y=alt.Y('flights:Q', title='Observed flights'),
            tooltip=[alt.Tooltip('lower:Q', title='Bin start'), alt.Tooltip('upper:Q', title='Bin end'),
                     alt.Tooltip('flights:Q', title='Flights', format=',')]).properties(height=245))
        st.caption(f'Display range −60 to 240 minutes. {excluded:,} observed delays fall outside this chart and remain in the numerical summaries.')


def show_flag_table(rows, flag, metric):
    if rows.empty:
        st.info('No flagged dates in this display range.')
        return
    table = rows[['date', 'observed_labels', metric, f'{metric}_threshold', flag]].copy()
    table[flag] = data.flag_labels(table[flag])
    table['date'] = table.date.dt.date
    table.columns = ['Date', KNOWN_DELAYS, 'Rate', 'Past-only threshold', 'Flag']
    formatted_table(table, {KNOWN_DELAYS: '{:,.0f}', 'Rate': '{:.2%}', 'Past-only threshold': '{:.2%}'})


def draw_network_timeline(days, metric, title, include_clusters=False):
    plot = days.assign(
        cluster_label=days.cluster.map({0: 'Cluster 0', 1: 'Cluster 1'}).fillna('Unassigned'),
        flag_label=data.flag_labels(days[f'{metric}_flag']),
    )
    y = alt.Y(f'{metric}:Q', title=title, axis=alt.Axis(format='%'))
    trend = alt.Chart(plot).mark_line(color='#A0AFBA').encode(DATE, y)
    tooltip = [alt.Tooltip('date:T', title='Date', format='%Y-%m-%d'),
               alt.Tooltip(f'{metric}:Q', title='Observed rate', format='.2%'),
               alt.Tooltip(f'{metric}_threshold:Q', title='Past-only threshold', format='.2%'),
               alt.Tooltip('observed_labels:Q', title=KNOWN_DELAYS, format=','),
               alt.Tooltip('flag_label:N', title='Saved flag')]
    points = alt.Chart(plot).mark_circle(size=31, opacity=.85).encode(DATE, y, tooltip=tooltip)
    if include_clusters:
        points = points.encode(color=alt.Color(
            'cluster_label:N', title='Daily cluster', scale=alt.Scale(
                domain=['Cluster 0', 'Cluster 1', 'Unassigned'], range=[TEAL, BLUE, GRAY])))
    else:
        points = points.mark_circle(size=28, color=GRAY, opacity=.8)
    threshold = alt.Chart(plot).mark_line(color=GRAY, strokeDash=[5, 4], invalid='break-paths-show-domains').encode(
        DATE, alt.Y(f'{metric}_threshold:Q', title=title))
    flags = alt.Chart(plot.loc[plot.flag_label.eq('Flagged')]).mark_point(
        color=ORANGE, filled=True, shape='diamond', size=105, stroke='white', strokeWidth=1).encode(
        DATE, y, tooltip=tooltip)
    draw((trend + threshold + points + flags).properties(height=245))


def disruption():
    st.header('Daily disruption patterns')
    st.markdown(f'**{data.NETWORK_SCOPE}**')
    st.caption('The date range changes which saved days are displayed. Cluster profiles always describe the full year.')
    days = read_artifact(data.DAYS, parse_dates=['date'], dtype=data.DAILY_DTYPES)
    if days is None or days.empty:
        if days is not None:
            st.info('The saved daily patterns table has no dates.')
        return
    dates = date_range(days, 'Network display date range', 'network_dates')
    visible = data.network_days(days, dates)
    if visible.empty:
        st.info('Choose a complete date range to see the saved network days.')
        return

    st.subheader('Daily delay rate and label coverage')
    st.caption('Diamonds mark saved anomalies. Dashed lines show thresholds calculated from preceding days.')
    draw_network_timeline(visible, 'delay_rate', 'Departed 15+ min late', include_clusters=True)
    draw_network_timeline(visible, 'missing_rate', UNKNOWN_DELAYS)

    st.subheader('Two descriptive clusters')
    st.caption('Cluster 0 groups lower-disruption days. Cluster 1 groups higher-disruption days. '
               'The profiles are day averages for the full NYC network and describe patterns without establishing their causes.')
    eligible = int(days.eligible.sum())
    unassigned = int(days.cluster.isna().sum())
    st.caption(f'Across 2013, {eligible:,} of {len(days):,} dates were eligible and {unassigned:,} were unassigned. '
               'A date needed at least 500 known departure delays covering at least 50% of its flights. '
               'Departure delay rate, median delay, 90th-percentile delay, and flight count all had to be available.')
    profiles = read_artifact(data.PROFILES)
    if profiles is not None:
        names = {'cluster': 'Cluster', 'days': 'Days', 'flights_mean': 'Mean flights per day',
                 'delay_rate_mean': 'Mean delay rate', 'median_delay_mean': 'Mean daily median (min)',
                 'p90_delay_mean': 'Mean daily P90 (min)', 'missing_rate_mean': 'Mean unknown share'}
        formatted_table(profiles.rename(columns=names), {
            'Days': '{:,.0f}', 'Mean flights per day': '{:,.1f}',
            'Mean delay rate': '{:.2%}', 'Mean unknown share': '{:.2%}',
            'Mean daily median (min)': '{:.1f}', 'Mean daily P90 (min)': '{:.1f}',
        })
    st.caption('These full-year outcomes are for description and stay out of the pre-departure classifier.')

    with st.expander('Delay and missing-label anomaly flags'):
        for path, heading, flag, metric in [
            (data.DELAY_EVENTS, 'Delay anomalies', 'delay_rate_flag', 'delay_rate'),
            (data.MISSING_EVENTS, 'Missing-label anomalies', 'missing_rate_flag', 'missing_rate'),
        ]:
            st.markdown(f'**{heading}**')
            events = read_artifact(path, parse_dates=['date'], dtype=data.DAILY_DTYPES)
            if events is not None:
                show_flag_table(data.network_days(events, dates), flag, metric)
    with st.expander('Daily assignments and flag status'):
        st.caption('Unknown means there was insufficient prior history or the day was ineligible for delay scoring.')
        display = visible[['date', 'cluster', 'delay_rate_flag', 'missing_rate_flag']].copy()
        display['date'] = display.date.dt.date
        display['cluster'] = display.cluster.astype('string').fillna('Unassigned')
        display['delay_rate_flag'] = data.flag_labels(display.delay_rate_flag)
        display['missing_rate_flag'] = data.flag_labels(display.missing_rate_flag)
        display.columns = ['Date', 'Cluster', 'Delay flag', 'Missing-label flag']
        st.dataframe(display, hide_index=True, width='stretch')

    st.subheader('Unusual days in the saved case studies')
    cases = read_artifact(data.CASES, dtype=data.DAILY_DTYPES)
    if cases is None:
        return
    # The notebook serialized the date index as the first, unnamed CSV column.
    cases = cases.rename(columns={'Unnamed: 0': 'date'})
    cases['date'] = pd.to_datetime(cases.date)
    cases = data.network_days(cases, dates).sort_values('date')
    if cases.empty:
        st.info('No saved case-study dates fall in this range.')
    for row in cases.itertuples():
        delay_flag = data.flag_labels(pd.Series([row.delay_rate_flag])).iat[0]
        missing_flag = data.flag_labels(pd.Series([row.missing_rate_flag])).iat[0]
        delay_detail = ('Day ineligible for delay flagging.' if pd.isna(row.delay_rate_flag) and not row.eligible else
                        f'Delay flag {delay_flag.lower()}. Past-only delay threshold {rate(row.delay_rate_threshold)}.')
        st.markdown(f'**{row.date:%b %d, %Y}** · {rate(row.delay_rate)} late among flights with a known delay · '
                    f'{rate(row.missing_rate)} with unknown departure delay. {delay_detail} '
                    f'Missing-label flag {missing_flag.lower()}.')


def show_models(rows):
    if not rows:
        st.info('No saved results are available for this comparison.')
        return
    formats = {**dict.fromkeys(data.METRICS.values(), '{:.4f}'), 'n': '{:,.0f}', 'Cutoff': '{:.3f}'}
    formatted_table(data.model_table(rows), formats, column_config={'Role': st.column_config.TextColumn(width='medium')})
    with st.expander('Saved metric sources and row identifiers'):
        references = data.source_table(rows)
        st.dataframe(references, hide_index=True, width='stretch')
        missing = [file for file in references['Saved file'].unique() if not (data.ROOT / file).is_file()]
        if missing:
            st.warning('Some source tables are absent locally. Restore these saved files using README.md. '
                       + ', '.join(f'`{file}`' for file in missing))


def original_models(summary):
    st.subheader('Original four-model experiment')
    st.caption('Schedule-only departure prediction. Fit January–August 2013, selection September–October, '
               'and frozen assessment November–December.')
    selection = read_artifact(data.SELECTION)
    if selection is not None:
        name = data.MODEL_NAMES.get(selection['selected_model'], selection['selected_model'])
        st.markdown(f"**Validation-selected model** · {name} at cutoff **{selection['threshold']:.3f}**. "
                    f"Selection used {selection['selection_metric']}.")
        st.caption(f'Selection record · {data.SELECTION}')
    comparison = read_artifact(data.ORIGINAL)
    if comparison is not None:
        average_precision = read_artifact(data.ORIGINAL_AP, usecols=['model', 'split', 'average_precision'])
        rows = data.original_results(comparison, selection or {}, average_precision)
    else:
        rows = [row for row in summary if row['group'] == 'original_historical']
        if rows:
            st.info('Showing the canonical rows for the original selected model. The four-model comparison needs its saved table.')

    window = st.selectbox('Original assessment period',
                          ['November–December · frozen assessment', 'September–October · selection period'],
                          key='original_period')
    split = 'test' if window.startswith('November') else 'validation'
    show_models([row for row in rows if row['assessment_period'] == data.ORIGINAL_SPLITS[split]])
    st.caption('The majority baseline predicts the more common class, on time or less than 15 minutes late. '
               'It can have higher accuracy while identifying zero delayed flights. Its precision is undefined '
               'because it makes no positive predictions.')
    st.caption('Lowering a cutoff flags more flights and can improve recall while reducing precision. '
               'The selection period fixed the cutoffs before the frozen assessment.')


def schedule_models(rows, canonical):
    st.subheader('Schedule-only development')
    st.caption('The 14-configuration study compared cyclic time encoding, carrier-qualified flight numbers, '
               'and feature interactions. Its interaction forest led the internal folds. Its September–October '
               'results reuse that period for cutoff selection.')
    st.caption('The seven-candidate audit compared compact logistic regression (F) and compact boosting (G) '
               'on the same inputs. Those inputs exclude month, day of year, and flight number.')
    ranking = canonical.get('grid_selection')
    if ranking:
        candidate = ranking['source']['row']['candidate']
        st.markdown(f"**Schedule grid selection** · {data.MODEL_NAMES.get(candidate, candidate)} "
                    f"led folds A–C with mean F1 **{ranking['mean_f1']:.4f}**. "
                    'Fixed compact boosting subsequently had higher F1 in fold D.')
        st.caption(f"Saved ranking · {ranking['source']['file']}")
    show_models(rows)
    st.markdown('**Calibration decision**')
    st.caption('The intercept-only adjustment improved probability error in some months, but the gains did not '
               'transfer consistently. Calibration was not adopted. Treat raw scores as decision scores, not '
               'reliable probabilities for individual flights.')


def operational_models(rows):
    st.subheader('One hour before scheduled departure')
    selected = next((row for row in rows if row['role'] == 'selected_before_assessment'), None)
    if selected:
        model = data.MODEL_NAMES.get(selected['model'], selected['model'])
        st.markdown(f"**Selected in August** · {model} at cutoff **{selected['threshold']:.3f}**.")
    st.caption('Operational logistic regression had the highest observed F1 in both later assessments among '
               'carried-forward candidates. August selected operational boosting. The schedule-only row uses '
               'published information at the same checkpoint.')
    st.caption('The operational replay uses recent reports, waiting counts and matched prior-service information '
               'as reconstructed at each cutoff. It assumes departure and arrival reports became available at '
               'their reconstructed event times.')
    show_models(rows)


def departure_models(rows, agreement):
    st.subheader('Departure status near and after schedule')
    st.caption('The target is departure delay of at least 15 minutes. Unresolved flights exclude those already '
               'reported departed less than 15 minutes late. Available status changes as the checkpoint moves.')
    if rows:
        checkpoints = list(dict.fromkeys(row['prediction_time'] for row in rows))
        plus_ten = '10 minutes after scheduled departure'
        chosen = st.selectbox('Departure checkpoint', checkpoints,
                              index=checkpoints.index(plus_ten) if plus_ten in checkpoints else 0,
                              key='departure_checkpoint')
        show_models([row for row in rows if row['prediction_time'] == chosen])
    else:
        show_models(rows)

    if agreement:
        counts = ' and '.join(f"{row['different_decisions']:,} of {row['n']:,} flights in {row['source']['row']['period']}"
                              for row in agreement)
        st.markdown('**About 97% accuracy at +10 minutes** · The selected model essentially matched the simple '
                    f'waiting rule. Their decisions differed on {counts}. Only five minutes remained until '
                    'the 15-minute delay definition, so this result does not establish strong advance prediction.')
        st.caption(f"Saved rule agreement · {agreement[0]['source']['file']}")


def arrival_models(rows):
    st.subheader('Arrival delay at scheduled arrival')
    st.caption('The target is arrival delay of at least 15 minutes at the destination-local scheduled arrival. '
               'The model uses departure information and recent reported arrivals. Unresolved flights are those '
               'without an arrival report by that checkpoint. Compare rows within the same population and period.')
    show_models(rows)


def models():
    st.header('Models and findings')
    st.caption('The original November–December assessment used the frozen original model and cutoffs. Later '
               'September–October and November–December replays are exploratory comparisons on previously '
               'examined periods. Metrics are shown by assessment period and population.')
    canonical = read_artifact(data.RESULTS) or {}
    rows = canonical.get('results', [])
    original, schedule, operational, status, arrival = st.tabs([
        'Original experiment', 'Schedule-only development', 'One-hour operational',
        'Departure status', 'Scheduled arrival',
    ])
    with original:
        original_models(rows)
    with schedule:
        schedule_models([row for row in rows if row['group'] == 'schedule_development'], canonical)
    with operational:
        operational_models([row for row in rows if row['group'] == 'operational_one_hour'])
    with status:
        departure_models([row for row in rows if row['group'] == 'status_assisted_departure'],
                         canonical.get('waiting_rule_agreement_at_plus_10_minutes', []))
    with arrival:
        arrival_models([row for row in rows if row['group'] == 'scheduled_arrival'])
    st.caption('F1 balances precision and recall. ROC-AUC and average precision summarize ranking. '
               'A dash marks an unavailable or undefined metric.')


def main():
    st.set_page_config(page_title='NYC Flight Delays & Airline Reliability', page_icon='✈️', layout='wide')
    st.html("""
    <style>
        [data-testid="stMainBlockContainer"] { max-width: 1280px; padding-top: 2.5rem; }
        .research-header { background: linear-gradient(116deg,#102C41,#16535E); color: #fff;
            border-radius: 16px; padding: clamp(1.25rem, 4vw, 2.6rem); margin-bottom: .65rem; }
        .research-header h1 { color: #fff; font-size: clamp(1.65rem, 3.2vw, 2.8rem);
            max-width: 960px; line-height: 1.15; margin: 0; padding: 0; }
        .research-header p { color: #CAF2EF; font-size: 1rem; margin: 0 0 .65rem; }
    </style>
    <header class="research-header">
        <p>NYC Flights 2013 · JFK, LGA and EWR</p>
        <h1>Predicting Flight Delays and Analyzing Airline Reliability Over Time</h1>
    </header>
    """)
    sections = {'Flight reliability': reliability, 'Daily disruption': disruption, 'Models and findings': models}
    selected = st.radio('Dashboard section', list(sections), horizontal=True, key='section',
                        label_visibility='collapsed')
    sections[selected]()


if __name__ == '__main__':
    main()
