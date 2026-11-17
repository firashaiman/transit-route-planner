"""Transit Insight. Run with: streamlit run route_planner.py."""

import datetime as dt
import math
import re
from pathlib import Path

import pandas as pd

CSV_PATH = Path(__file__).with_name("route_planner_trips.csv")
DAY_OPTIONS = {
    "Weekday (Mon–Fri)": "Weekday",
    "Weekend / holiday": "Weekend/Holiday",
}
REQUIRED_COLUMNS = {
    "Trip_id", "stop_seq", "Stop_search", "Arrival_time", "Service_Days",
    "Route_long_name",
}


def parse_service_time(value):
    """Read a scheduled time without wrapping hours greater than 23."""
    match = re.fullmatch(r"(\d+):([0-5]\d)(?::([0-5]\d))?", str(value).strip())
    if match is None:
        return None
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds or 0)


def format_service_time(seconds):
    hours, remainder = divmod(int(seconds), 3600)
    minutes = remainder // 60
    days, hour = divmod(hours, 24)
    clock = f"{hour:02d}:{minutes:02d}"
    if remainder % 60:
        clock += f":{remainder % 60:02d}"
    return f"{clock} (+{days}d)" if days else clock


def find_column(frame, *names):
    names_by_case = {name.casefold(): name for name in frame.columns}
    return next(
        (names_by_case[name.casefold()] for name in names if name.casefold() in names_by_case),
        None,
    )


def prepare_schedule(frame):
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))
    frame = frame.copy()
    for column in ("Trip_id", "Stop_search", "Service_Days", "Route_long_name"):
        frame[column] = frame[column].astype("string").str.strip().replace("", pd.NA)
    frame["stop_seq"] = pd.to_numeric(frame["stop_seq"], errors="coerce")
    frame = frame.dropna(subset=list(REQUIRED_COLUMNS - {"Arrival_time"})).copy()
    frame["_arrival"] = frame["Arrival_time"].map(parse_service_time)
    departure_column = find_column(frame, "Departure_time", "DepartureTime", "Departure Time")
    departure = (
        frame[departure_column].map(parse_service_time)
        if departure_column else frame["_arrival"].copy()
    )
    frame["_departure_estimated"] = departure.isna()
    if departure_column is None:
        frame["_departure_estimated"] = True
    frame["_departure"] = departure.fillna(frame["_arrival"])
    for column in ("Stop_lat", "Stop_lon", "avg_rating"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce") if column in frame else math.nan
    frame["avg_rating"] = frame["avg_rating"].where(frame["avg_rating"].between(0, 5))
    if "main_sentiment" not in frame:
        frame["main_sentiment"] = pd.NA
    frame["main_sentiment"] = frame["main_sentiment"].astype("string").str.strip().str.title()
    frame["main_sentiment"] = frame["main_sentiment"].where(
        frame["main_sentiment"].isin(["Positive", "Neutral", "Negative"])
    )
    sources = [
        find_column(frame, "feed_id", "feed_name", "source_feed"),
        find_column(frame, "agency_id", "agency_name", "operator"),
        find_column(frame, "route_id"),
        find_column(frame, "Transport_type"),
    ]
    route_keys = list(dict.fromkeys(key for key in [*sources, "Route_long_name"] if key))
    trip_keys = ["Service_Days", *route_keys, "Trip_id"]
    frame = correct_ktm_order(frame, trip_keys)
    return unwrap_midnight(frame, trip_keys), route_keys


def correct_ktm_order(frame, trip_keys):
    """Repair KTM rows in this export whose sequence contradicts the timetable."""
    mode = find_column(frame, "Transport_type")
    if mode is None:
        return frame

    ktm = frame[frame[mode].astype("string").str.strip().str.upper().eq("KTM")].copy()
    if ktm.empty:
        return frame

    timed = ktm.dropna(subset=["_arrival", "_departure"])
    for _, trip in timed.groupby(trip_keys, dropna=False, sort=False):
        original = trip.sort_values("stop_seq")
        if original["_arrival"].is_monotonic_increasing:
            continue

        ordered = trip.sort_values(["_arrival", "_departure"])
        if ordered.duplicated(["_arrival", "_departure"]).any() or (
            ordered["_departure"].shift() > ordered["_arrival"]
        ).any() or (ordered["_arrival"] > ordered["_departure"]).any():
            frame = frame.drop(index=trip.index)
            continue

        # Do not combine recovered positions with rows missing timetable times.
        matched = ktm.loc[trip.index[0], trip_keys]
        same_trip = pd.Series(True, index=ktm.index)
        for key, value in matched.items():
            same_trip &= ktm[key].isna() if pd.isna(value) else ktm[key].eq(value)
        if int(same_trip.sum()) != len(trip):
            continue

        frame.loc[ordered.index, "stop_seq"] = range(len(ordered))
    return frame


def unwrap_midnight(frame, trip_keys):
    """Convert a near-midnight clock reset to elapsed service-day seconds."""
    day = 24 * 3600
    ordered = frame.sort_values([*trip_keys, "stop_seq"])
    timeline = ordered.set_index(trip_keys, append=True)[["_arrival", "_departure"]]
    # Stack arrivals and departures in event order, omitting missing events.
    events = timeline.stack().dropna()
    groups = list(range(1, len(trip_keys) + 1))
    previous = events.groupby(level=groups, dropna=False, sort=False).shift()
    rollover = (previous - events).ge(20 * 3600) & previous.mod(day).ge(22 * 3600) & events.le(2 * 3600)
    if not rollover.any():
        return frame
    crossed = rollover.groupby(level=groups, dropna=False, sort=False).cummax()
    needs_offset = crossed & events.lt(day)
    adjusted = (events.loc[needs_offset] + day).reset_index()
    for column in ("_arrival", "_departure"):
        updates = adjusted[adjusted.iloc[:, -2].eq(column)]
        frame.loc[updates.iloc[:, 0], column] = updates.iloc[:, -1].to_numpy()
    return frame


def read_schedule(path, modified_at):
    # The modification time invalidates the Streamlit cache when the CSV changes.
    return prepare_schedule(pd.read_csv(path, low_memory=False, dtype="string"))


def find_journeys(frame, route_keys, origin, destination, service_day, earliest):
    """Return the first feasible boarding per trip, ordered by departure."""
    day = frame[frame["Service_Days"].eq(service_day)]
    trip_keys = [*route_keys, "Trip_id"]
    candidate_keys = day.loc[day["Stop_search"].eq(origin), trip_keys].drop_duplicates()
    day = day.merge(candidate_keys, on=trip_keys)
    journeys = []
    for identity, trip in day.groupby(trip_keys, sort=False, dropna=False):
        trip = trip.sort_values("stop_seq", kind="stable").reset_index(drop=True)
        origins = trip.index[trip["Stop_search"].eq(origin)]
        destinations = trip.index[trip["Stop_search"].eq(destination)]
        if destinations.empty or trip["stop_seq"].duplicated().any():
            # Conflicting stop orders cannot be turned into a reliable itinerary.
            continue

        best = None
        for start in origins:
            depart = trip.at[start, "_departure"]
            if pd.isna(depart) or (earliest is not None and depart < earliest):
                continue

            for end in destinations[destinations > start]:
                arrive = trip.at[end, "_arrival"]
                if pd.isna(arrive) or arrive < depart:
                    continue
                segment = trip.iloc[start:end + 1]
                known_times = [
                    value for value in (
                        depart,
                        *(time for row in segment.iloc[1:-1][["_arrival", "_departure"]]
                          .itertuples(index=False, name=None) for time in row),
                        arrive,
                    ) if pd.notna(value)
                ]
                if any(first > second for first, second in zip(known_times, known_times[1:])):
                    continue
                order = (depart, arrive, len(segment))
                if best is None or order < best["order"]:
                    best = {
                        "route": str(trip.at[start, "Route_long_name"]),
                        "route_key": identity[:-1],
                        "depart": depart,
                        "arrive": arrive,
                        "departure_estimated": bool(trip.at[start, "_departure_estimated"]),
                        "segment": segment,
                        "order": order,
                    }
                break
        if best is not None:
            journeys.append(best)
    return sorted(journeys, key=lambda journey: (*journey["order"], journey["route"]))


def station_summary(segment):
    stations = segment.drop_duplicates("Stop_search")
    ratings = stations["avg_rating"].dropna()
    labels = stations["main_sentiment"].dropna().value_counts()
    if labels.empty:
        sentiment = "No review data"
    elif len(labels) > 1 and labels.iloc[0] == labels.iloc[1]:
        sentiment = "Mixed"
    else:
        sentiment = labels.index[0]
    return ratings.mean(), sentiment


def draw_map(st, pdk, segment):
    markers = []
    paths = []
    path = []
    for index, row in enumerate(segment.itertuples(index=False)):
        lat, lon = row.Stop_lat, row.Stop_lon
        if not (pd.notna(lat) and pd.notna(lon) and abs(lat) <= 90 and abs(lon) <= 180):
            if len(path) >= 2:
                paths.append({"path": path})
            path = []
            continue
        point = [lon, lat]
        path.append(point)
        color = [29, 112, 88] if index == 0 else (
            [181, 78, 45] if index == len(segment) - 1 else [245, 206, 139]
        )
        markers.append({"position": point, "stop": row.Stop_search, "color": color})
    if len(path) >= 2:
        paths.append({"path": path})
    if not markers:
        st.info("Map coordinates are unavailable for these stops. The timetable is shown below.")
        return

    longitude, latitude = zip(*(marker["position"] for marker in markers))
    span = max(max(latitude) - min(latitude), (max(longitude) - min(longitude)) * 1.35, 0.005)
    zoom = min(14, max(1, math.log2(360 / span) - 1.65))
    layers = [
        pdk.Layer("PathLayer", paths, get_color=[33, 112, 92], width_min_pixels=4),
        pdk.Layer(
            "ScatterplotLayer", markers, get_position="position", get_fill_color="color",
            get_radius=60, radius_min_pixels=5, radius_max_pixels=13, pickable=True,
            stroked=True, get_line_color=[24, 55, 46], line_width_min_pixels=1,
        ),
    ]
    st.pydeck_chart(pdk.Deck(
        map_provider="carto", map_style="light", layers=layers,
        initial_view_state=pdk.ViewState(
            latitude=(min(latitude) + max(latitude)) / 2,
            longitude=(min(longitude) + max(longitude)) / 2, zoom=zoom, pitch=0,
        ),
        tooltip={"text": "{stop}"},
    ))
    st.caption("Green marks the origin and orange marks the destination. Lines connect scheduled stops and do not trace the exact tracks or roads.")


def show_journey(st, pdk, journey):
    segment = journey["segment"]
    st.header(journey["route"])
    st.write(f"{segment.iloc[0]['Stop_search']} → {segment.iloc[-1]['Stop_search']}")

    departure_label = "Estimated departure" if journey["departure_estimated"] else "Departs"
    duration = math.ceil((journey["arrive"] - journey["depart"]) / 60)
    columns = st.columns(4)
    columns[0].metric(departure_label, format_service_time(journey["depart"]))
    columns[1].metric("Arrives", format_service_time(journey["arrive"]))
    columns[2].metric("Scheduled travel time", f"{duration} min")
    columns[3].metric("Stops incl. endpoints", len(segment))
    if journey["departure_estimated"]:
        st.caption("The CSV has no boarding-stop departure time for this trip, so its scheduled arrival is used as an estimate.")
    if journey["depart"] >= 86400 or journey["arrive"] >= 86400:
        st.caption("+1d means after midnight on the following calendar day, still within the selected service schedule.")

    map_tab, review_tab = st.tabs(["Route & timetable", "Station reviews"])
    with map_tab:
        draw_map(st, pdk, segment)
        rows = []
        for index, (name, arrival) in enumerate(zip(segment["Stop_search"], segment["_arrival"]), 1):
            if index == 1:
                station_time = format_service_time(journey["depart"])
                event = departure_label
            else:
                station_time = format_service_time(arrival) if pd.notna(arrival) else "—"
                event = "Arrival"
            rows.append({"Stop": index, "Station": name, "Scheduled time": station_time, "Event": event})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    with review_tab:
        rating, sentiment = station_summary(segment)
        left, right = st.columns(2)
        left.metric("Average station rating", f"{rating:.2f} / 5" if pd.notna(rating) else "Unavailable")
        right.metric("Most common station sentiment", sentiment)
        st.caption("Each station counts once. The rating averages available station ratings, without weighting by review count.")
        reviews = segment.drop_duplicates("Stop_search")[["Stop_search", "avg_rating", "main_sentiment"]]
        st.dataframe(reviews.rename(columns={
            "Stop_search": "Station", "avg_rating": "Rating / 5", "main_sentiment": "Sentiment",
        }), hide_index=True)


def main():
    import pydeck as pdk
    import streamlit as st

    st.set_page_config(page_title="Transit Insight | Route Planner", layout="wide")
    st.markdown("""
        <style>
          .transit-hero {
            padding: clamp(1.4rem, 4vw, 3rem);
            border-radius: 24px;
            background: linear-gradient(115deg, #143f35, #245e4d);
            color: #fff9ea;
            margin-bottom: 1.2rem;
          }
          .transit-hero h1 { color: inherit; padding: .25rem 0; }
          .transit-hero p { max-width: 56ch; margin: 0; }
          .transit-hero .eyebrow { color: #ffd499; font-weight: 700; letter-spacing: .09em; text-transform: uppercase; }
          div[data-testid="stForm"] { border-radius: 16px; }
          div[data-testid="stMetric"] { padding: .65rem .9rem; border: 1px solid #7f9a8c65; border-radius: 14px; }
        </style>
        <div class="transit-hero">
          <p class="eyebrow">Transit Insight</p>
          <h1>Plan a direct journey</h1>
          <p>Find a scheduled trip and explore the stops and commuter reviews along the way.</p>
        </div>
        """, unsafe_allow_html=True)

    try:
        dataset = CSV_PATH
        if not dataset.exists():
            dataset = CSV_PATH.with_name("route_planner_trips (1).csv")
        frame, route_keys = st.cache_data(show_spinner="Loading transit schedules…")(read_schedule)(
            str(dataset), dataset.stat().st_mtime_ns
        )
    except (OSError, ValueError) as error:
        st.error(f"Unable to load route_planner_trips.csv. {error}")
        st.stop()

    stops = sorted(frame["Stop_search"].unique())
    with st.form("journey_search"):
        st.subheader("Journey search")
        left, right = st.columns(2)
        origin = left.selectbox("Origin", stops, index=None, placeholder="Choose a station")
        destination = right.selectbox("Destination", stops, index=None, placeholder="Choose a station")
        day_label = left.selectbox("Travel day", list(DAY_OPTIONS))
        preferred = right.time_input("Leave at or after (optional)", value=None, step=60)
        st.caption("Leave the time blank to see the earliest trips. Times are Malaysia time. Weekend and holiday trips are grouped in the data. Check the operator for a specific date.")
        submitted = st.form_submit_button("Find direct trips", type="primary")

    if submitted:
        st.session_state["journey_query"] = origin, destination, day_label, preferred
    if "journey_query" not in st.session_state:
        st.info("Choose your origin and destination to see scheduled direct trips.")
        return

    origin, destination, day_label, preferred = st.session_state["journey_query"]
    if not origin or not destination:
        st.warning("Choose both an origin and a destination.")
        return
    if origin == destination:
        st.warning("Choose a different destination from your origin.")
        return

    earliest = preferred.hour * 3600 + preferred.minute * 60 if isinstance(preferred, dt.time) else None
    service_day = DAY_OPTIONS[day_label]
    with st.spinner("Finding matching trips…"):
        journeys = find_journeys(frame, route_keys, origin, destination, service_day, earliest)
    if not journeys:
        earlier = earliest is not None and find_journeys(
            frame, route_keys, origin, destination, service_day, None
        )
        if earlier:
            st.info("Direct trips run earlier in this schedule. Try an earlier time or clear the time field.")
        else:
            st.info("No scheduled direct trip was found for these stations on this service day. Try another service day or station pair.")
        return

    st.subheader("Scheduled departures")
    # Include the first departure from each route before offering additional trips.
    first = {}
    for journey in journeys:
        key = tuple(None if pd.isna(value) else value for value in journey["route_key"])
        first.setdefault(key, journey)
    featured = list(first.values())
    featured_ids = {id(journey) for journey in featured}
    featured.extend(journey for journey in journeys if id(journey) not in featured_ids)
    options = sorted(featured[:max(20, len(first))], key=lambda trip: trip["order"])
    st.caption(f"{origin} → {destination} · {day_label} · Showing {len(options)} of {len(journeys)} matching trips, including the first matching trip on each route.")
    labels = [
        f"{format_service_time(journey['depart'])} → {format_service_time(journey['arrive'])}"
        f"  |  {journey['route']}"
        + (" (departure estimated)" if journey["departure_estimated"] else "")
        for journey in options
    ]
    selected = st.selectbox(
        "Choose a trip", range(len(options)), format_func=labels.__getitem__,
        key=f"trip-choice:{origin}:{destination}:{day_label}:{preferred}",
    )
    show_journey(st, pdk, options[selected])


if __name__ == "__main__":
    main()
