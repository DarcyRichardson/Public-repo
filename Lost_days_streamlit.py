"""
Lost Days Predictor 🚧

Interactive Streamlit app that estimates *lost construction days* for
crane operations and general weather downtime based on historical Bureau of
Meteorology (via the Open‑Meteo archive API) data, simple machine‑learning
models and Monte‑Carlo simulation.

```bash
# 1 – create the env the first time only
conda create -n weather-app python=3.11

# 2 – activate it whenever you work on the project
conda activate weather-app

# 3 – install packages (one-off inside this env)
pip install streamlit pandas numpy requests pydeck geopy scikit-learn

# 4 – move to your project folder
cd "C:/Users/DarcyRichardson/cerecon.com.au/Cerecon Business - Documents/14. Business intelligence/Lost days weather app"

# 5 – run the Streamlit server

streamlit run Lost_days_streamlit.py
```

Author: Adapted from your Jupyter notebook “Lost_days_01”.
"""


# app.py  ──────────────────────────────────────────────────────────────
from __future__ import annotations
import streamlit as st
import pandas as pd, numpy as np, requests, calendar, plotly.express as px, plotly.graph_objects as go
from datetime import date, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

# ─────────────────────────  CONFIG  ──────────────────────────────────
YEARS_BACK            = 10
OPENMETEO_TIMEOUT     = 60          # s
WIND_SPEED_THRESHOLD  = 36.0        # km/h  limit for wind stop
LIGHT_RAIN_THRESHOLD_MM = 2.5      # mm    limit for light rain stop
BUSINESS_RULES_MD     = """
### Business-rule assumptions

* **Shift window:** **{start_hr}:00 – {end_hr}:00** &nbsp; (**{shift_len} h**)

* **Heat stop**  
  When the temperature reaches **35 °C** or higher, **all remaining hours** in the shift are lost.

* **Rain stop**  
  * **Light rain:** 0 < rain ≤ {light_thresh} mm → hour counts as **0.5** lost  
  * **Heavy rain:** rain > {light_thresh} mm     → hour counts as **1** lost  
  * If (0.5 + 1) hours reach **≥ 50 % of the shift** → **entire shift lost**;  
    otherwise only the flagged hours are lost  
  * *Note:* this rule feeds into the **weather–downtime** calculation;  
    the rain-stop flag on its own represents just the estimated **rain-only** downtime.

* **Wind stop**  
  An hour is lost to crane operations if the *effective* wind speed exceeds **{wind_threshold} km/h**.  
  Effective speed is determined by **“{wind_method}”**:
  * `gust` – use the raw gust  
  * `reduced_gust` – gust × (1 − *reduction*)  
  * `blend` – wind + (gust − wind) × *blend-weight*

* **Combined weather downtime**  
  We take **max(heat stop, rain stop)** for each day—so no double-counting.
"""

# ── Helpers: API + flags ────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def om_archive_hourly(lat, lon, start, end) -> pd.DataFrame:
    url = (
        "https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat:.4f}&longitude={lon:.4f}"
        f"&start_date={start}&end_date={end}"
        "&hourly=precipitation,temperature_2m,windspeed_10m,windgusts_10m"
        "&timezone=Australia%2FMelbourne"
    )
    js = requests.get(url, timeout=OPENMETEO_TIMEOUT).json()
    if "hourly" not in js:
        raise RuntimeError(f"API error: {js}")
    return pd.DataFrame({
        "dt":            pd.to_datetime(js["hourly"]["time"]),
        "rain_mm":       js["hourly"]["precipitation"],
        "temp_c":        js["hourly"]["temperature_2m"],
        "wind_kmh":      js["hourly"]["windspeed_10m"],
        "windgusts_kmh": js["hourly"]["windgusts_10m"],
    }).set_index("dt")

def heat_downtime_flag(df: pd.DataFrame, shift_start_hr: int) -> pd.Series:
    df = df.copy()
    df["shift_day"] = (df.index - pd.Timedelta(hours=shift_start_hr)).date
    first_hot = df[df["temp_c"] >= 35].groupby("shift_day").apply(lambda x: x.index.min())
    df["heat_start"] = df["shift_day"].map(first_hot)
    return (df.index >= df["heat_start"]).astype(int)

def rain_downtime_flag(day_df: pd.DataFrame) -> pd.Series:
    """
    Rain downtime flag (0 / 0.5 / 1).

    0      – completely dry
    0.5    – light rain  (0 < rain_mm ≤ 2.5 mm)
    1      – heavier rain (> 2.5 mm)
    """
    df = day_df.copy()

    if df.empty:
        # keep dtype=float to match the 0.5 rule
        return pd.Series(0.0, index=day_df.index, dtype=float)

    # Vectorised classification
    conds   = [
        df["rain_mm"] == 0,
        (df["rain_mm"] > 0) & (df["rain_mm"] <= LIGHT_RAIN_THRESHOLD_MM),
        df["rain_mm"] > LIGHT_RAIN_THRESHOLD_MM,
    ]
    choices = [0.0, 0.5, 1.0]

    df["rain_stop"] = np.select(conds, choices, default=0.0).astype(float)
    return df["rain_stop"]

def wind_downtime_flag(
    df: pd.DataFrame,
    *,
    method="gust",
    gust_reduction=0.20,
    blend_weight=0.50,
    threshold=WIND_SPEED_THRESHOLD,
) -> pd.Series:
    if method == "gust":
        eff = df["windgusts_kmh"]
    elif method == "reduced_gust":
        eff = df["windgusts_kmh"] * (1 - gust_reduction)
    elif method == "blend":
        eff = df["wind_kmh"] + (df["windgusts_kmh"] - df["wind_kmh"]) * blend_weight
    else:
        raise ValueError
    return (eff > threshold).astype(int)

def calculate_downtime(hourly, shift_start_hr, shift_end_hr,
                       wind_method, gust_reduction, blend_weight):
    df = hourly.copy()
    df["heat_stop"] = heat_downtime_flag(df, shift_start_hr)
    df["rain_stop"] = rain_downtime_flag(df)
    df["wind_stop"] = wind_downtime_flag(
        df, method=wind_method,
        gust_reduction=gust_reduction,
        blend_weight=blend_weight
    )

    # slice paid hours
    if shift_end_hr > shift_start_hr:
        shift_hours = shift_end_hr - shift_start_hr
        df_shift = df[(df.index.hour >= shift_start_hr) &
                      (df.index.hour < shift_end_hr)]
    else:
        shift_hours = (24 - shift_start_hr) + shift_end_hr
        df_shift = df[(df.index.hour >= shift_start_hr) |
                      (df.index.hour < shift_end_hr)]

    # shift-day label (no clock shift)
    df_shift["shift_day"] = np.where(
        df_shift.index.hour >= shift_start_hr,
        df_shift.index.normalize(),
        (df_shift.index - pd.Timedelta(days=1)).normalize()
    )

    daily = (
        df_shift
          .groupby("shift_day")
          .agg(
              total_rain_mm=('rain_mm', 'sum'),
              avg_temp_c=('temp_c', 'mean'),
              avg_windgust_kmh=('windgusts_kmh', 'mean'),
              heat_stop_hours=('heat_stop', 'sum'),
              rain_stop_hours=('rain_stop', 'sum'),
              wind_stop_hours=('wind_stop', 'sum'),
          )
          .reset_index()
          .rename(columns={"shift_day": "dt"})
    )
    daily['dt'] = pd.to_datetime(daily['dt'])

    # business rules
    daily["shift_lost_rain"] = np.where(
        daily["rain_stop_hours"] >= shift_hours / 2,
        shift_hours,
        daily["rain_stop_hours"])
    daily["weather_stop_hours"] = daily[["heat_stop_hours",
                                         "shift_lost_rain"]].max(axis=1)
    daily["crane_stop_hours"] = daily["wind_stop_hours"]
    return daily, shift_hours, df_shift

def add_feats(df):
    df = df.copy()
    df["month"] = df["dt"].dt.month
    df["doy"]   = df["dt"].dt.dayofyear
    return df

@st.cache_resource(show_spinner=False)
def train_models(lat, lon, shift_start_hr, shift_end_hr,
                 wind_method, gust_reduction, blend_weight):
    end = date.today()
    start = end - timedelta(days=YEARS_BACK*365)
    hourly = om_archive_hourly(lat, lon, start, end)

    daily, shift_hours, hourly_flags = calculate_downtime(
        hourly, shift_start_hr, shift_end_hr,
        wind_method, gust_reduction, blend_weight)

    X = add_feats(daily)[["month","doy"]]
    y_crane   = daily["crane_stop_hours"]   / shift_hours
    y_weather = daily["weather_stop_hours"] / shift_hours

    # train / calibrate
    def fit_pair(X, y):
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        rf = RandomForestRegressor(n_estimators=400, max_depth=10,
                                   min_samples_leaf=5, random_state=42)
        rf.fit(X_tr, y_tr)
        iso = IsotonicRegression(out_of_bounds="clip").fit(rf.predict(X_val), y_val)
        return rf, iso

    rf_crane, iso_crane   = fit_pair(X, y_crane)
    rf_weather, iso_weather = fit_pair(X, y_weather)

    return rf_crane, iso_crane, rf_weather, iso_weather, daily, shift_hours, hourly, hourly_flags

def simulate(ratios, shift_hours, n=10_000):
    hours = np.random.binomial(shift_hours, ratios, size=(n, len(ratios)))
    days  = hours.sum(axis=1) / shift_hours
    qs    = np.quantile(days, [0.25,0.5,0.7,0.8,0.9])
    return qs, days

def summary_df(qs, label, days):
    qs = pd.Series(qs, index=["P25","P50","P70","P80","P90"])
    pct = qs / days
    out = pd.DataFrame({
        "Lost days": qs.astype(int),
        "% of period": pct.mul(100).round(1).map("{:.1f} %".format)
    })
    out.index.name = label
    return out

def create_seasonal_summary(daily):
    df = daily.copy()
    df["Month"] = df["dt"].dt.month
    out = df.groupby("Month").agg(
        rain=("total_rain_mm","mean"),
        temp=("avg_temp_c","mean")
    ).reset_index()
    out["Month"] = out["Month"].apply(lambda m: calendar.month_abbr[m])
    return out

# ── Streamlit UI ────────────────────────────────────────────────────
st.title("🚧 Lost Construction Days Forecaster")

with st.sidebar:
    st.header("Site")
    lat = st.number_input("Latitude",  value=-37.8136, format="%.4f")
    lon = st.number_input("Longitude", value=144.9631, format="%.4f")
    start_date = st.date_input("Start date", date.today())
    end_date   = st.date_input("End date",   date.today() + timedelta(days=180))

    st.header("Shift")
    shift_start_hr = st.number_input(
        "Shift start (0–23)",
        min_value=0,
        max_value=23,
        value=8,
        step=1
    )

    shift_end_hr = st.number_input(
        "Shift end (0–23)",
        min_value=0,
        max_value=23,
        value=16,
        step=1
    )

    iterations = st.number_input(
        "Monte-Carlo iterations",
        min_value=1_000,
        max_value=100_000,
        value=10_000,
        step=1_000,
    )

    st.header("Wind-stop method")
    wind_method = st.selectbox("Method", ["gust", "reduced_gust", "blend"])
    if wind_method == "reduced_gust":
        gust_reduction = st.slider("Reduction %", 0.0, 0.9, 0.30, 0.05)
        blend_weight = 0.5
    elif wind_method == "blend":
        blend_weight = st.slider("Blend weight (0 = wind, 1 = gust)", 0.0, 1.0, 0.5, 0.05)
        gust_reduction = 0.2
    else:
        gust_reduction = 0.2
        blend_weight = 0.5

    #iterations = st.number_input("Monte-Carlo iterations", 1_000, 100, 100_000, 1_000, value=10_000)
    run = st.button("Run forecast")

if not run:
    st.info("Set parameters → **Run forecast**")
    st.stop()

# ── Train / simulate ────────────────────────────────────────────────
rf_c, iso_c, rf_w, iso_w, daily_summary, shift_hours, hourly_raw, hourly_flags = \
    train_models(lat, lon, shift_start_hr, shift_end_hr,
                 wind_method, gust_reduction, blend_weight)

rng = pd.date_range(start_date, end_date)
future = add_feats(pd.DataFrame({"dt": rng}))

rat_c   = iso_c.predict(rf_c.predict(future[["month","doy"]]))
rat_w   = iso_w.predict(rf_w.predict(future[["month","doy"]]))

qs_c, dist_c = simulate(rat_c, shift_hours, iterations)
qs_w, dist_w = simulate(rat_w, shift_hours, iterations)
horizon_days = (end_date - start_date).days + 1

# ── Tabs UI ─────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Risk summary", "Hourly plots", "Monthly & seasonal", "Business rule assumptions"])

with tab1:
    st.subheader("Site location")
    site_df = pd.DataFrame({"lat": [lat], "lon": [lon]})
    st.map(site_df, zoom=12)          # adjust zoom to taste

# ── Monthly downtime table ─────────────────────────────────────────
    # 1️⃣  Make sure dt is datetime64
    if not pd.api.types.is_datetime64_any_dtype(daily_summary["dt"]):
        daily_summary["dt"] = pd.to_datetime(daily_summary["dt"])

    # 2️⃣  Row-level percentages (adds cols but does NOT mutate original ds)
    ds = daily_summary.copy()
    ds["pct_rain_downtime"]    = (ds["rain_stop_hours"]    / shift_hours * 100).round(1)
    ds["pct_heat_downtime"]    = (ds["heat_stop_hours"]    / shift_hours * 100).round(1)
    ds["pct_weather_downtime"] = (ds["weather_stop_hours"] / shift_hours * 100).round(1)
    ds["pct_crane_downtime"]   = (ds["crane_stop_hours"]   / shift_hours * 100).round(1)
    ds["pct_any_downtime"]     = (
        (ds["weather_stop_hours"] + ds["crane_stop_hours"])
        .clip(upper=shift_hours) / shift_hours * 100
    ).round(1)

    # 3️⃣  Month-level aggregation
    monthly = (
        ds.groupby(ds["dt"].dt.month)
        .agg(
            total_rain_stop_hours    = ("rain_stop_hours",    "sum"),
            total_heat_stop_hours    = ("heat_stop_hours",    "sum"),
            total_weather_stop_hours = ("weather_stop_hours", "sum"),
            total_crane_stop_hours   = ("crane_stop_hours",   "sum"),
            shift_days               = ("dt",                 "count"),
        )
    )

    monthly["total_hours"] = monthly["shift_days"] * shift_hours
    monthly["pct_rain"]    = (monthly["total_rain_stop_hours"]    / monthly["total_hours"] * 100).round(1)
    monthly["pct_heat"]    = (monthly["total_heat_stop_hours"]    / monthly["total_hours"] * 100).round(1)
    monthly["pct_weather"] = (monthly["total_weather_stop_hours"] / monthly["total_hours"] * 100).round(1)
    monthly["pct_crane"]   = (monthly["total_crane_stop_hours"]   / monthly["total_hours"] * 100).round(1)

    # 4️⃣  Pretty labels & order
    monthly.index      = monthly.index.map(lambda m: calendar.month_abbr[m])
    monthly.index.name = "Month"
    monthly = monthly[
        ["total_hours",
        "total_rain_stop_hours", "total_heat_stop_hours",
        "total_weather_stop_hours", "total_crane_stop_hours",
        "pct_rain", "pct_heat", "pct_weather", "pct_crane"]
    ]

    # 5️⃣  Show in the tab
    st.subheader("Monthly downtime summary - From past 10 years of data")
    st.dataframe(
        monthly.style.format({
            "total_hours": "{:,.0f}",
            "pct_rain": "{:.1f} %",
            "pct_heat": "{:.1f} %",
            "pct_weather": "{:.1f} %",
            "pct_crane": "{:.1f} %",
        }),
        use_container_width=True
    )

    st.subheader(f"Crane downtime risk - ML/Monte Carlo model")
    st.table(summary_df(qs_c, "Crane", horizon_days))
    st.subheader("Weather downtime risk - ML/Monte Carlo model")
    st.table(summary_df(qs_w, "Weather", horizon_days))

    # histogram
    fig = go.Figure()
    fig.add_histogram(x=dist_c, name="Crane")
    fig.add_histogram(x=dist_w, name="Weather")
    fig.update_layout(
        barmode="overlay",
        title=f"Lost working days over {horizon_days}-day horizon",
        xaxis_title="Lost days",
        yaxis_title="Frequency",
        template="plotly_white",
    )
    fig.update_traces(opacity=0.6)
    st.plotly_chart(fig, use_container_width=True)


with tab2:
    # hourly rain probability
    rain_prob = (
        hourly_raw.groupby(hourly_raw.index.hour)["rain_mm"]
        .apply(lambda x: (x > 0).mean())
        .reset_index()
    )
    rain_prob.columns = ["Hour of Day", "Rain Probability"]
    fig = px.line(rain_prob, x="Hour of Day", y="Rain Probability",
                  title="Average Hourly Rain Probability",
                  template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # ── Hourly wind / gust / blended (Plotly) ───────────────────────
    hourly_wind = (
        hourly_raw
        .assign(
            blended=lambda d: d["wind_kmh"] +
                    (d["windgusts_kmh"] - d["wind_kmh"]) * blend_weight
        )
        .groupby(hourly_raw.index.hour)
        .agg(
            avg_wind_kmh = ("wind_kmh",       "mean"),
            avg_gust_kmh = ("windgusts_kmh",  "mean"),
            avg_blend    = ("blended",        "mean"),
        )
        .reset_index()        # now two columns: 0, …
    )

    # give stable names so Plotly can find them
    hourly_wind.columns = ["Hour of Day",
                        "avg_wind_kmh",
                        "avg_gust_kmh",
                        "avg_blend"]

    fig = px.line(
        hourly_wind,
        x="Hour of Day",
        y=["avg_wind_kmh", "avg_gust_kmh", "avg_blend"],
        title="Average Hourly Wind / Gust / Blended",
        labels={
            "value": "Speed (km/h)",
            "variable": "Metric",
            "avg_wind_kmh": "Wind",
            "avg_gust_kmh": "Gust",
            "avg_blend": f"Blended ({blend_weight:.2f} gust)",
        },
        template="plotly_white",
    )
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    # downtime flags vs hour
    tmp = hourly_flags.copy()
    tmp["Month"] = tmp.index.month
    tmp["Hour"]  = tmp.index.hour
    by_hour = (
        tmp.groupby("Hour")[["heat_stop","rain_stop","wind_stop"]]
           .mean()
           .reset_index()
    )
    fig = px.line(by_hour, x="Hour",
                  y=["heat_stop","rain_stop","wind_stop"],
                  labels={"value":"Average Flag (0–1)"},
                  title="Average Downtime Flags by Hour of Day",
                  template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # rain downtime vs month
    by_month = (tmp.groupby("Month")["rain_stop"].mean().reset_index())
    by_month["Month"] = by_month["Month"].apply(lambda m: calendar.month_abbr[m])
    fig = px.bar(by_month, x="Month", y="rain_stop",
                 labels={"rain_stop":"Average Flag (0–1)"},
                 title="Average Rain Downtime by Month",
                 template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # seasonal rainfall & temp
    seasonal = create_seasonal_summary(daily_summary)
    fig = go.Figure()
    fig.add_bar(x=seasonal["Month"], y=seasonal["rain"], name="Rain (mm)")
    fig.add_scatter(x=seasonal["Month"], y=seasonal["temp"], name="Temp (°C)",
                    mode="lines+markers", yaxis="y2")
    fig.update_layout(
        title="Seasonal Rainfall & Temperature",
        yaxis=dict(title="Rain (mm)"),
        yaxis2=dict(title="Temp (°C)", overlaying="y", side="right"),
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)


with tab4:
    st.markdown(
        BUSINESS_RULES_MD.format(
            start_hr       = shift_start_hr,
            end_hr         = shift_end_hr,
            shift_len      = shift_hours,
            light_thresh   = LIGHT_RAIN_THRESHOLD_MM,   # ← add this line
            wind_threshold = WIND_SPEED_THRESHOLD,
            wind_method    = wind_method,
        )
    )
st.success("Forecast complete!")