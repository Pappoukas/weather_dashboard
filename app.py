"""
Weather Dashboard for Kastoria (2008-2026)
Interactive visualizations for temperature, precipitation, wind, and extremes.

Station: Kastoria (LGC0), elevation 623 m.
Initially located at the Makedni Town Hall; relocated within the city of
Kastoria on 2010-12-08 (same elevation, anemometer raised from 3 m to 5 m).
Data begin September 2008. Known data-quality issues (documented by the
station operator) are encoded in KNOWN_RAIN_ISSUES / KNOWN_WIND_ISSUES below
and are surfaced throughout the dashboard.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------
st.set_page_config(page_title="Kastoria Weather Dashboard", layout="wide")
st.title("🌤️ Kastoria Weather Dashboard (2008–2026)")

STATION_RELOCATION = pd.Timestamp("2010-12-08")

# Known periods with lost / partial rainfall data (from the station log).
# (start, end, approx mm lost or None if unknown, note)
KNOWN_RAIN_ISSUES = [
    ("2008-09-01", "2008-09-15", None, "All rainfall lost from start of operation"),
    ("2009-01-22", "2009-01-23", None, "Partial recording (technical problem)"),
    ("2009-08-05", "2009-08-07", None, "Rainfall data lost"),
    ("2009-08-24", "2009-08-26", None, "Rainfall data lost"),
    ("2009-10-24", "2009-12-19", None, "Rainfall data lost (extended outage)"),
    ("2010-03-07", "2010-03-08", None, "Partial snowfall recording"),
    ("2010-05-24", "2010-05-24", None, "Partial rainfall recording"),
    ("2010-09-11", "2010-09-11", None, "Partial rainfall recording"),
    ("2010-09-25", "2010-10-06", None, "Rainfall data lost"),
    ("2011-11-11", "2011-11-12", None, "Rainfall data lost"),
    ("2012-09-14", "2012-09-15", None, "Partial rainfall loss"),
    ("2012-10-14", "2012-10-14", None, "Partial rainfall loss"),
    ("2013-05-08", "2013-05-08", None, "Partial rainfall loss"),
    ("2014-04-05", "2014-04-06", None, "Partial / delayed rainfall recording"),
    ("2015-05-31", "2015-06-01", None, "Partial / delayed rainfall recording"),
    ("2016-09-01", "2016-09-01", 15, "Significant rainfall loss"),
    ("2016-10-10", "2016-10-12", 5, "Partial / delayed rainfall recording"),
    ("2018-05-01", "2018-05-08", 35, "Partial / delayed rainfall recording"),
    ("2018-06-13", "2018-06-15", 10, "Partial / delayed rainfall recording"),
    ("2018-06-26", "2018-06-28", 105, "Major loss (~60+20+25 mm)"),
    ("2018-11-16", "2018-11-17", 12, "Rainfall data lost"),
    ("2019-09-19", "2019-09-26", 20, "Rainfall data lost"),
    ("2019-10-04", "2019-10-04", 20, "Rainfall data lost"),
    ("2020-09-22", "2020-09-29", 45, "Rainfall data lost on several days"),
    ("2021-11-22", "2021-11-24", 8, "Partial / delayed rainfall recording"),
    ("2022-09-17", "2022-09-30", 15, "Rainfall data lost"),
    ("2022-10-01", "2022-10-14", 30, "Rainfall data lost"),
    ("2022-12-14", "2022-12-15", 5, "Partial / delayed rainfall recording"),
    ("2023-01-09", "2023-01-10", 15, "Partial / delayed rainfall recording"),
    ("2023-07-02", "2023-07-05", 10, "Rainfall data lost (2/7 and 5/7)"),
    ("2024-05-19", "2024-05-28", 5, "Data outage 18-31/05"),
    ("2025-04-07", "2025-04-07", None, "Delayed snowfall recording"),
]

# Known periods with lost or underestimated wind data.
KNOWN_WIND_ISSUES = [
    ("2008-10-15", "2008-10-15", "Wind data lost"),
    ("2011-09-17", "2011-09-28", "Wind data lost"),
    ("2012-05-26", "2012-05-26", "Wind data lost"),
    ("2013-09-01", "2013-09-17", "Wind data lost"),
    ("2018-03-01", "2018-06-01", "Wind speed underestimated"),
    ("2021-09-01", "2021-10-04", "Wind speed underestimated"),
]

RAIN_ISSUE_INTERVALS = [
    (pd.Timestamp(s), pd.Timestamp(e), mm, note)
    for s, e, mm, note in KNOWN_RAIN_ISSUES
]

WIND_DIRS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
             "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


# ------------------------------------------------------------
# Load data with caching
# ------------------------------------------------------------
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip().str.lstrip("\ufeff")
    for col in ["MeanTemp", "HighTemp", "LowTemp", "Rain_mm",
                "AvgWindSpeed_kmh", "MaxWindSpeed_kmh",
                "HeatDegDays", "CoolDegDays"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Date"] = pd.to_datetime(df[["Year", "Month", "Day"]])
    df["DominantWindDir"] = df["DominantWindDir"].replace("---", np.nan)
    df = df.sort_values("Date").reset_index(drop=True)
    return df


uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = load_data(uploaded_file)
else:
    try:
        df = load_data("kastoria_daily_all_years.csv")
        st.sidebar.success("Loaded default file: kastoria_daily_all_years.csv")
    except FileNotFoundError:
        st.error("Please upload the CSV file or place it in the same directory "
                 "as 'kastoria_daily_all_years.csv'.")
        st.stop()

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def find_streaks(dates, condition, min_length=1):
    """
    Given a boolean Series indexed by consecutive calendar dates, return a
    DataFrame of streaks (start, end, length) where the condition holds for
    at least `min_length` consecutive days. Missing days break streaks.
    """
    s = condition.reindex(pd.date_range(dates.min(), dates.max(), freq="D"),
                          fill_value=False)
    groups = (~s).cumsum()
    out = []
    for g, run in s[s].groupby(groups[s]):
        if len(run) >= min_length:
            out.append({"Start": run.index.min(), "End": run.index.max(),
                        "Days": len(run)})
    return pd.DataFrame(out)


def longest_dry_spell(data):
    """
    (length, start, end) of the longest run of consecutive CALENDAR days with
    Rain_mm == 0. Missing rain values (NaN) or dates absent from the dataset
    break the streak, so gaps cannot inflate the result.
    """
    s = data.set_index("Date")["Rain_mm"]
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"))
    is_dry = (s == 0)
    streaks = find_streaks(s.index.to_series(), is_dry)
    if streaks.empty:
        return 0, None, None
    best = streaks.loc[streaks["Days"].idxmax()]
    return int(best["Days"]), best["Start"], best["End"]


def rain_issues_overlapping(start, end):
    """Return known rain-data issues overlapping the interval [start, end]."""
    if start is None or end is None:
        return []
    return [(s, e, mm, note) for s, e, mm, note in RAIN_ISSUE_INTERVALS
            if s <= end and e >= start]


# ------------------------------------------------------------
# Sidebar filters
# ------------------------------------------------------------
st.sidebar.header("Filters")
min_date = df["Date"].min().date()
max_date = df["Date"].max().date()
date_range = st.sidebar.date_input(
    "Date range", value=(min_date, max_date),
    min_value=min_date, max_value=max_date,
)
if len(date_range) == 2:
    start_date, end_date = date_range
    mask = (df["Date"].dt.date >= start_date) & (df["Date"].dt.date <= end_date)
    df_filtered = df.loc[mask].copy()
else:
    df_filtered = df.copy()

vars_for_plot = st.sidebar.multiselect(
    "Variables to display in time series",
    options=["MeanTemp", "HighTemp", "LowTemp", "Rain_mm", "AvgWindSpeed_kmh"],
    default=["MeanTemp", "HighTemp", "LowTemp"],
)

# ------------------------------------------------------------
# Overview statistics
# ------------------------------------------------------------
st.header("📊 Overview Statistics")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Mean Temperature", f"{df_filtered['MeanTemp'].mean():.1f} °C")
with col2:
    st.metric("Total Rain", f"{df_filtered['Rain_mm'].sum():.1f} mm",
              help="Recorded total; known gaps mean the true total is higher "
                   "(see Data Quality section).")
with col3:
    st.metric("Max Wind Gust", f"{df_filtered['MaxWindSpeed_kmh'].max():.1f} km/h")
with col4:
    st.metric("Days with Rain > 1 mm", f"{(df_filtered['Rain_mm'] > 1).sum()}")

# ------------------------------------------------------------
# All-time records (within selected date range)
# ------------------------------------------------------------
st.header("🏅 Records")

hot_row = df_filtered.loc[df_filtered["HighTemp"].idxmax()]
cold_row = df_filtered.loc[df_filtered["LowTemp"].idxmin()]
rain_valid = df_filtered.dropna(subset=["Rain_mm"])
wet_row = rain_valid.loc[rain_valid["Rain_mm"].idxmax()] if not rain_valid.empty else None
dry_len, dry_start, dry_end = longest_dry_spell(df_filtered)

rec1, rec2, rec3, rec4 = st.columns(4)
with rec1:
    st.metric("🔥 Hottest day", f"{hot_row['HighTemp']:.1f} °C",
              help=f"Time of maximum: {hot_row.get('HighTime', '—')}")
    st.caption(hot_row["Date"].strftime("%d %b %Y"))
with rec2:
    st.metric("🥶 Coldest day", f"{cold_row['LowTemp']:.1f} °C",
              help=f"Time of minimum: {cold_row.get('LowTime', '—')}")
    st.caption(cold_row["Date"].strftime("%d %b %Y"))
with rec3:
    if wet_row is not None:
        st.metric("🌧️ Rainiest day", f"{wet_row['Rain_mm']:.1f} mm")
        st.caption(wet_row["Date"].strftime("%d %b %Y"))
    else:
        st.metric("🌧️ Rainiest day", "—")
with rec4:
    st.metric("☀️ Longest dry spell", f"{dry_len} days")
    if dry_start is not None:
        st.caption(f"{dry_start.strftime('%d %b %Y')} → {dry_end.strftime('%d %b %Y')}")

# Reliability checks against the station's known-issues log
dry_overlaps = rain_issues_overlapping(dry_start, dry_end)
if dry_overlaps:
    st.warning(
        "⚠️ The longest dry spell overlaps a period with known rainfall data "
        "losses, so it may be an artifact of missing data: "
        + "; ".join(f"{s.date()}–{e.date()} ({note})" for s, e, mm, note in dry_overlaps)
    )
st.caption(
    "The rainiest-day record reflects *recorded* rainfall. Note that on "
    "26/06/2018 approximately 60 mm went unrecorded due to a sensor fault, "
    "so the true daily record may differ. Dry spells are counted as "
    "consecutive calendar days with 0 mm; missing days or missing values "
    "break the streak (conservative lower bound)."
)

# ------------------------------------------------------------
# Time series
# ------------------------------------------------------------
st.header("📈 Time Series Evolution")
if vars_for_plot:
    fig = px.line(df_filtered, x="Date", y=vars_for_plot,
                  title="Daily Weather Variables",
                  labels={"value": "Value", "Date": ""},
                  color_discrete_sequence=px.colors.qualitative.Set1)
    st.plotly_chart(fig, width='stretch')
else:
    st.info("Select at least one variable in the sidebar.")

st.subheader("🌧️ Daily Precipitation")
fig_rain = px.bar(df_filtered, x="Date", y="Rain_mm",
                  title="Daily Rainfall (mm)",
                  labels={"Rain_mm": "Rain (mm)", "Date": ""},
                  color_discrete_sequence=["#1f77b4"])
st.plotly_chart(fig_rain, width='stretch')

# ------------------------------------------------------------
# Extremes (top-10 tables)
# ------------------------------------------------------------
st.header("🏆 Extremes")
tab1, tab2, tab3 = st.tabs(["Hottest Days", "Coldest Days", "Rainiest Days"])

with tab1:
    hottest = df_filtered.nlargest(10, "HighTemp")[["Date", "HighTemp", "MeanTemp", "LowTemp"]]
    st.dataframe(hottest.style.highlight_max(subset=["HighTemp"], color="salmon"))
    fig_hot = px.bar(hottest.sort_values("HighTemp"), x="HighTemp", y="Date",
                     orientation="h", title="Top 10 Hottest Days (by daily maximum)")
    st.plotly_chart(fig_hot, width='stretch')

with tab2:
    coldest = df_filtered.nsmallest(10, "LowTemp")[["Date", "LowTemp", "MeanTemp", "HighTemp"]]
    st.dataframe(coldest.style.highlight_min(subset=["LowTemp"], color="lightblue"))
    fig_cold = px.bar(coldest.sort_values("LowTemp", ascending=False),
                      x="LowTemp", y="Date", orientation="h",
                      title="Top 10 Coldest Days (by daily minimum)")
    st.plotly_chart(fig_cold, width='stretch')

with tab3:
    rainiest = df_filtered.nlargest(10, "Rain_mm")[["Date", "Rain_mm", "MeanTemp"]]
    st.dataframe(rainiest.style.highlight_max(subset=["Rain_mm"], color="lightgreen"))
    fig_rainiest = px.bar(rainiest.sort_values("Rain_mm"), x="Rain_mm", y="Date",
                          orientation="h", title="Top 10 Rainiest Days (recorded)")
    st.plotly_chart(fig_rainiest, width='stretch')

# ------------------------------------------------------------
# Climate indices & trend (complete years only)
# ------------------------------------------------------------
st.header("🌍 Climate Indices & Trend")

complete_years = [y for y, n in df.groupby("Year").size().items() if n >= 330]
df_cy = df[df["Year"].isin(complete_years)]
st.caption(
    f"Computed on (nearly) complete years only: {min(complete_years)}–{max(complete_years)}. "
    "Partial years (2008 starts in September; 2026 ends in February) are excluded "
    "so annual counts and trends are comparable."
)

indices = df_cy.groupby("Year").agg(
    HotDays=("HighTemp", lambda s: (s >= 35).sum()),
    SummerDays=("HighTemp", lambda s: (s >= 25).sum()),
    TropicalNights=("LowTemp", lambda s: (s >= 20).sum()),
    FrostDays=("LowTemp", lambda s: (s < 0).sum()),
    IceDays=("HighTemp", lambda s: (s < 0).sum()),
    MeanTemp=("MeanTemp", "mean"),
).reset_index()

ic1, ic2 = st.columns(2)
with ic1:
    fig_idx_warm = px.line(indices, x="Year", y=["HotDays", "TropicalNights"],
                           markers=True,
                           title="Warm indices per year (Hot days ≥35 °C, Tropical nights ≥20 °C)",
                           labels={"value": "Days", "Year": ""})
    st.plotly_chart(fig_idx_warm, width='stretch')
with ic2:
    fig_idx_cold = px.line(indices, x="Year", y=["FrostDays", "IceDays"],
                           markers=True,
                           title="Cold indices per year (Frost days Tmin<0 °C, Ice days Tmax<0 °C)",
                           labels={"value": "Days", "Year": ""})
    st.plotly_chart(fig_idx_cold, width='stretch')

# Annual mean temperature with linear trend
x = indices["Year"].to_numpy(dtype=float)
y = indices["MeanTemp"].to_numpy(dtype=float)
slope, intercept = np.polyfit(x, y, 1)
trend_y = slope * x + intercept

fig_trend = go.Figure()
fig_trend.add_trace(go.Scatter(x=indices["Year"], y=indices["MeanTemp"],
                               mode="lines+markers", name="Annual mean temp"))
fig_trend.add_trace(go.Scatter(x=indices["Year"], y=trend_y, mode="lines",
                               name=f"Trend: {slope*10:+.2f} °C / decade",
                               line=dict(dash="dash", color="firebrick")))
fig_trend.update_layout(title="Annual Mean Temperature & Linear Trend",
                        yaxis_title="°C")
st.plotly_chart(fig_trend, width='stretch')
st.caption(
    "⚠️ Homogeneity caveat: the station was relocated within Kastoria on "
    "08/12/2010 (same elevation). Trends spanning that date mix two siting "
    "environments and should be interpreted with caution."
)

# ------------------------------------------------------------
# Heatwaves & cold spells
# ------------------------------------------------------------
st.header("🌡️ Heatwaves & Cold Spells")
hw1, hw2 = st.columns(2)
with hw1:
    heat_thresh = st.slider("Heatwave threshold: daily max ≥ (°C)", 30, 42, 35)
    heat_days = st.slider("Minimum consecutive days (heatwave)", 2, 7, 3)
with hw2:
    cold_thresh = st.slider("Cold-spell threshold: daily min ≤ (°C)", -15, 5, -5)
    cold_days = st.slider("Minimum consecutive days (cold spell)", 2, 7, 3)

t = df_filtered.set_index("Date")
heatwaves = find_streaks(t.index.to_series(), t["HighTemp"] >= heat_thresh, heat_days)
coldspells = find_streaks(t.index.to_series(), t["LowTemp"] <= cold_thresh, cold_days)

hwc1, hwc2 = st.columns(2)
with hwc1:
    st.subheader(f"🔥 Heatwaves (≥{heat_thresh} °C for {heat_days}+ days)")
    if heatwaves.empty:
        st.info("No heatwaves found for these criteria.")
    else:
        st.dataframe(heatwaves.sort_values("Days", ascending=False),
                     width='stretch')
with hwc2:
    st.subheader(f"❄️ Cold spells (≤{cold_thresh} °C for {cold_days}+ days)")
    if coldspells.empty:
        st.info("No cold spells found for these criteria.")
    else:
        st.dataframe(coldspells.sort_values("Days", ascending=False),
                     width='stretch')

# ------------------------------------------------------------
# Wind rose
# ------------------------------------------------------------
st.header("🧭 Wind Rose")
wind_period = st.radio(
    "Period", ["All data", "Before relocation (≤ 07/12/2010, anemometer 3 m)",
               "After relocation (≥ 08/12/2010, anemometer 5 m)"],
    horizontal=True,
)
wdf = df_filtered.dropna(subset=["DominantWindDir"])
if wind_period.startswith("Before"):
    wdf = wdf[wdf["Date"] < STATION_RELOCATION]
elif wind_period.startswith("After"):
    wdf = wdf[wdf["Date"] >= STATION_RELOCATION]

if wdf.empty:
    st.info("No wind-direction data for the selected period.")
else:
    rose = (wdf.groupby("DominantWindDir")
            .agg(Days=("DominantWindDir", "size"),
                 AvgSpeed=("AvgWindSpeed_kmh", "mean"))
            .reindex(WIND_DIRS).fillna(0).reset_index())
    fig_rose = go.Figure(go.Barpolar(
        r=rose["Days"], theta=rose["DominantWindDir"],
        marker=dict(color=rose["AvgSpeed"], colorscale="Viridis",
                    colorbar=dict(title="Avg speed<br>(km/h)")),
    ))
    fig_rose.update_layout(
        title="Dominant Daily Wind Direction (bar length = days, color = avg speed)",
        polar=dict(angularaxis=dict(direction="clockwise", rotation=90)),
    )
    st.plotly_chart(fig_rose, width='stretch')
    st.caption(
        "The anemometer height changed from 3 m to 5 m at the relocation "
        "(08/12/2010), so wind speeds before/after are not directly comparable. "
        "Known wind-data gaps and underestimation periods are listed in the "
        "Data Quality section."
    )

# ------------------------------------------------------------
# Climatology: a year vs the long-term normal
# ------------------------------------------------------------
st.header("📆 Year vs Climatological Normal")
clim = (df_cy.groupby(["Month", "Day"])["MeanTemp"]
        .agg(Normal="mean", P10=lambda s: s.quantile(0.10),
             P90=lambda s: s.quantile(0.90))
        .reset_index())
clim["DOY"] = pd.to_datetime(
    dict(year=2001, month=clim["Month"], day=clim["Day"]), errors="coerce")
clim = clim.dropna(subset=["DOY"]).sort_values("DOY")

sel_year = st.selectbox("Select a year to compare with the normal",
                        sorted(df["Year"].unique(), reverse=True))
yr = df[df["Year"] == sel_year][["Month", "Day", "MeanTemp"]]
comp = clim.merge(yr, on=["Month", "Day"], how="left")

fig_clim = go.Figure()
fig_clim.add_trace(go.Scatter(x=comp["DOY"], y=comp["P90"], mode="lines",
                              line=dict(width=0), showlegend=False))
fig_clim.add_trace(go.Scatter(x=comp["DOY"], y=comp["P10"], mode="lines",
                              line=dict(width=0), fill="tonexty",
                              fillcolor="rgba(128,128,128,0.25)",
                              name="10th–90th percentile"))
fig_clim.add_trace(go.Scatter(x=comp["DOY"], y=comp["Normal"], mode="lines",
                              name=f"Normal ({min(complete_years)}–{max(complete_years)})",
                              line=dict(color="black", dash="dot")))
fig_clim.add_trace(go.Scatter(x=comp["DOY"], y=comp["MeanTemp"], mode="lines",
                              name=f"{sel_year}", line=dict(color="firebrick")))
fig_clim.update_layout(title=f"Daily Mean Temperature: {sel_year} vs Normal",
                       yaxis_title="°C",
                       xaxis=dict(tickformat="%b", dtick="M1"))
st.plotly_chart(fig_clim, width='stretch')

# ------------------------------------------------------------
# Aggregated views (dual-axis, so temperature and rain are readable)
# ------------------------------------------------------------
st.header("📅 Aggregated Views")

yearly = df_filtered.groupby("Year").agg(
    MeanTemp=("MeanTemp", "mean"), Rain=("Rain_mm", "sum")).reset_index()
fig_yearly = make_subplots(specs=[[{"secondary_y": True}]])
fig_yearly.add_trace(go.Bar(x=yearly["Year"], y=yearly["Rain"],
                            name="Total rain (mm)",
                            marker_color="steelblue", opacity=0.7),
                     secondary_y=False)
fig_yearly.add_trace(go.Scatter(x=yearly["Year"], y=yearly["MeanTemp"],
                                name="Mean temp (°C)", mode="lines+markers",
                                line=dict(color="firebrick")),
                     secondary_y=True)
fig_yearly.update_layout(title="Yearly Totals & Averages")
fig_yearly.update_yaxes(title_text="Rain (mm)", secondary_y=False)
fig_yearly.update_yaxes(title_text="Mean temp (°C)", secondary_y=True)
st.plotly_chart(fig_yearly, width='stretch')
st.caption("Rain totals for years with known data losses understate the true "
           "amount (e.g. 2009, 2018, 2022 — see Data Quality).")

monthly = df_cy.groupby("Month").agg(
    MeanTemp=("MeanTemp", "mean"),
    Rain=("Rain_mm", "sum")).reset_index()
monthly["Rain"] = monthly["Rain"] / len(complete_years)  # avg per month
fig_monthly = make_subplots(specs=[[{"secondary_y": True}]])
fig_monthly.add_trace(go.Bar(x=monthly["Month"], y=monthly["Rain"],
                             name="Avg monthly rain (mm)",
                             marker_color="steelblue", opacity=0.7),
                      secondary_y=False)
fig_monthly.add_trace(go.Scatter(x=monthly["Month"], y=monthly["MeanTemp"],
                                 name="Mean temp (°C)", mode="lines+markers",
                                 line=dict(color="firebrick")),
                      secondary_y=True)
fig_monthly.update_layout(
    title="Climogram: Average Monthly Rainfall & Temperature",
    xaxis=dict(tickmode="array", tickvals=list(range(1, 13)),
               ticktext=["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]))
fig_monthly.update_yaxes(title_text="Rain (mm)", secondary_y=False)
fig_monthly.update_yaxes(title_text="Mean temp (°C)", secondary_y=True)
st.plotly_chart(fig_monthly, width='stretch')

st.subheader("🌡️ Monthly Temperature Heatmap")
heatmap_data = df_filtered.pivot_table(index="Year", columns="Month",
                                       values="MeanTemp", aggfunc="mean")
fig_heat = px.imshow(heatmap_data,
                     labels=dict(x="Month", y="Year", color="Mean Temp (°C)"),
                     color_continuous_scale="RdBu_r",
                     title="Average Temperature by Year and Month")
st.plotly_chart(fig_heat, width='stretch')

# ------------------------------------------------------------
# Distributions
# ------------------------------------------------------------
st.header("📊 Distributions")
d1, d2 = st.columns(2)
with d1:
    fig_hist = px.histogram(df_filtered, x="MeanTemp", nbins=50,
                            title="Temperature Distribution")
    st.plotly_chart(fig_hist, width='stretch')
with d2:
    fig_box = px.box(df_filtered, x="Month", y="MeanTemp",
                     title="Temperature by Month (box plot)")
    st.plotly_chart(fig_box, width='stretch')

fig_rain_dist = px.histogram(df_filtered[df_filtered["Rain_mm"] > 0],
                             x="Rain_mm", nbins=50,
                             title="Rainfall Distribution (days with rain)")
st.plotly_chart(fig_rain_dist, width='stretch')

# ------------------------------------------------------------
# Data quality & station metadata
# ------------------------------------------------------------
st.header("🔍 Data Quality & Station Info")
with st.expander("Station metadata and known data issues", expanded=False):
    st.markdown(
        """
**Station: Kastoria (LGC0)** — elevation 623 m.
Originally located at the Makedni Town Hall (grass surface, temp/humidity
sensors at 2 m, anemometer at 3 m). **Relocated within the city of Kastoria
on 08/12/2010** at the same elevation; the anemometer was raised to 5 m.
Data begin in **September 2008**; no observations exist before that date.

**Implications for this dashboard**
- Rainfall totals are **lower bounds**: numerous documented sensor faults
  mean real rainfall went unrecorded (roughly 350+ mm across 2008–2025,
  with the largest single loss ~60 mm on 26/06/2018).
- Wind speeds before and after 08/12/2010 are **not directly comparable**
  (anemometer height 3 m → 5 m), and two periods
  (01/03–01/06/2018, 01/09–04/10/2021) are known to **underestimate** speed.
- Temperature series spans two station sitings; long-term trends should be
  read with this inhomogeneity in mind.
        """
    )

    full_range = pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")
    missing_dates = full_range.difference(df["Date"])
    qc1, qc2, qc3 = st.columns(3)
    qc1.metric("Calendar days missing from CSV", len(missing_dates))
    qc2.metric("Days with missing rain value", int(df["Rain_mm"].isna().sum()))
    qc3.metric("Days with unknown wind direction",
               int(df["DominantWindDir"].isna().sum()))

    st.markdown("**Known rainfall data issues (station log):**")
    issues_df = pd.DataFrame(
        [(s.date(), e.date(), f"~{mm} mm" if mm else "unknown", note)
         for s, e, mm, note in RAIN_ISSUE_INTERVALS],
        columns=["From", "To", "Est. loss", "Note"])
    st.dataframe(issues_df, width='stretch')

    st.markdown("**Known wind data issues (station log):**")
    wind_df = pd.DataFrame(
        [(pd.Timestamp(s).date(), pd.Timestamp(e).date(), note)
         for s, e, note in KNOWN_WIND_ISSUES],
        columns=["From", "To", "Note"])
    st.dataframe(wind_df, width='stretch')

    if len(missing_dates) > 0:
        st.markdown("**Dates entirely absent from the CSV:**")
        st.dataframe(pd.DataFrame({"Missing date": missing_dates.date}),
                     width='stretch', height=200)

# ------------------------------------------------------------
# Raw data table
# ------------------------------------------------------------
st.header("📋 Raw Data (filtered)")
if st.checkbox("Show raw data"):
    st.dataframe(df_filtered[["Date", "MeanTemp", "HighTemp", "LowTemp",
                              "Rain_mm", "AvgWindSpeed_kmh",
                              "MaxWindSpeed_kmh", "DominantWindDir"]])

# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------
st.caption(
    "Data source: Kastoria meteorological station LGC0 (Sep 2008 – Feb 2026), "
    "elevation 623 m. Rainfall and wind records include documented gaps; see "
    "the Data Quality section. Dashboard built with Streamlit and Plotly."
)
