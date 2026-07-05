"""
Weather Dashboard for Kastoria (2008–2026)
Interactive visualizations for temperature, precipitation, wind, and extremes.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------
st.set_page_config(page_title="Kastoria Weather Dashboard", layout="wide")
st.title("🌤️ Kastoria Weather Dashboard (2008–2026)")

# ------------------------------------------------------------
# Load data with caching
# ------------------------------------------------------------
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    # Convert rain columns (empty strings -> NaN)
    df['Rain_mm'] = pd.to_numeric(df['Rain_mm'], errors='coerce')
    # Create date column
    df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)
    return df

# File uploader (so user can provide their own CSV)
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = load_data(uploaded_file)
else:
    # If no file uploaded, try to load from current directory (for local testing)
    try:
        df = load_data("kastoria_daily_all_years.csv")
        st.sidebar.success("Loaded default file: kastoria_daily_all_years.csv")
    except FileNotFoundError:
        st.error("Please upload the CSV file or place it in the same directory as 'kastoria_daily_all_years.csv'.")
        st.stop()

# ------------------------------------------------------------
# Sidebar filters
# ------------------------------------------------------------
st.sidebar.header("Filters")
min_date = df['Date'].min().date()
max_date = df['Date'].max().date()
date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)
if len(date_range) == 2:
    start_date, end_date = date_range
    mask = (df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)
    df_filtered = df.loc[mask].copy()
else:
    df_filtered = df.copy()

# Additional filter: variable selection for time series
vars_for_plot = st.sidebar.multiselect(
    "Variables to display in time series",
    options=['MeanTemp', 'HighTemp', 'LowTemp', 'Rain_mm', 'AvgWindSpeed_kmh'],
    default=['MeanTemp', 'HighTemp', 'LowTemp']
)

# ------------------------------------------------------------
# Main dashboard
# ------------------------------------------------------------
st.header("📊 Overview Statistics")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Mean Temperature", f"{df_filtered['MeanTemp'].mean():.1f} °C")
with col2:
    st.metric("Total Rain", f"{df_filtered['Rain_mm'].sum():.1f} mm")
with col3:
    st.metric("Max Wind Gust", f"{df_filtered['MaxWindSpeed_kmh'].max():.1f} km/h")
with col4:
    st.metric("Days with Rain > 1 mm", f"{(df_filtered['Rain_mm'] > 1).sum()}")

# ------------------------------------------------------------
# Time series plots
# ------------------------------------------------------------
st.header("📈 Time Series Evolution")

if vars_for_plot:
    fig = px.line(
        df_filtered,
        x='Date',
        y=vars_for_plot,
        title="Daily Weather Variables",
        labels={'value': 'Value', 'Date': ''},
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Select at least one variable in the sidebar.")

# Precipitation chart (separate for clarity)
st.subheader("🌧️ Daily Precipitation")
fig_rain = px.bar(
    df_filtered,
    x='Date',
    y='Rain_mm',
    title="Daily Rainfall (mm)",
    labels={'Rain_mm': 'Rain (mm)', 'Date': ''},
    color_discrete_sequence=['blue']
)
st.plotly_chart(fig_rain, use_container_width=True)

# ------------------------------------------------------------
# All-time Records (within selected date range)
# ------------------------------------------------------------
st.header("🏅 Records")

def longest_dry_spell(data):
    """
    Return (length_in_days, start_date, end_date) of the longest run of
    consecutive CALENDAR days with Rain_mm == 0.
    Days with missing rain values (NaN) or days absent from the dataset
    break the streak, so gaps in the record can't inflate the result.
    """
    s = data.set_index('Date')['Rain_mm']
    # Reindex to the full calendar so missing dates appear as NaN
    full_index = pd.date_range(s.index.min(), s.index.max(), freq='D')
    s = s.reindex(full_index)
    is_dry = (s == 0)  # NaN -> False, so unknown days break the streak
    # Assign a group id that increments every time the streak breaks
    groups = (~is_dry).cumsum()
    dry_runs = is_dry.groupby(groups).sum()
    if dry_runs.max() == 0:
        return 0, None, None
    best_group = dry_runs.idxmax()
    run_days = s.index[(groups == best_group) & is_dry]
    return len(run_days), run_days.min(), run_days.max()

# Hottest day (by daily maximum temperature)
hot_row = df_filtered.loc[df_filtered['HighTemp'].idxmax()]
# Coldest day (by daily minimum temperature)
cold_row = df_filtered.loc[df_filtered['LowTemp'].idxmin()]
# Rainiest day
rain_valid = df_filtered.dropna(subset=['Rain_mm'])
wet_row = rain_valid.loc[rain_valid['Rain_mm'].idxmax()] if not rain_valid.empty else None
# Longest dry spell
dry_len, dry_start, dry_end = longest_dry_spell(df_filtered)

rec1, rec2, rec3, rec4 = st.columns(4)
with rec1:
    st.metric(
        "🔥 Hottest day",
        f"{hot_row['HighTemp']:.1f} °C",
        help=f"Time of maximum: {hot_row.get('HighTime', '—')}"
    )
    st.caption(hot_row['Date'].strftime('%d %b %Y'))
with rec2:
    st.metric(
        "🥶 Coldest day",
        f"{cold_row['LowTemp']:.1f} °C",
        help=f"Time of minimum: {cold_row.get('LowTime', '—')}"
    )
    st.caption(cold_row['Date'].strftime('%d %b %Y'))
with rec3:
    if wet_row is not None:
        st.metric("🌧️ Rainiest day", f"{wet_row['Rain_mm']:.1f} mm")
        st.caption(wet_row['Date'].strftime('%d %b %Y'))
    else:
        st.metric("🌧️ Rainiest day", "—")
with rec4:
    st.metric("☀️ Longest dry spell", f"{dry_len} days")
    if dry_start is not None:
        st.caption(f"{dry_start.strftime('%d %b %Y')} → {dry_end.strftime('%d %b %Y')}")

st.caption(
    "Note: the dry-spell calculation counts consecutive calendar days with 0 mm of rain. "
    "Days with missing rain data (70 days, 2008–2011) or dates absent from the dataset "
    "break the streak, so the record is a conservative lower bound."
)

# ------------------------------------------------------------
# Extremes
# ------------------------------------------------------------
st.header("🏆 Extremes")

tab1, tab2, tab3 = st.tabs(["Hottest Days", "Coldest Days", "Rainiest Days"])

with tab1:
    hottest = df_filtered.nlargest(10, 'MeanTemp')[['Date', 'MeanTemp', 'HighTemp', 'LowTemp']]
    st.dataframe(hottest.style.highlight_max(subset=['MeanTemp'], color='red'))
    fig_hot = px.bar(hottest, x='Date', y='MeanTemp', title="Top 10 Hottest Days (by Mean Temp)")
    st.plotly_chart(fig_hot, use_container_width=True)

with tab2:
    coldest = df_filtered.nsmallest(10, 'MeanTemp')[['Date', 'MeanTemp', 'HighTemp', 'LowTemp']]
    st.dataframe(coldest.style.highlight_min(subset=['MeanTemp'], color='blue'))
    fig_cold = px.bar(coldest, x='Date', y='MeanTemp', title="Top 10 Coldest Days (by Mean Temp)")
    st.plotly_chart(fig_cold, use_container_width=True)

with tab3:
    rainiest = df_filtered.nlargest(10, 'Rain_mm')[['Date', 'Rain_mm', 'MeanTemp']]
    st.dataframe(rainiest.style.highlight_max(subset=['Rain_mm'], color='green'))
    fig_rainiest = px.bar(rainiest, x='Date', y='Rain_mm', title="Top 10 Rainiest Days")
    st.plotly_chart(fig_rainiest, use_container_width=True)

# ------------------------------------------------------------
# Monthly and Yearly Aggregations
# ------------------------------------------------------------
st.header("📅 Aggregated Views")

# Yearly averages
yearly = df_filtered.groupby('Year').agg({
    'MeanTemp': 'mean',
    'Rain_mm': 'sum',
    'AvgWindSpeed_kmh': 'mean'
}).reset_index()

fig_yearly = px.line(
    yearly,
    x='Year',
    y=['MeanTemp', 'Rain_mm', 'AvgWindSpeed_kmh'],
    title="Yearly Averages / Totals",
    labels={'value': 'Value', 'Year': ''}
)
st.plotly_chart(fig_yearly, use_container_width=True)

# Monthly averages (across all years)
monthly = df_filtered.groupby('Month').agg({
    'MeanTemp': 'mean',
    'Rain_mm': 'sum',
    'AvgWindSpeed_kmh': 'mean'
}).reset_index()

fig_monthly = px.bar(
    monthly,
    x='Month',
    y=['MeanTemp', 'Rain_mm', 'AvgWindSpeed_kmh'],
    title="Monthly Averages (all years combined)",
    labels={'value': 'Value', 'Month': ''},
    barmode='group'
)
st.plotly_chart(fig_monthly, use_container_width=True)

# Heatmap: average temperature by month and year
st.subheader("🌡️ Monthly Temperature Heatmap")
# Pivot table: years vs months
heatmap_data = df_filtered.pivot_table(
    index='Year',
    columns='Month',
    values='MeanTemp',
    aggfunc='mean'
)
fig_heat = px.imshow(
    heatmap_data,
    labels=dict(x="Month", y="Year", color="Mean Temp (°C)"),
    color_continuous_scale="RdBu_r",
    title="Average Temperature by Year and Month"
)
st.plotly_chart(fig_heat, use_container_width=True)

# ------------------------------------------------------------
# Distributions
# ------------------------------------------------------------
st.header("📊 Distributions")

col1, col2 = st.columns(2)
with col1:
    fig_hist = px.histogram(df_filtered, x='MeanTemp', nbins=50, title="Temperature Distribution")
    st.plotly_chart(fig_hist, use_container_width=True)
with col2:
    fig_box = px.box(df_filtered, y='MeanTemp', title="Temperature Box Plot")
    st.plotly_chart(fig_box, use_container_width=True)

# Rain distribution
fig_rain_dist = px.histogram(df_filtered[df_filtered['Rain_mm'] > 0], x='Rain_mm', nbins=50, title="Rainfall Distribution (days with rain)")
st.plotly_chart(fig_rain_dist, use_container_width=True)

# ------------------------------------------------------------
# Raw Data Table (optional)
# ------------------------------------------------------------
st.header("📋 Raw Data (filtered)")
if st.checkbox("Show raw data"):
    st.dataframe(df_filtered[['Date', 'MeanTemp', 'HighTemp', 'LowTemp', 'Rain_mm', 'AvgWindSpeed_kmh', 'MaxWindSpeed_kmh']])

# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------
st.caption("Data source: Kastoria meteorological station (2008–2026). Dashboard built with Streamlit and Plotly.")
