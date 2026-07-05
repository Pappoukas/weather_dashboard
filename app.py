"""
Μετεωρολογικός πίνακας Καστοριάς (2008-2026)
Διαδραστικές οπτικοποιήσεις θερμοκρασίας, βροχόπτωσης, ανέμου και ακραίων τιμών.

Σταθμός: Καστοριά (LGC0), υψόμετρο 623 m.
Αρχικά στο Δημαρχείο Μακεδνών· μετεγκαταστάθηκε εντός της πόλης της Καστοριάς
στις 08/12/2010 (ίδιο υψόμετρο, ανεμόμετρο από 3 m σε 5 m).
Τα δεδομένα ξεκινούν τον Σεπτέμβριο του 2008. Τα γνωστά προβλήματα ποιότητας
(από το ημερολόγιο βλαβών του σταθμού) είναι κωδικοποιημένα στα
KNOWN_RAIN_ISSUES / KNOWN_WIND_ISSUES και εμφανίζονται σε όλο το dashboard.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ------------------------------------------------------------
# Ρυθμίσεις σελίδας
# ------------------------------------------------------------
st.set_page_config(page_title="Μετεωρολογικός Πίνακας Καστοριάς", layout="wide")
st.title("🌤️ Μετεωρολογικός Πίνακας Καστοριάς (2008–2026)")

STATION_RELOCATION = pd.Timestamp("2010-12-08")

GR_MONTHS = ["", "Ιαν", "Φεβ", "Μάρ", "Απρ", "Μάι", "Ιούν",
             "Ιούλ", "Αύγ", "Σεπ", "Οκτ", "Νοέ", "Δεκ"]
GR_MONTHS_FULL = ["", "Ιανουάριος", "Φεβρουάριος", "Μάρτιος", "Απρίλιος",
                  "Μάιος", "Ιούνιος", "Ιούλιος", "Αύγουστος", "Σεπτέμβριος",
                  "Οκτώβριος", "Νοέμβριος", "Δεκέμβριος"]

VAR_LABELS = {
    "MeanTemp": "Μέση θερμοκρασία (°C)",
    "HighTemp": "Μέγιστη θερμοκρασία (°C)",
    "LowTemp": "Ελάχιστη θερμοκρασία (°C)",
    "Rain_mm": "Βροχόπτωση (mm)",
    "AvgWindSpeed_kmh": "Μέση ταχύτητα ανέμου (km/h)",
}


def fmt_date(d, with_year=True):
    """Μορφοποίηση ημερομηνίας στα ελληνικά, π.χ. '22 Ιούλ 2025'."""
    if pd.isna(d):
        return "—"
    d = pd.Timestamp(d)
    return (f"{d.day} {GR_MONTHS[d.month]} {d.year}" if with_year
            else f"{d.day} {GR_MONTHS[d.month]}")


# Γνωστές περίοδοι με απώλειες / μερικές καταγραφές βροχόπτωσης (ημερολόγιο σταθμού).
# (έναρξη, λήξη, εκτιμώμενα χαμένα mm ή None, σημείωση)
KNOWN_RAIN_ISSUES = [
    ("2008-09-01", "2008-09-15", None, "Απώλεια όλης της βροχόπτωσης από την έναρξη λειτουργίας"),
    ("2009-01-22", "2009-01-23", None, "Μερική καταγραφή (τεχνικό πρόβλημα)"),
    ("2009-08-05", "2009-08-07", None, "Απώλεια δεδομένων βροχόπτωσης"),
    ("2009-08-24", "2009-08-26", None, "Απώλεια δεδομένων βροχόπτωσης"),
    ("2009-10-24", "2009-12-19", None, "Απώλεια δεδομένων βροχόπτωσης (παρατεταμένη βλάβη)"),
    ("2010-03-07", "2010-03-08", None, "Μερική καταγραφή χιονόπτωσης"),
    ("2010-05-24", "2010-05-24", None, "Μερική καταγραφή βροχόπτωσης"),
    ("2010-09-11", "2010-09-11", None, "Μερική καταγραφή βροχόπτωσης"),
    ("2010-09-25", "2010-10-06", None, "Απώλεια δεδομένων βροχόπτωσης"),
    ("2011-11-11", "2011-11-12", None, "Απώλεια δεδομένων βροχόπτωσης"),
    ("2012-09-14", "2012-09-15", None, "Μερική απώλεια βροχόπτωσης"),
    ("2012-10-14", "2012-10-14", None, "Μερική απώλεια βροχόπτωσης"),
    ("2013-05-08", "2013-05-08", None, "Μερική απώλεια βροχόπτωσης"),
    ("2014-04-05", "2014-04-06", None, "Μερική / καθυστερημένη καταγραφή"),
    ("2015-05-31", "2015-06-01", None, "Μερική / καθυστερημένη καταγραφή"),
    ("2016-09-01", "2016-09-01", 15, "Σημαντική απώλεια βροχόπτωσης"),
    ("2016-10-10", "2016-10-12", 5, "Μερική / καθυστερημένη καταγραφή"),
    ("2018-05-01", "2018-05-08", 35, "Μερική / καθυστερημένη καταγραφή"),
    ("2018-06-13", "2018-06-15", 10, "Μερική / καθυστερημένη καταγραφή"),
    ("2018-06-26", "2018-06-28", 105, "Μεγάλη απώλεια (~60+20+25 mm)"),
    ("2018-11-16", "2018-11-17", 12, "Απώλεια δεδομένων βροχόπτωσης"),
    ("2019-09-19", "2019-09-26", 20, "Απώλεια δεδομένων βροχόπτωσης"),
    ("2019-10-04", "2019-10-04", 20, "Απώλεια δεδομένων βροχόπτωσης"),
    ("2020-09-22", "2020-09-29", 45, "Απώλεια βροχόπτωσης σε πολλές ημέρες"),
    ("2021-11-22", "2021-11-24", 8, "Μερική / καθυστερημένη καταγραφή"),
    ("2022-09-17", "2022-09-30", 15, "Απώλεια δεδομένων βροχόπτωσης"),
    ("2022-10-01", "2022-10-14", 30, "Απώλεια δεδομένων βροχόπτωσης"),
    ("2022-12-14", "2022-12-15", 5, "Μερική / καθυστερημένη καταγραφή"),
    ("2023-01-09", "2023-01-10", 15, "Μερική / καθυστερημένη καταγραφή"),
    ("2023-07-02", "2023-07-05", 10, "Απώλεια βροχόπτωσης (2/7 και 5/7)"),
    ("2024-05-19", "2024-05-28", 5, "Διακοπή δεδομένων 18–31/05"),
    ("2025-04-07", "2025-04-07", None, "Καθυστερημένη καταγραφή χιονόπτωσης"),
]

# Γνωστές περίοδοι με απώλειες ή υποεκτίμηση δεδομένων ανέμου.
KNOWN_WIND_ISSUES = [
    ("2008-10-15", "2008-10-15", "Απώλεια δεδομένων ανέμου"),
    ("2011-09-17", "2011-09-28", "Απώλεια δεδομένων ανέμου"),
    ("2012-05-26", "2012-05-26", "Απώλεια δεδομένων ανέμου"),
    ("2013-09-01", "2013-09-17", "Απώλεια δεδομένων ανέμου"),
    ("2018-03-01", "2018-06-01", "Υποεκτίμηση ταχύτητας ανέμου"),
    ("2021-09-01", "2021-10-04", "Υποεκτίμηση ταχύτητας ανέμου"),
]

RAIN_ISSUE_INTERVALS = [
    (pd.Timestamp(s), pd.Timestamp(e), mm, note)
    for s, e, mm, note in KNOWN_RAIN_ISSUES
]

WIND_DIRS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
             "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


# ------------------------------------------------------------
# Φόρτωση δεδομένων με cache
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


uploaded_file = st.sidebar.file_uploader("Ανεβάστε αρχείο CSV", type=["csv"])
if uploaded_file is not None:
    df = load_data(uploaded_file)
else:
    try:
        df = load_data("kastoria_daily_all_years.csv")
        st.sidebar.success("Φορτώθηκε το αρχείο: kastoria_daily_all_years.csv")
    except FileNotFoundError:
        st.error("Ανεβάστε το αρχείο CSV ή τοποθετήστε το στον ίδιο φάκελο "
                 "με όνομα 'kastoria_daily_all_years.csv'.")
        st.stop()

# ------------------------------------------------------------
# Βοηθητικές συναρτήσεις
# ------------------------------------------------------------
def find_streaks(dates, condition, min_length=1):
    """
    Δέχεται boolean Series με ευρετήριο συνεχόμενες ημερολογιακές ημέρες και
    επιστρέφει DataFrame με τα σερί (έναρξη, λήξη, διάρκεια) όπου η συνθήκη
    ισχύει για τουλάχιστον `min_length` συνεχόμενες ημέρες.
    Ημέρες που λείπουν διακόπτουν το σερί.
    """
    s = condition.reindex(pd.date_range(dates.min(), dates.max(), freq="D"),
                          fill_value=False)
    groups = (~s).cumsum()
    out = []
    for g, run in s[s].groupby(groups[s]):
        if len(run) >= min_length:
            out.append({"Έναρξη": run.index.min(), "Λήξη": run.index.max(),
                        "Ημέρες": len(run)})
    return pd.DataFrame(out)


def longest_dry_spell(data):
    """
    (διάρκεια, έναρξη, λήξη) του μεγαλύτερου σερί συνεχόμενων ΗΜΕΡΟΛΟΓΙΑΚΩΝ
    ημερών με Rain_mm == 0. Κενές τιμές (NaN) ή ημερομηνίες που λείπουν από
    το αρχείο διακόπτουν το σερί, ώστε τα κενά να μη διογκώνουν το ρεκόρ.
    """
    s = data.set_index("Date")["Rain_mm"]
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"))
    is_dry = (s == 0)
    streaks = find_streaks(s.index.to_series(), is_dry)
    if streaks.empty:
        return 0, None, None
    best = streaks.loc[streaks["Ημέρες"].idxmax()]
    return int(best["Ημέρες"]), best["Έναρξη"], best["Λήξη"]


def rain_issues_overlapping(start, end):
    """Γνωστά προβλήματα βροχομετρικών δεδομένων που τέμνουν το [start, end]."""
    if start is None or end is None:
        return []
    return [(s, e, mm, note) for s, e, mm, note in RAIN_ISSUE_INTERVALS
            if s <= end and e >= start]


# ------------------------------------------------------------
# Φίλτρα (sidebar)
# ------------------------------------------------------------
st.sidebar.header("Φίλτρα")
min_date = df["Date"].min().date()
max_date = df["Date"].max().date()
date_range = st.sidebar.date_input(
    "Εύρος ημερομηνιών", value=(min_date, max_date),
    min_value=min_date, max_value=max_date,
)
if len(date_range) == 2:
    start_date, end_date = date_range
    mask = (df["Date"].dt.date >= start_date) & (df["Date"].dt.date <= end_date)
    df_filtered = df.loc[mask].copy()
else:
    df_filtered = df.copy()

vars_for_plot = st.sidebar.multiselect(
    "Μεταβλητές χρονοσειράς",
    options=list(VAR_LABELS.keys()),
    default=["MeanTemp", "HighTemp", "LowTemp"],
    format_func=lambda v: VAR_LABELS[v],
)

# ------------------------------------------------------------
# Συνοπτικά στατιστικά
# ------------------------------------------------------------
st.header("📊 Συνοπτικά Στατιστικά")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Μέση θερμοκρασία", f"{df_filtered['MeanTemp'].mean():.1f} °C")
with col2:
    st.metric("Συνολική βροχόπτωση", f"{df_filtered['Rain_mm'].sum():.1f} mm",
              help="Καταγεγραμμένο σύνολο· λόγω γνωστών βλαβών το πραγματικό "
                   "είναι υψηλότερο (βλ. ενότητα Ποιότητα Δεδομένων).")
with col3:
    st.metric("Μέγιστη ριπή ανέμου", f"{df_filtered['MaxWindSpeed_kmh'].max():.1f} km/h")
with col4:
    st.metric("Ημέρες με βροχή > 1 mm", f"{(df_filtered['Rain_mm'] > 1).sum()}")

# ------------------------------------------------------------
# Ρεκόρ (εντός του επιλεγμένου εύρους ημερομηνιών)
# ------------------------------------------------------------
st.header("🏅 Ρεκόρ")

hot_row = df_filtered.loc[df_filtered["HighTemp"].idxmax()]
cold_row = df_filtered.loc[df_filtered["LowTemp"].idxmin()]
rain_valid = df_filtered.dropna(subset=["Rain_mm"])
wet_row = rain_valid.loc[rain_valid["Rain_mm"].idxmax()] if not rain_valid.empty else None
dry_len, dry_start, dry_end = longest_dry_spell(df_filtered)

rec1, rec2, rec3, rec4 = st.columns(4)
with rec1:
    st.metric("🔥 Θερμότερη μέρα", f"{hot_row['HighTemp']:.1f} °C",
              help=f"Ώρα μεγίστου: {hot_row.get('HighTime', '—')}")
    st.caption(fmt_date(hot_row["Date"]))
with rec2:
    st.metric("🥶 Ψυχρότερη μέρα", f"{cold_row['LowTemp']:.1f} °C",
              help=f"Ώρα ελαχίστου: {cold_row.get('LowTime', '—')}")
    st.caption(fmt_date(cold_row["Date"]))
with rec3:
    if wet_row is not None:
        st.metric("🌧️ Βροχερότερη μέρα", f"{wet_row['Rain_mm']:.1f} mm")
        st.caption(fmt_date(wet_row["Date"]))
    else:
        st.metric("🌧️ Βροχερότερη μέρα", "—")
with rec4:
    st.metric("☀️ Μεγαλύτερη ανομβρία", f"{dry_len} ημέρες")
    if dry_start is not None:
        st.caption(f"{fmt_date(dry_start)} → {fmt_date(dry_end)}")

# Έλεγχος αξιοπιστίας έναντι του ημερολογίου βλαβών του σταθμού
dry_overlaps = rain_issues_overlapping(dry_start, dry_end)
if dry_overlaps:
    st.warning(
        "⚠️ Η μεγαλύτερη ανομβρία επικαλύπτεται με περίοδο γνωστών απωλειών "
        "βροχομετρικών δεδομένων και μπορεί να είναι τεχνούργημα ελλιπών "
        "δεδομένων: "
        + "· ".join(f"{fmt_date(s)}–{fmt_date(e)} ({note})"
                    for s, e, mm, note in dry_overlaps)
    )
st.caption(
    "Το ρεκόρ βροχερότερης μέρας αφορά την *καταγεγραμμένη* βροχόπτωση. "
    "Σημειώνεται ότι στις 26/06/2018 δεν καταγράφηκαν περίπου 60 mm λόγω "
    "βλάβης αισθητήρα, οπότε το πραγματικό ημερήσιο ρεκόρ μπορεί να διαφέρει. "
    "Η ανομβρία μετρά συνεχόμενες ημερολογιακές ημέρες με 0 mm· ημέρες που "
    "λείπουν ή έχουν κενή τιμή διακόπτουν το σερί (συντηρητικό κάτω όριο)."
)

# ------------------------------------------------------------
# Χρονοσειρές
# ------------------------------------------------------------
st.header("📈 Χρονική Εξέλιξη")
if vars_for_plot:
    fig = px.line(df_filtered, x="Date", y=vars_for_plot,
                  title="Ημερήσιες μεταβλητές",
                  labels={"value": "Τιμή", "Date": "", "variable": "Μεταβλητή"},
                  color_discrete_sequence=px.colors.qualitative.Set1)
    fig.for_each_trace(lambda t: t.update(name=VAR_LABELS.get(t.name, t.name)))
    st.plotly_chart(fig, width='stretch')
else:
    st.info("Επιλέξτε τουλάχιστον μία μεταβλητή από το πλαϊνό μενού.")

st.subheader("🌧️ Ημερήσια Βροχόπτωση")
fig_rain = px.bar(df_filtered, x="Date", y="Rain_mm",
                  title="Ημερήσια βροχόπτωση (mm)",
                  labels={"Rain_mm": "Βροχή (mm)", "Date": ""},
                  color_discrete_sequence=["#1f77b4"])
st.plotly_chart(fig_rain, width='stretch')

# ------------------------------------------------------------
# Ακραίες τιμές (πίνακες top-10)
# ------------------------------------------------------------
st.header("🏆 Ακραίες Τιμές")
tab1, tab2, tab3 = st.tabs(["Θερμότερες ημέρες", "Ψυχρότερες ημέρες",
                            "Βροχερότερες ημέρες"])

GR_COLS = {"Date": "Ημερομηνία", "HighTemp": "Μέγιστη (°C)",
           "MeanTemp": "Μέση (°C)", "LowTemp": "Ελάχιστη (°C)",
           "Rain_mm": "Βροχή (mm)"}

with tab1:
    hottest = df_filtered.nlargest(10, "HighTemp")[
        ["Date", "HighTemp", "MeanTemp", "LowTemp"]].copy()
    hottest["Date"] = hottest["Date"].apply(fmt_date)
    st.dataframe(hottest.rename(columns=GR_COLS).style
                 .highlight_max(subset=["Μέγιστη (°C)"], color="salmon"))

    # Πίτα: σε ποια έτη πέφτουν οι 20 θερμότερες ημέρες της περιόδου
    top20 = df_filtered.nlargest(20, "HighTemp")
    per_year = top20.groupby("Year").size().reset_index(name="Ημέρες")
    per_year["Έτος"] = per_year["Year"].astype(str)
    fig_pie = px.pie(per_year, values="Ημέρες", names="Έτος",
                     title="Κατανομή των 20 θερμότερων ημερών ανά έτος",
                     hole=0.35)
    fig_pie.update_traces(textinfo="label+value+percent",
                          hovertemplate="%{label}: %{value} ημέρες (%{percent})")
    st.plotly_chart(fig_pie, width='stretch')
    st.caption(
        "Η πίτα δείχνει πόσες από τις 20 θερμότερες ημέρες της επιλεγμένης "
        "περιόδου ανήκουν σε κάθε έτος — η συγκέντρωση στα πρόσφατα έτη "
        "αποτυπώνει τη θερμική τάση με μια ματιά."
    )

with tab2:
    coldest = df_filtered.nsmallest(10, "LowTemp")[
        ["Date", "LowTemp", "MeanTemp", "HighTemp"]].copy()
    coldest["Date"] = coldest["Date"].apply(fmt_date)
    st.dataframe(coldest.rename(columns=GR_COLS).style
                 .highlight_min(subset=["Ελάχιστη (°C)"], color="lightblue"))
    cold_plot = coldest.rename(columns=GR_COLS).sort_values(
        "Ελάχιστη (°C)", ascending=False)
    fig_cold = px.bar(cold_plot, x="Ελάχιστη (°C)", y="Ημερομηνία",
                      orientation="h",
                      title="Οι 10 ψυχρότερες ημέρες (κατά ελάχιστη)")
    fig_cold.update_yaxes(type="category")
    st.plotly_chart(fig_cold, width='stretch')

with tab3:
    rainiest = df_filtered.nlargest(10, "Rain_mm")[
        ["Date", "Rain_mm", "MeanTemp"]].copy()
    rainiest["Date"] = rainiest["Date"].apply(fmt_date)
    st.dataframe(rainiest.rename(columns=GR_COLS).style
                 .highlight_max(subset=["Βροχή (mm)"], color="lightgreen"))
    rain_plot = rainiest.rename(columns=GR_COLS).sort_values("Βροχή (mm)")
    fig_rainiest = px.bar(rain_plot, x="Βροχή (mm)", y="Ημερομηνία",
                          orientation="h",
                          title="Οι 10 βροχερότερες ημέρες (καταγεγραμμένες)")
    fig_rainiest.update_yaxes(type="category")
    st.plotly_chart(fig_rainiest, width='stretch')

# ------------------------------------------------------------
# Κλιματικοί δείκτες & τάση (μόνο πλήρη έτη)
# ------------------------------------------------------------
st.header("🌍 Κλιματικοί Δείκτες & Τάση")

complete_years = [y for y, n in df.groupby("Year").size().items() if n >= 330]
df_cy = df[df["Year"].isin(complete_years)]
st.caption(
    f"Υπολογίζονται μόνο στα (σχεδόν) πλήρη έτη: {min(complete_years)}–{max(complete_years)}. "
    "Τα μερικά έτη (το 2008 ξεκινά τον Σεπτέμβριο, το 2026 φτάνει έως τον "
    "Φεβρουάριο) εξαιρούνται ώστε οι ετήσιες τιμές να είναι συγκρίσιμες."
)

indices = df_cy.groupby("Year").agg(
    Καύσωνες=("HighTemp", lambda s: (s >= 35).sum()),
    Θερινές=("HighTemp", lambda s: (s >= 25).sum()),
    Τροπικές_νύχτες=("LowTemp", lambda s: (s >= 20).sum()),
    Παγετός=("LowTemp", lambda s: (s < 0).sum()),
    Ολικός_παγετός=("HighTemp", lambda s: (s < 0).sum()),
    Μέση=("MeanTemp", "mean"),
).reset_index().rename(columns={"Year": "Έτος"})

ic1, ic2 = st.columns(2)
with ic1:
    fig_idx_warm = px.line(
        indices, x="Έτος", y=["Καύσωνες", "Τροπικές_νύχτες"], markers=True,
        title="Θερμοί δείκτες ανά έτος (Ημέρες ≥35 °C, Τροπικές νύχτες ≥20 °C)",
        labels={"value": "Ημέρες", "Έτος": "", "variable": "Δείκτης"})
    fig_idx_warm.for_each_trace(
        lambda t: t.update(name=t.name.replace("_", " ")))
    st.plotly_chart(fig_idx_warm, width='stretch')
with ic2:
    fig_idx_cold = px.line(
        indices, x="Έτος", y=["Παγετός", "Ολικός_παγετός"], markers=True,
        title="Ψυχροί δείκτες ανά έτος (Παγετός Tmin<0 °C, Ολικός παγετός Tmax<0 °C)",
        labels={"value": "Ημέρες", "Έτος": "", "variable": "Δείκτης"})
    fig_idx_cold.for_each_trace(
        lambda t: t.update(name=t.name.replace("_", " ")))
    st.plotly_chart(fig_idx_cold, width='stretch')

# Ετήσια μέση θερμοκρασία με γραμμική τάση
x = indices["Έτος"].to_numpy(dtype=float)
y = indices["Μέση"].to_numpy(dtype=float)
slope, intercept = np.polyfit(x, y, 1)
trend_y = slope * x + intercept

fig_trend = go.Figure()
fig_trend.add_trace(go.Scatter(x=indices["Έτος"], y=indices["Μέση"],
                               mode="lines+markers",
                               name="Ετήσια μέση θερμοκρασία"))
fig_trend.add_trace(go.Scatter(x=indices["Έτος"], y=trend_y, mode="lines",
                               name=f"Τάση: {slope*10:+.2f} °C / δεκαετία",
                               line=dict(dash="dash", color="firebrick")))
fig_trend.update_layout(title="Ετήσια Μέση Θερμοκρασία & Γραμμική Τάση",
                        yaxis_title="°C")
st.plotly_chart(fig_trend, width='stretch')
st.caption(
    "⚠️ Επιφύλαξη ομοιογένειας: ο σταθμός μετεγκαταστάθηκε εντός της Καστοριάς "
    "στις 08/12/2010 (ίδιο υψόμετρο). Οι τάσεις που εκτείνονται πέρα από αυτή "
    "την ημερομηνία αναμειγνύουν δύο θέσεις μέτρησης και χρειάζονται προσοχή "
    "στην ερμηνεία."
)

# ------------------------------------------------------------
# Καύσωνες & ψυχρές εισβολές
# ------------------------------------------------------------
st.header("🌡️ Καύσωνες & Ψυχρές Εισβολές")
hw1, hw2 = st.columns(2)
with hw1:
    heat_thresh = st.slider("Όριο καύσωνα: ημερήσια μέγιστη ≥ (°C)", 30, 42, 35)
    heat_days = st.slider("Ελάχιστες συνεχόμενες ημέρες (καύσωνας)", 2, 7, 3)
with hw2:
    cold_thresh = st.slider("Όριο ψυχρής εισβολής: ημερήσια ελάχιστη ≤ (°C)",
                            -15, 5, -5)
    cold_days = st.slider("Ελάχιστες συνεχόμενες ημέρες (ψυχρή εισβολή)", 2, 7, 3)

t = df_filtered.set_index("Date")
heatwaves = find_streaks(t.index.to_series(), t["HighTemp"] >= heat_thresh, heat_days)
coldspells = find_streaks(t.index.to_series(), t["LowTemp"] <= cold_thresh, cold_days)


def fmt_streak_table(streaks):
    out = streaks.copy()
    out["Έναρξη"] = out["Έναρξη"].apply(fmt_date)
    out["Λήξη"] = out["Λήξη"].apply(fmt_date)
    return out.sort_values("Ημέρες", ascending=False)


hwc1, hwc2 = st.columns(2)
with hwc1:
    st.subheader(f"🔥 Καύσωνες (≥{heat_thresh} °C για {heat_days}+ ημέρες)")
    if heatwaves.empty:
        st.info("Δεν βρέθηκαν καύσωνες με αυτά τα κριτήρια.")
    else:
        st.dataframe(fmt_streak_table(heatwaves), width='stretch')
with hwc2:
    st.subheader(f"❄️ Ψυχρές εισβολές (≤{cold_thresh} °C για {cold_days}+ ημέρες)")
    if coldspells.empty:
        st.info("Δεν βρέθηκαν ψυχρές εισβολές με αυτά τα κριτήρια.")
    else:
        st.dataframe(fmt_streak_table(coldspells), width='stretch')

# ------------------------------------------------------------
# Ροδόγραμμα ανέμου
# ------------------------------------------------------------
st.header("🧭 Ροδόγραμμα Ανέμου")
wind_period = st.radio(
    "Περίοδος",
    ["Όλα τα δεδομένα",
     "Πριν τη μετεγκατάσταση (≤ 07/12/2010, ανεμόμετρο 3 m)",
     "Μετά τη μετεγκατάσταση (≥ 08/12/2010, ανεμόμετρο 5 m)"],
    horizontal=True,
)
wdf = df_filtered.dropna(subset=["DominantWindDir"])
if wind_period.startswith("Πριν"):
    wdf = wdf[wdf["Date"] < STATION_RELOCATION]
elif wind_period.startswith("Μετά"):
    wdf = wdf[wdf["Date"] >= STATION_RELOCATION]

if wdf.empty:
    st.info("Δεν υπάρχουν δεδομένα διεύθυνσης ανέμου για την επιλεγμένη περίοδο.")
else:
    rose = (wdf.groupby("DominantWindDir")
            .agg(Ημέρες=("DominantWindDir", "size"),
                 Ταχύτητα=("AvgWindSpeed_kmh", "mean"))
            .reindex(WIND_DIRS).fillna(0).reset_index())
    fig_rose = go.Figure(go.Barpolar(
        r=rose["Ημέρες"], theta=rose["DominantWindDir"],
        marker=dict(color=rose["Ταχύτητα"], colorscale="Viridis",
                    colorbar=dict(title="Μέση ταχύτητα<br>(km/h)")),
    ))
    fig_rose.update_layout(
        title="Επικρατούσα ημερήσια διεύθυνση ανέμου "
              "(μήκος = ημέρες, χρώμα = μέση ταχύτητα)",
        polar=dict(angularaxis=dict(direction="clockwise", rotation=90)),
    )
    st.plotly_chart(fig_rose, width='stretch')
    st.caption(
        "Το ύψος του ανεμομέτρου άλλαξε από 3 m σε 5 m κατά τη μετεγκατάσταση "
        "(08/12/2010), οπότε οι ταχύτητες πριν/μετά δεν είναι άμεσα "
        "συγκρίσιμες. Οι γνωστές απώλειες και περίοδοι υποεκτίμησης "
        "αναφέρονται στην ενότητα Ποιότητα Δεδομένων."
    )

# ------------------------------------------------------------
# Κλιματολογία: ένα έτος σε σχέση με το «κανονικό»
# ------------------------------------------------------------
st.header("📆 Έτος σε Σύγκριση με το Κλιματολογικό «Κανονικό»")
clim = (df_cy.groupby(["Month", "Day"])["MeanTemp"]
        .agg(Normal="mean", P10=lambda s: s.quantile(0.10),
             P90=lambda s: s.quantile(0.90))
        .reset_index())
clim["DOY"] = pd.to_datetime(
    dict(year=2001, month=clim["Month"], day=clim["Day"]), errors="coerce")
clim = clim.dropna(subset=["DOY"]).sort_values("DOY")

sel_year = st.selectbox("Επιλέξτε έτος για σύγκριση με το κανονικό",
                        sorted(df["Year"].unique(), reverse=True))
yr = df[df["Year"] == sel_year][["Month", "Day", "MeanTemp"]]
comp = clim.merge(yr, on=["Month", "Day"], how="left")

fig_clim = go.Figure()
fig_clim.add_trace(go.Scatter(x=comp["DOY"], y=comp["P90"], mode="lines",
                              line=dict(width=0), showlegend=False))
fig_clim.add_trace(go.Scatter(x=comp["DOY"], y=comp["P10"], mode="lines",
                              line=dict(width=0), fill="tonexty",
                              fillcolor="rgba(128,128,128,0.25)",
                              name="10ο–90ό εκατοστημόριο"))
fig_clim.add_trace(go.Scatter(x=comp["DOY"], y=comp["Normal"], mode="lines",
                              name=f"Κανονικό ({min(complete_years)}–{max(complete_years)})",
                              line=dict(color="black", dash="dot")))
fig_clim.add_trace(go.Scatter(x=comp["DOY"], y=comp["MeanTemp"], mode="lines",
                              name=f"{sel_year}", line=dict(color="firebrick")))
fig_clim.update_layout(
    title=f"Ημερήσια μέση θερμοκρασία: {sel_year} σε σχέση με το κανονικό",
    yaxis_title="°C",
    xaxis=dict(tickmode="array",
               tickvals=pd.to_datetime([f"2001-{m:02d}-01" for m in range(1, 13)]),
               ticktext=GR_MONTHS[1:]))
st.plotly_chart(fig_clim, width='stretch')

# ------------------------------------------------------------
# Ετήσιο δελτίο: μονοσέλιδη σύνοψη επιλεγμένου έτους
# ------------------------------------------------------------
st.header("📰 Ετήσιο Δελτίο")
bul_year = st.selectbox("Επιλέξτε έτος", sorted(df["Year"].unique(), reverse=True),
                        key="bulletin_year")
ydf = df[df["Year"] == bul_year]
is_complete = bul_year in complete_years
if not is_complete:
    st.info(f"Το {bul_year} είναι μερικό έτος — τα σύνολα και οι μετρήσεις "
            "παρακάτω καλύπτουν μόνο τους καταγεγραμμένους μήνες.")

normal_mean = df_cy["MeanTemp"].mean()
y_mean = ydf["MeanTemp"].mean()
y_hot = ydf.loc[ydf["HighTemp"].idxmax()]
y_cold = ydf.loc[ydf["LowTemp"].idxmin()]
y_rain_valid = ydf.dropna(subset=["Rain_mm"])
y_wet = y_rain_valid.loc[y_rain_valid["Rain_mm"].idxmax()] if not y_rain_valid.empty else None
y_dry_len, y_dry_start, y_dry_end = longest_dry_spell(ydf)
y_issues = rain_issues_overlapping(ydf["Date"].min(), ydf["Date"].max())

b1, b2, b3, b4 = st.columns(4)
with b1:
    delta = (f"{y_mean - normal_mean:+.1f} °C από το κανονικό"
             if is_complete else None)
    st.metric("Ετήσια μέση θερμοκρασία", f"{y_mean:.1f} °C", delta=delta)
with b2:
    st.metric("Συνολική βροχή (καταγεγραμμένη)", f"{ydf['Rain_mm'].sum():.0f} mm")
with b3:
    st.metric("Ημέρες ≥ 35 °C", int((ydf["HighTemp"] >= 35).sum()))
with b4:
    st.metric("Τροπικές νύχτες (Tmin ≥ 20 °C)", int((ydf["LowTemp"] >= 20).sum()))

b5, b6, b7, b8 = st.columns(4)
with b5:
    st.metric("Ημέρες παγετού (Tmin < 0 °C)", int((ydf["LowTemp"] < 0).sum()))
with b6:
    st.metric("Θερμότερη μέρα", f"{y_hot['HighTemp']:.1f} °C")
    st.caption(fmt_date(y_hot["Date"], with_year=False))
with b7:
    st.metric("Ψυχρότερη μέρα", f"{y_cold['LowTemp']:.1f} °C")
    st.caption(fmt_date(y_cold["Date"], with_year=False))
with b8:
    st.metric("Μεγαλύτερη ανομβρία", f"{y_dry_len} ημέρες")
    if y_dry_start is not None:
        st.caption(f"{fmt_date(y_dry_start, False)} → {fmt_date(y_dry_end, False)}")

if y_wet is not None:
    st.caption(f"Βροχερότερη μέρα του {bul_year}: {y_wet['Rain_mm']:.1f} mm "
               f"στις {fmt_date(y_wet['Date'], with_year=False)}.")
if y_issues:
    st.warning(f"⚠️ Το {bul_year} περιλαμβάνει {len(y_issues)} καταγεγραμμένα "
               "προβλήματα βροχομετρικών δεδομένων· τα σύνολα βροχής και οι "
               "ανομβρίες του έτους ενδέχεται να επηρεάζονται "
               "(βλ. Ποιότητα Δεδομένων).")

# Μηνιαία ανωμαλία σε σχέση με το μακροχρόνιο κανονικό
if is_complete:
    month_norm = df_cy.groupby("Month")["MeanTemp"].mean()
    month_year = ydf.groupby("Month")["MeanTemp"].mean()
    anom = (month_year - month_norm).reset_index()
    anom.columns = ["Μήνας", "Ανωμαλία"]
    fig_anom = px.bar(anom, x="Μήνας", y="Ανωμαλία",
                      color="Ανωμαλία", color_continuous_scale="RdBu_r",
                      range_color=[-4, 4],
                      title=f"Μηνιαία θερμοκρασιακή ανωμαλία {bul_year} σε σχέση "
                            f"με το κανονικό ({min(complete_years)}–{max(complete_years)})",
                      labels={"Ανωμαλία": "Δ °C"})
    fig_anom.update_layout(coloraxis_showscale=False,
                           xaxis=dict(tickmode="array",
                                      tickvals=list(range(1, 13)),
                                      ticktext=GR_MONTHS[1:]))
    st.plotly_chart(fig_anom, width='stretch')

# ------------------------------------------------------------
# Συγκεντρωτικές όψεις (διπλός άξονας για αναγνωσιμότητα)
# ------------------------------------------------------------
st.header("📅 Συγκεντρωτικές Όψεις")

yearly = df_filtered.groupby("Year").agg(
    MeanTemp=("MeanTemp", "mean"), Rain=("Rain_mm", "sum")).reset_index()
fig_yearly = make_subplots(specs=[[{"secondary_y": True}]])
fig_yearly.add_trace(go.Bar(x=yearly["Year"], y=yearly["Rain"],
                            name="Συνολική βροχή (mm)",
                            marker_color="steelblue", opacity=0.7),
                     secondary_y=False)
fig_yearly.add_trace(go.Scatter(x=yearly["Year"], y=yearly["MeanTemp"],
                                name="Μέση θερμοκρασία (°C)",
                                mode="lines+markers",
                                line=dict(color="firebrick")),
                     secondary_y=True)
fig_yearly.update_layout(title="Ετήσια σύνολα & μέσοι όροι")
fig_yearly.update_yaxes(title_text="Βροχή (mm)", secondary_y=False)
fig_yearly.update_yaxes(title_text="Μέση θερμοκρασία (°C)", secondary_y=True)
st.plotly_chart(fig_yearly, width='stretch')
st.caption("Τα σύνολα βροχής σε έτη με γνωστές απώλειες υποεκτιμούν την "
           "πραγματική ποσότητα (π.χ. 2009, 2018, 2022 — βλ. Ποιότητα Δεδομένων).")

monthly = df_cy.groupby("Month").agg(
    MeanTemp=("MeanTemp", "mean"),
    Rain=("Rain_mm", "sum")).reset_index()
monthly["Rain"] = monthly["Rain"] / len(complete_years)  # μέσος όρος ανά μήνα
fig_monthly = make_subplots(specs=[[{"secondary_y": True}]])
fig_monthly.add_trace(go.Bar(x=monthly["Month"], y=monthly["Rain"],
                             name="Μέση μηνιαία βροχή (mm)",
                             marker_color="steelblue", opacity=0.7),
                      secondary_y=False)
fig_monthly.add_trace(go.Scatter(x=monthly["Month"], y=monthly["MeanTemp"],
                                 name="Μέση θερμοκρασία (°C)",
                                 mode="lines+markers",
                                 line=dict(color="firebrick")),
                      secondary_y=True)
fig_monthly.update_layout(
    title="Κλιμόγραμμα: μέση μηνιαία βροχόπτωση & θερμοκρασία",
    xaxis=dict(tickmode="array", tickvals=list(range(1, 13)),
               ticktext=GR_MONTHS[1:]))
fig_monthly.update_yaxes(title_text="Βροχή (mm)", secondary_y=False)
fig_monthly.update_yaxes(title_text="Μέση θερμοκρασία (°C)", secondary_y=True)
st.plotly_chart(fig_monthly, width='stretch')

st.subheader("🌡️ Χάρτης Θερμότητας Μηνιαίων Θερμοκρασιών")
heatmap_data = df_filtered.pivot_table(index="Year", columns="Month",
                                       values="MeanTemp", aggfunc="mean")
fig_heat = px.imshow(heatmap_data,
                     labels=dict(x="Μήνας", y="Έτος",
                                 color="Μέση θερμ. (°C)"),
                     color_continuous_scale="RdBu_r",
                     title="Μέση θερμοκρασία ανά έτος και μήνα")
fig_heat.update_xaxes(tickmode="array", tickvals=list(range(1, 13)),
                      ticktext=GR_MONTHS[1:])
st.plotly_chart(fig_heat, width='stretch')

# ------------------------------------------------------------
# Κατανομές
# ------------------------------------------------------------
st.header("📊 Κατανομές")
d1, d2 = st.columns(2)
with d1:
    fig_hist = px.histogram(df_filtered, x="MeanTemp", nbins=50,
                            title="Κατανομή θερμοκρασίας",
                            labels={"MeanTemp": "Μέση θερμοκρασία (°C)",
                                    "count": "Ημέρες"})
    fig_hist.update_yaxes(title_text="Ημέρες")
    st.plotly_chart(fig_hist, width='stretch')
with d2:
    fig_box = px.box(df_filtered, x="Month", y="MeanTemp",
                     title="Θερμοκρασία ανά μήνα (θηκόγραμμα)",
                     labels={"Month": "Μήνας",
                             "MeanTemp": "Μέση θερμοκρασία (°C)"})
    fig_box.update_xaxes(tickmode="array", tickvals=list(range(1, 13)),
                         ticktext=GR_MONTHS[1:])
    st.plotly_chart(fig_box, width='stretch')

fig_rain_dist = px.histogram(df_filtered[df_filtered["Rain_mm"] > 0],
                             x="Rain_mm", nbins=50,
                             title="Κατανομή βροχόπτωσης (ημέρες με βροχή)",
                             labels={"Rain_mm": "Βροχή (mm)"})
fig_rain_dist.update_yaxes(title_text="Ημέρες")
st.plotly_chart(fig_rain_dist, width='stretch')

# ------------------------------------------------------------
# Ποιότητα δεδομένων & πληροφορίες σταθμού
# ------------------------------------------------------------
st.header("🔍 Ποιότητα Δεδομένων & Πληροφορίες Σταθμού")
with st.expander("Μεταδεδομένα σταθμού και γνωστά προβλήματα", expanded=False):
    st.markdown(
        """
**Σταθμός: Καστοριά (LGC0)** — υψόμετρο 623 m.
Αρχικά στο Δημαρχείο Μακεδνών (σε γρασίδι, αισθητήρες θερμοκρασίας/υγρασίας
στα 2 m, ανεμόμετρο στα 3 m). **Μετεγκαταστάθηκε εντός της πόλης της
Καστοριάς στις 08/12/2010** σε ίδιο υψόμετρο· το ανεμόμετρο τοποθετήθηκε
στα 5 m. Τα δεδομένα ξεκινούν τον **Σεπτέμβριο του 2008**· δεν υπάρχουν
παρατηρήσεις πριν από αυτή την ημερομηνία.

**Τι σημαίνει αυτό για τον πίνακα**
- Τα σύνολα βροχόπτωσης είναι **κάτω όρια**: πολλές τεκμηριωμένες βλάβες
  αισθητήρων σημαίνουν ότι πραγματική βροχή δεν καταγράφηκε (συνολικά
  350+ mm την περίοδο 2008–2025, με μεγαλύτερη μεμονωμένη απώλεια
  ~60 mm στις 26/06/2018).
- Οι ταχύτητες ανέμου πριν και μετά τις 08/12/2010 **δεν συγκρίνονται
  άμεσα** (ύψος ανεμομέτρου 3 m → 5 m), ενώ σε δύο περιόδους
  (01/03–01/06/2018, 01/09–04/10/2021) είναι γνωστό ότι **υποεκτιμώνται**.
- Η σειρά θερμοκρασίας καλύπτει δύο θέσεις σταθμού· οι μακροχρόνιες τάσεις
  πρέπει να διαβάζονται με επίγνωση αυτής της ανομοιογένειας.
        """
    )

    full_range = pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")
    missing_dates = full_range.difference(df["Date"])
    qc1, qc2, qc3 = st.columns(3)
    qc1.metric("Ημερολογιακές ημέρες που λείπουν από το CSV", len(missing_dates))
    qc2.metric("Ημέρες με κενή τιμή βροχής", int(df["Rain_mm"].isna().sum()))
    qc3.metric("Ημέρες με άγνωστη διεύθυνση ανέμου",
               int(df["DominantWindDir"].isna().sum()))

    st.markdown("**Γνωστά προβλήματα βροχομετρικών δεδομένων (ημερολόγιο σταθμού):**")
    issues_df = pd.DataFrame(
        [(fmt_date(s), fmt_date(e), f"~{mm} mm" if mm else "άγνωστη", note)
         for s, e, mm, note in RAIN_ISSUE_INTERVALS],
        columns=["Από", "Έως", "Εκτ. απώλεια", "Σημείωση"])
    st.dataframe(issues_df, width='stretch')

    st.markdown("**Γνωστά προβλήματα δεδομένων ανέμου (ημερολόγιο σταθμού):**")
    wind_df = pd.DataFrame(
        [(fmt_date(pd.Timestamp(s)), fmt_date(pd.Timestamp(e)), note)
         for s, e, note in KNOWN_WIND_ISSUES],
        columns=["Από", "Έως", "Σημείωση"])
    st.dataframe(wind_df, width='stretch')

    if len(missing_dates) > 0:
        st.markdown("**Ημερομηνίες που απουσιάζουν εντελώς από το CSV:**")
        st.dataframe(
            pd.DataFrame({"Ημερομηνία": [fmt_date(d) for d in missing_dates]}),
            width='stretch', height=200)

# ------------------------------------------------------------
# Πρωτογενή δεδομένα
# ------------------------------------------------------------
st.header("📋 Πρωτογενή Δεδομένα (φιλτραρισμένα)")
if st.checkbox("Εμφάνιση πρωτογενών δεδομένων"):
    raw = df_filtered[["Date", "MeanTemp", "HighTemp", "LowTemp", "Rain_mm",
                       "AvgWindSpeed_kmh", "MaxWindSpeed_kmh",
                       "DominantWindDir"]].rename(columns={
        "Date": "Ημερομηνία", "MeanTemp": "Μέση (°C)",
        "HighTemp": "Μέγιστη (°C)", "LowTemp": "Ελάχιστη (°C)",
        "Rain_mm": "Βροχή (mm)", "AvgWindSpeed_kmh": "Μέσος άνεμος (km/h)",
        "MaxWindSpeed_kmh": "Μέγ. ριπή (km/h)",
        "DominantWindDir": "Διεύθυνση"})
    st.dataframe(raw)

# ------------------------------------------------------------
# Υποσέλιδο
# ------------------------------------------------------------
st.caption(
    "Πηγή δεδομένων: μετεωρολογικός σταθμός Καστοριάς LGC0 "
    "(Σεπ. 2008 – Φεβ. 2026), υψόμετρο 623 m. Οι καταγραφές βροχόπτωσης και "
    "ανέμου περιλαμβάνουν τεκμηριωμένα κενά· βλ. ενότητα Ποιότητα Δεδομένων. "
    "Ο πίνακας κατασκευάστηκε με Streamlit και Plotly."
)
