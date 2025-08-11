# app.py
# Streamlit dashboard for COVID-19 data with robust date slider and downloads.

import io
import numpy as np
from datetime import date
import pandas as pd
import streamlit as st
import altair as alt
import plotly.express as px


st.set_page_config(page_title="COVID-19 Impact Dashboard", layout="wide")

# -------------------- Sidebar config: file paths --------------------
st.sidebar.header("Data sources")

vacc_path = st.sidebar.text_input("Path to vaccinations.csv", "data/vaccinations.csv")
covid_path = st.sidebar.text_input("Path to covid_metrics.csv", "data/covid_metrics.csv")

# -------------------- Helpers --------------------
def alias_column(df: pd.DataFrame, targets: list[str], new_name: str):
    """If any of targets exist, create/rename to new_name."""
    for t in targets:
        if t in df.columns:
            if t != new_name:
                df[new_name] = df[t]
            return

def pick_first_col(df: pd.DataFrame, candidates: list[str]):
    """Return the first column name from candidates that exists, else None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None

# -------------------- Load, merge, clean --------------------
@st.cache_data
def load_merged(vacc_path: str, covid_path: str) -> pd.DataFrame:
    vacc_df = pd.read_csv(vacc_path)
    covid_df = pd.read_csv(covid_path)

    # Dates
    vacc_df["date"] = pd.to_datetime(vacc_df["date"], errors="coerce")
    covid_df["date"] = pd.to_datetime(covid_df["date"], errors="coerce")
    vacc_df = vacc_df.dropna(subset=["date"])
    covid_df = covid_df.dropna(subset=["date"])

    # Merge on location + date
    df = pd.merge(
        vacc_df,
        covid_df,
        on=["location", "date"],
        how="inner",
        suffixes=("_vac", "_cov")
    )

    # Canonicalize key columns
    alias_column(df, ["continent_cov", "continent_vac", "continent"], "continent")
    alias_column(df, ["new_cases", "new_cases_cov"], "new_cases")
    alias_column(df, ["new_deaths", "new_deaths_cov"], "new_deaths")
    alias_column(df, ["population_cov", "population_vac", "population"], "population")

    # Coerce numerics we care about
    for col in ["new_cases", "new_deaths", "population",
                "people_fully_vaccinated_per_hundred_vac",
                "people_fully_vaccinated_vac"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Build vaccination_rate (per 100)
    if "people_fully_vaccinated_per_hundred_vac" in df.columns:
        base_vax = df["people_fully_vaccinated_per_hundred_vac"]
        df["vaccination_rate"] = pd.to_numeric(base_vax, errors="coerce")
    else:
        # Derive from counts if per-hundred not present
        if ("people_fully_vaccinated_vac" in df.columns) and ("population" in df.columns):
            df["vaccination_rate"] = (
                pd.to_numeric(df["people_fully_vaccinated_vac"], errors="coerce")
                / pd.to_numeric(df["population"], errors="coerce") * 100
            )
        else:
            df["vaccination_rate"] = pd.NA

    # Sort and forward-fill vaccination_rate within each country
    df.sort_values(["location", "date"], inplace=True)
    df["vaccination_rate"] = (
    df.groupby("location")["vaccination_rate"].transform(lambda s: s.ffill())
)


    # Final tidy
    df = df.dropna(subset=["date"]).copy()
    return df

# Load data with error handling
df = pd.DataFrame()
load_error = None
try:
    df = load_merged(vacc_path, covid_path)
except Exception as e:
    load_error = str(e)

if load_error:
    st.error(f"Failed to load data. Please check file paths and formats.\n\nDetails: {load_error}")
    st.stop()

# -------------------- Sidebar: filters --------------------
st.sidebar.header("Filters")

# Continent selector (if available)
if "continent" in df.columns and df["continent"].notna().any():
    continents = ["All"] + sorted([c for c in df["continent"].dropna().unique()])
    selected_continent = st.sidebar.selectbox("Continent", continents, index=0)
else:
    selected_continent = "All"

# Scope by continent
if selected_continent != "All" and "continent" in df.columns:
    df_scope = df[df["continent"] == selected_continent].copy()
else:
    df_scope = df.copy()

# Country multiselect
if "location" in df_scope.columns:
    countries = sorted(df_scope["location"].dropna().unique().tolist())
    default_countries = countries[:3] if countries else []
    selected_countries = st.sidebar.multiselect("Countries", countries, default=default_countries)
    if selected_countries:
        df_scope = df_scope[df_scope["location"].isin(selected_countries)].copy()
else:
    selected_countries = []

# -------------------- Date range slider (native Python date) --------------------
scope_dates = df_scope["date"].dropna() if "date" in df_scope.columns else pd.Series([], dtype="datetime64[ns]")
if scope_dates.empty:
    st.warning("No dates found in the current scope.")
    st.stop()

min_dt = pd.to_datetime(scope_dates.min())
max_dt = pd.to_datetime(scope_dates.max())
min_date = date(min_dt.year, min_dt.month, min_dt.day)
max_date = date(max_dt.year, max_dt.month, max_dt.day)

date_range = st.sidebar.slider(
    "Date range",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
    format="YYYY-MM-DD",
)
start_date, end_date = date_range

# Apply date filter using .dt.date to compare with Python date
df_filtered = df_scope[(df_scope["date"].dt.date >= start_date) & (df_scope["date"].dt.date <= end_date)].copy()
df_filtered.sort_values("date", inplace=True)

# -------------------- Header --------------------
st.title("COVID-19 Impact Dashboard")
st.caption("Analyze the relationship between vaccination rates, cases, and deaths across countries and continents.")

c1, c2 = st.columns(2)
with c1:
    st.text_input("Min date in scope", str(min_dt.date()), disabled=True)
with c2:
    st.text_input("Max date in scope", str(max_dt.date()), disabled=True)

# -------------------- KPIs --------------------
k1, k2, k3 = st.columns(3)

if not df_filtered.empty:
    total_cases = int(df_filtered.get("new_cases", pd.Series(dtype=float)).fillna(0).sum())
    total_deaths = int(df_filtered.get("new_deaths", pd.Series(dtype=float)).fillna(0).sum())

    # Latest valid vaccination_rate within selected range per country
    if "vaccination_rate" in df_filtered.columns and "location" in df_filtered.columns:
        vax_ffill = (
            df_filtered.sort_values(["location", "date"])
            .groupby("location")["vaccination_rate"]
            .ffill()
        )
        last_per_country = vax_ffill.groupby(df_filtered["location"]).last()
        latest_vax_avg = last_per_country.dropna().mean()
    else:
        latest_vax_avg = pd.NA
else:
    total_cases = 0
    total_deaths = 0
    latest_vax_avg = pd.NA

k1.metric("Total cases (range)", f"{total_cases:,}")
k2.metric("Total deaths (range)", f"{total_deaths:,}")
vax_display = f"{latest_vax_avg:.2f}" if pd.notna(latest_vax_avg) else "Data not available"
k3.metric("Avg fully vax per 100 (latest in range)", vax_display)

st.divider()

    # -------------------- Downloads --------------------
st.subheader("Download filtered data")

if df_filtered.empty:
    st.info("No data to download for the selected filters and date range.")
else:
    # CSV download only
    fname_csv = f"covid_filtered_{start_date}_{end_date}.csv"
    csv_bytes = df_filtered.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download filtered data (CSV)",
        data=csv_bytes,
        file_name=fname_csv,
        mime="text/csv",
        help="Exports the currently filtered rows and columns to CSV."
    )

# -------------------- Charts --------------------
if df_filtered.empty:
    st.info("No data for the selected filters and date range.")
else:
    # ========== 1) Time series: new cases & deaths (Altair with custom colors) ==========
    st.subheader("New cases and deaths over time")

    plot_cols = []
    if "new_cases" in df_filtered.columns:
        plot_cols.append("new_cases")
    if "new_deaths" in df_filtered.columns:
        plot_cols.append("new_deaths")

    if plot_cols:
        ts = (
            df_filtered
            .groupby("date")[plot_cols]
            .sum(min_count=1)
            .fillna(0)
            .sort_index()
        )

        # Melt time series to long format for Altair
        ts_melted = ts.reset_index().melt("date", var_name="metric", value_name="count")

        # Custom color mapping
        color_scale = alt.Scale(
            domain=["new_cases", "new_deaths"],
            range=["#1f77b4", "#d62728"]  # Blue for cases, Red for deaths
        )

        line_chart = alt.Chart(ts_melted).mark_line().encode(
            x="date:T",
            y="count:Q",
            color=alt.Color("metric:N", scale=color_scale),
            tooltip=["date:T", "metric:N", "count:Q"]
        ).properties(height=320)

        st.altair_chart(line_chart, use_container_width=True)
    else:
        st.warning("Cases/deaths columns not found.")

    st.divider()

    # ========== 2) Plotly choropleth map ==========
    st.subheader("Geographic snapshot")

    # Build per-country aggregates within the selected range
    by_loc = df_filtered.copy()
    by_loc.sort_values(["location", "date"], inplace=True)

    # Latest vaccination rate per country (ffill then last)
    vax_last = None
    if "vaccination_rate" in by_loc.columns:
        vax_last = (
            by_loc.groupby("location")["vaccination_rate"]
            .ffill()
            .groupby(by_loc["location"])
            .last()
            .rename("vaccination_rate_latest")
        )

    # Totals per country
    totals = by_loc.groupby("location")[["new_cases", "new_deaths"]].sum(min_count=1)
    totals = totals.rename(columns={"new_cases": "total_cases", "new_deaths": "total_deaths"}).fillna(0)

    # Population: take last known in range
    pop_last = None
    if "population" in by_loc.columns:
        pop_last = (
            by_loc.groupby("location")["population"]
            .ffill()
            .groupby(by_loc["location"])
            .last()
            .rename("population")
        )

    # Combine
    df_map = totals.copy()
    if vax_last is not None:
        df_map = df_map.join(vax_last, how="left")
    if pop_last is not None:
        df_map = df_map.join(pop_last, how="left")
        # Per 100k metrics (avoid divide by zero)
        with pd.option_context('mode.use_inf_as_na', True):
            df_map["cases_per_100k"] = (df_map["total_cases"] / df_map["population"] * 100_000).where(df_map["population"] > 0)
            df_map["deaths_per_100k"] = (df_map["total_deaths"] / df_map["population"] * 100_000).where(df_map["population"] > 0)

    # Choose metric to map
    metric_options = ["vaccination_rate_latest", "total_cases", "total_deaths"]
    if "cases_per_100k" in df_map.columns and df_map["cases_per_100k"].notna().any():
        metric_options.append("cases_per_100k")
    if "deaths_per_100k" in df_map.columns and df_map["deaths_per_100k"].notna().any():
        metric_options.append("deaths_per_100k")

    map_metric = st.selectbox("Map metric", metric_options, index=0)

    # Color scale suggestion per metric
    scale_map = {
        "vaccination_rate_latest": "YlGnBu",
        "total_cases": "Blues",
        "total_deaths": "Reds",
        "cases_per_100k": "PuBu",
        "deaths_per_100k": "OrRd",
    }
    color_scale = scale_map.get(map_metric, "Plasma")

    # Optional continent join for hover, if available
    continent_last = None
    if "continent" in by_loc.columns:
        continent_last = (
            by_loc.groupby("location")["continent"]
            .ffill()
            .groupby(by_loc["location"])
            .last()
            .rename("continent")
        )
        df_map = df_map.join(continent_last, how="left")

    df_map_reset = df_map.reset_index().rename(columns={"location": "country"})

    if not df_map_reset.empty:
        fig = px.choropleth(
            df_map_reset,
            locations="country",
            locationmode="country names",
            color=map_metric,
            color_continuous_scale=color_scale,
            hover_name="country",
            hover_data={col: True for col in df_map_reset.columns if col not in ["country"]},
            title=None,
        )
        fig.update_layout(height=430, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No country-level data available for the current selection.")

    st.divider()

# ===================
# END
# ===================
st.markdown("Made by Bruna Gil. Data-driven, clean, and powerful.")
