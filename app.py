# app.py
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

st.set_page_config(page_title="Marugame Udon — Seasonality & Traffic", layout="wide")
st.title("🍜 Marugame Udon — Seasonality & Traffic")
st.caption("Upload CSV/XLSX exported from Google Sheets. App detects Sales/Revenue & Bowls/Transactions columns automatically.")

# -----------------------
# Helpers
# -----------------------
@st.cache_data
def load_file(uploaded_file):
    """Load csv or excel into DataFrame (strings to start)."""
    if uploaded_file is None:
        return pd.DataFrame()
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, dtype=str)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(uploaded_file, dtype=str)
    else:
        st.error("Unsupported file type. Upload a CSV or Excel file.")
        return pd.DataFrame()
    # normalize headers
    df.columns = [str(c).strip() for c in df.columns]
    return df

def parse_money(series):
    """Convert currency-ish strings to numeric float"""
    if series is None:
        return pd.Series(dtype="float64")
    if series.dtype != object:
        return pd.to_numeric(series, errors="coerce")
    cleaned = series.str.replace(r"[\$,]", "", regex=True).str.strip()
    return pd.to_numeric(cleaned, errors="coerce")

def parse_int(series):
    """Convert bowls/transactions strings to numeric"""
    if series is None:
        return pd.Series(dtype="float64")
    if series.dtype != object:
        return pd.to_numeric(series, errors="coerce")
    cleaned = series.str.replace(r"[,\s]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")

def detect_store_pairs(columns):
    """
    Detect pairs of columns where one is Sales/Revenue and the matching is Bowls/Transactions.
    Returns list of tuples: (store_name_base, sales_col, bowls_col)
    """
    sales_cols = []
    bowls_cols = []
    for col in columns:
        c = col.lower()
        if ("sales" in c) or ("revenue" in c) or (re.search(r"\bsales\b", c) or re.search(r"\brevenue\b", c)):
            sales_cols.append(col)
        if ("bowl" in c) or ("transaction" in c) or ("transactions" in c):
            bowls_cols.append(col)

    pairs = []
    for s in sales_cols:
        # remove suffix ' sales' or ' revenue'
        base = re.sub(r"\s*(sales|revenue)\s*$", "", s, flags=re.I).strip()
        # find matching bowls column with same base prefix
        candidates = [b for b in bowls_cols if re.sub(r"\s*(bowls?|transactions?)\s*$", "", b, flags=re.I).strip() == base]
        if candidates:
            pairs.append((base if base else s, s, candidates[0]))
    return pairs

def find_date_column(df):
    # 1) common names
    for cand in ["Date", "date", "DATE"]:
        if cand in df.columns:
            return cand
    # 2) look for the column with most parseable dates
    best_col, best_score = None, -1
    for col in df.columns:
        try:
            parsed = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
            score = parsed.notna().sum()
            if score > best_score:
                best_score = score
                best_col = col
        except Exception:
            continue
    # require at least some parseable dates
    if best_score > 0:
        return best_col
    return None

def reshape_long(df_raw):
    """Return DataFrame with columns: Date, Store, Sales, Bowls"""
    if df_raw.empty:
        return pd.DataFrame(columns=["Date","Store","Sales","Bowls"])
    date_col = find_date_column(df_raw)
    if date_col is None:
        st.error(f"Could not detect a date column. Your headers: {list(df_raw.columns)}")
        st.stop()
    # parse date
    df = df_raw.copy()
    df["__Date"] = pd.to_datetime(df[date_col], errors="coerce", infer_datetime_format=True)
    df = df[df["__Date"].notna()].copy()
    # detect store pairs
    pairs = detect_store_pairs(df.columns.tolist())
    if not pairs:
        st.error("Could not detect Sales/Bowls column pairs. Ensure headers like 'MRG Berkeley Sales' and 'MRG Berkeley Bowls'.")
        st.stop()
    frames = []
    for store_base, sales_col, bowls_col in pairs:
        sales_series = parse_money(df[sales_col]) if sales_col in df.columns else pd.Series([np.nan]*len(df))
        bowls_series = parse_int(df[bowls_col]) if bowls_col in df.columns else pd.Series([np.nan]*len(df))
        tmp = pd.DataFrame({
            "Date": df["__Date"].values,
            "Store": store_base,
            "Sales": sales_series.values,
            "Bowls": bowls_series.values
        })
        frames.append(tmp)
    long_df = pd.concat(frames, ignore_index=True)
    # drop rows where both null
    long_df = long_df[~(long_df["Sales"].isna() & long_df["Bowls"].isna())].copy()
    long_df.sort_values(["Store","Date"], inplace=True)
    long_df.reset_index(drop=True, inplace=True)
    return long_df

def aggregate(df_long, stores, start, end, timeframe, mode):
    """Aggregate filtered df_long according to timeframe and selection.
       timeframe: 'Daily','Weekly','Monthly','Quarterly','Annual'
       mode: 'combined' or 'by_store'
    """
    if df_long.empty:
        return pd.DataFrame()
    mask = (df_long["Date"] >= pd.to_datetime(start)) & (df_long["Date"] <= pd.to_datetime(end))
    filt = df_long.loc[mask & df_long["Store"].isin(stores)].copy()
    if filt.empty:
        return pd.DataFrame()
    # choose rule - weekly anchored to Monday
    tf_map = {"Daily":"D", "Weekly":"W-MON", "Monthly":"M", "Quarterly":"Q", "Annual":"Y"}
    rule = tf_map.get(timeframe, "M")
    if mode == "combined":
        g = filt.set_index("Date").resample(rule)[["Sales","Bowls"]].sum().reset_index()
        g["Store"] = "All Selected"
        return g[["Date","Store","Sales","Bowls"]]
    # by store
    g = (filt.set_index("Date")
           .groupby("Store")
           .resample(rule)[["Sales","Bowls"]]
           .sum()
           .reset_index())
    return g[["Date","Store","Sales","Bowls"]]

def dual_axis_plot(df_agg, timeframe, chart_type):
    """Plot combined sales/bowls (single series) with dual axis."""
    if df_agg.empty:
        st.info("No aggregated data to plot.")
        return
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if chart_type=="Line":
        fig.add_trace(go.Scatter(x=df_agg["Date"], y=df_agg["Sales"], mode="lines+markers", name="Sales ($)"), secondary_y=False)
        fig.add_trace(go.Scatter(x=df_agg["Date"], y=df_agg["Bowls"], mode="lines+markers", name="Bowls"), secondary_y=True)
    else:
        # Bar for Sales, line for Bowls
        fig.add_trace(go.Bar(x=df_agg["Date"], y=df_agg["Sales"], name="Sales ($)"), secondary_y=False)
        fig.add_trace(go.Scatter(x=df_agg["Date"], y=df_agg["Bowls"], mode="lines+markers", name="Bowls"), secondary_y=True)
    fig.update_layout(title=f"{timeframe} Sales & Bowls — Combined", hovermode="x unified", legend=dict(orientation="h", y=-0.2))
    fig.update_yaxes(title_text="Sales ($)", secondary_y=False)
    fig.update_yaxes(title_text="Bowls (Traffic)", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

def show_metrics(df_for_kpis, start_date, end_date):
    """Compute and show KPI metrics for the filtered selection (df_for_kpis should be raw daily rows or aggregated daily)"""
    if df_for_kpis.empty:
        st.info("No data for selected filters.")
        return
    # use raw daily granularity to compute totals
    total_sales = float(df_for_kpis["Sales"].sum())
    total_bowls = float(df_for_kpis["Bowls"].sum())
    sales_per_person = total_sales/total_bowls if total_bowls else np.nan

    # period length in days & months
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    days = (end - start).days + 1
    months = (end.year - start.year)*12 + (end.month - start.month) + 1
    # averages
    avg_daily_sales = total_sales/days if days else np.nan
    avg_daily_bowls = total_bowls/days if days else np.nan
    avg_weekly_sales = avg_daily_sales * 7
    avg_weekly_bowls = avg_daily_bowls * 7
    avg_monthly_sales = total_sales / months if months else np.nan
    avg_monthly_bowls = total_bowls / months if months else np.nan
    annualized_sales = avg_daily_sales * 365
    annualized_bowls = avg_daily_bowls * 365

    # display metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Sales (selected)", f"${total_sales:,.0f}")
    c2.metric("Total Bowls (selected)", f"{total_bowls:,.0f}")
    c3.metric("Sales / Person", f"${sales_per_person:,.2f}" if np.isfinite(sales_per_person) else "—")
    c4.metric("Days in window", f"{days}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Avg Daily Sales", f"${avg_daily_sales:,.0f}")
    c6.metric("Avg Weekly Sales", f"${avg_weekly_sales:,.0f}")
    c7.metric("Avg Monthly Sales", f"${avg_monthly_sales:,.0f}")
    c8.metric("Annualized Sales", f"${annualized_sales:,.0f}")

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Avg Daily Bowls", f"{avg_daily_bowls:,.0f}")
    d2.metric("Avg Weekly Bowls", f"{avg_weekly_bowls:,.0f}")
    d3.metric("Avg Monthly Bowls", f"{avg_monthly_bowls:,.0f}")
    d4.metric("Annualized Bowls", f"{annualized_bowls:,.0f}")

# -----------------------
# UI: File upload & filters
# -----------------------
st.markdown("## Upload data")
uploaded = st.file_uploader("Upload CSV or Excel (your Google Sheet export)", type=["csv","xlsx","xls"])
if uploaded is None:
    st.info("Upload your file (CSV or Excel). Expected: Date column, and for each store a pair of columns like 'MRG Berkeley Sales' and 'MRG Berkeley Bowls'.")
    st.stop()

raw = load_file(uploaded)
# show column names if user wants quick check
with st.expander("Show uploaded columns"):
    st.write(list(raw.columns)[:100])

# reshape to long
long_df = reshape_long(raw)
# quick preview
with st.expander("Preview reshaped data (first 30 rows)"):
    st.dataframe(long_df.head(30))

# Filters in sidebar
with st.sidebar:
    st.header("Filters")
    min_date = long_df["Date"].min().date()
    max_date = long_df["Date"].max().date()
    date_range = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    if isinstance(date_range, tuple):
        start_date, end_date = date_range
    else:
        start_date, end_date = date_range, max_date

    all_stores = sorted(long_df["Store"].unique().tolist())
    stores = st.multiselect("Stores (multi-select)", options=all_stores, default=all_stores)

    timeframe = st.selectbox("Timeline", ["Daily","Weekly","Monthly","Quarterly","Annual"], index=1)
    chart_type = st.radio("Chart Type", ["Line","Bar"], index=0)
    mode = st.radio("Mode", ["Combined (sum across selected stores)", "By store (compare selected stores)"], index=0)

# Show KPIs (based on raw long_df filtered by stores/date)
mask_raw = (long_df["Date"] >= pd.to_datetime(start_date)) & (long_df["Date"] <= pd.to_datetime(end_date)) & (long_df["Store"].isin(stores))
filtered_raw = long_df.loc[mask_raw].copy()
show_metrics(filtered_raw, start_date, end_date)

st.divider()

# Aggregate for charting
if mode.startswith("Combined"):
    agg = aggregate(long_df, stores, start_date, end_date, timeframe, "combined")
    if agg.empty:
        st.warning("No aggregated data found for selected filters.")
    else:
        dual_axis_plot(agg, timeframe, chart_type)
else:
    agg = aggregate(long_df, stores, start_date, end_date, timeframe, "by_store")
    if agg.empty:
        st.warning("No aggregated data found for selected filters.")
    else:
        # Sales chart
        if chart_type == "Line":
            fig_sales = px.line(agg, x="Date", y="Sales", color="Store", markers=True, title=f"{timeframe} Sales by Store")
            fig_bowls = px.line(agg, x="Date", y="Bowls", color="Store", markers=True, title=f"{timeframe} Bowls by Store")
        else:
            fig_sales = px.bar(agg, x="Date", y="Sales", color="Store", barmode="group", title=f"{timeframe} Sales by Store")
            fig_bowls = px.bar(agg, x="Date", y="Bowls", color="Store", barmode="group", title=f"{timeframe} Bowls by Store")
        st.plotly_chart(fig_sales, use_container_width=True)
        st.plotly_chart(fig_bowls, use_container_width=True)

# Download aggregated CSV
if 'agg' in locals() and not agg.empty:
    csv_bytes = agg.to_csv(index=False).encode("utf-8")
    st.download_button("Download aggregated CSV", csv_bytes, file_name=f"marugame_{timeframe.lower()}_aggregated.csv", mime="text/csv")
