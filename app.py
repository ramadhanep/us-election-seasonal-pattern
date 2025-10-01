import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
import time

# --------------------------------------------------
# Set Streamlit page config for wide layout
# --------------------------------------------------
st.set_page_config(layout="wide", page_title="Election Cycle Seasonal Chart")

# --------------------------------------------------
# 1) HELPER FUNCTIONS
# --------------------------------------------------

def fetch_data_from_yahoo(symbol, start_date, end_date):
    """
    Fetch daily data from Yahoo Finance for the given symbol between
    start_date and end_date (Unix timestamps). Returns a DataFrame with 
    columns: ['date', 'adjclose'].
    """
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?interval=1d&period1={start_date}&period2={end_date}"
    )
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/110.0.0.0 Safari/537.36"
        )
    }
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    result = data["chart"]["result"][0]
    timestamps = result["timestamp"]
    adjclose = result["indicators"]["adjclose"][0]["adjclose"]
    df = pd.DataFrame({
        "date": pd.to_datetime(timestamps, unit='s'),
        "adjclose": adjclose
    })
    return df

def remove_incomplete_years(df, min_days=200):
    """
    Remove any year from the DataFrame that does not have at least `min_days`
    data points. This is applied only to historical data.
    """
    df = df.copy()
    df['year'] = df['date'].dt.year
    counts = df.groupby('year')['date'].count()
    valid_years = counts[counts >= min_days].index
    return df[df['year'].isin(valid_years)]

def compute_daily_returns(df):
    """
    For each year, compute the daily return:
      daily_return = (adjclose_t / adjclose_(t-1)) - 1.
    Stores the result in a new column 'daily_return', and also sets 'day_of_year'.
    """
    df = df.copy().sort_values('date')
    df['year'] = df['date'].dt.year
    df['day_of_year'] = df['date'].dt.dayofyear
    df['prev_close'] = df.groupby('year')['adjclose'].shift(1)
    df['daily_return'] = df['adjclose'] / df['prev_close'] - 1
    df['daily_return'] = df['daily_return'].fillna(0.0)
    return df

def get_election_cycle_label(year):
    """
    Classify a year into one of the election cycle categories:
      - 'Election Year' if year % 4 == 0
      - 'Post-Election Year' if year % 4 == 1
      - 'Mid-Term Year' if year % 4 == 2
      - 'Pre-Election Year' if year % 4 == 3
    """
    cyc = year % 4
    if cyc == 0:
        return "Election Year"
    elif cyc == 1:
        return "Post-Election Year"
    elif cyc == 2:
        return "Mid-Term Year"
    else:
        return "Pre-Election Year"

def hirsch_style_seasonal_pattern(df):
    """
    Given a DataFrame of daily returns (for multiple years), pivot the data 
    so that rows are 'day_of_year' and columns are years, then average the 
    daily returns, compound them, and convert to cumulative % change.
    Returns a DataFrame with ['day_of_year', 'pct_change_ytd'].
    """
    pivot_df = df.pivot(index='day_of_year', columns='year', values='daily_return')
    pivot_df['avg_daily_return'] = pivot_df.mean(axis=1, skipna=True)
    pivot_df = pivot_df.sort_index()  # ensure day_of_year ascending
    pivot_df['cumulative_factor'] = (1 + pivot_df['avg_daily_return']).cumprod()
    pivot_df['pct_change_ytd'] = (pivot_df['cumulative_factor'] - 1.0) * 100.0
    out = pivot_df.reset_index()[['day_of_year', 'pct_change_ytd', 'cumulative_factor']]
    return out

def compute_single_year_pattern(df, single_year):
    """
    For a single year, compound its daily_return to get a Hirsch-style line.
    Returns a DataFrame with ['day_of_year', 'pct_change_ytd'].
    """
    df = df.copy()
    # Ensure columns exist
    if 'date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    if 'year' not in df.columns:
        df['year'] = df['date'].dt.year.astype(int)
    else:
        df['year'] = df['year'].astype(int)
    if 'day_of_year' not in df.columns:
        df['day_of_year'] = df['date'].dt.dayofyear

    single_year = int(single_year)
    temp = df[df['year'] == single_year].copy()
    if temp.empty:
        st.write(f"No data found for year {single_year}. Available years: {df['year'].unique()}")
        return pd.DataFrame(columns=['day_of_year', 'pct_change_ytd'])
    temp = temp.sort_values('day_of_year')
    temp['cumulative_factor'] = (1 + temp['daily_return']).cumprod()
    temp['pct_change_ytd'] = (temp['cumulative_factor'] - 1.0) * 100.0
    return temp[['day_of_year', 'pct_change_ytd', 'cumulative_factor']]

def day_of_year_to_month_date(day_of_year):
    """
    Convert an integer day_of_year into a dummy date (using year 2000) so we 
    can display months on the x-axis.
    """
    base = datetime(2000, 1, 1)
    return base + timedelta(days=day_of_year - 1)

# --------------------------------------------------
# 2) STREAMLIT APP
# --------------------------------------------------
def render_home(stock_df: pd.DataFrame):
    st.markdown("<h1 style='margin-bottom:0'>US Election Cycle Seasonal Pattern</h1>", unsafe_allow_html=True)
    st.caption("Hirsch-style seasonal profile with election-cycle overlays")

    # 2.2) Symbol selection
    symbol_list = stock_df["symbol"].unique().tolist()
    chosen_symbol = st.selectbox("Select a Symbol", symbol_list, index=0)

    # 2.3) Cycle visibility config (keep current logic as-is: all shown)
    st.write("### Display Settings")
    show_all_years = True
    show_pre       = True
    show_election  = True
    show_mid       = True
    show_post      = True
    show_current   = True

    # 2.3b) Y-axis scale toggle
    scale_choice = st.radio(
        "Y-axis scale",
        ["Linear (% change)", "Logarithmic (cumulative factor)"],
        index=0,
        horizontal=True,
    )

    # 2.4) Define dynamic date ranges
    current_year = datetime.now().year
    start_dt = datetime(1971, 1, 1)
    end_dt = datetime.now()
    start_unix = int(time.mktime(start_dt.timetuple()))
    end_unix = int(time.mktime(end_dt.timetuple()))

    with st.spinner("Fetching data from Yahoo Finance..."):
        df_raw = fetch_data_from_yahoo(chosen_symbol, start_unix, end_unix)

    # Ensure year column exists
    df_raw['year'] = df_raw['date'].dt.year

    # 2.5) Split raw data
    df_hist_raw = df_raw[df_raw['year'] < current_year].copy()
    df_current_raw = df_raw[df_raw['year'] == current_year].copy()

    # 2.6) Clean and compute daily returns
    df_hist_clean = remove_incomplete_years(df_hist_raw, min_days=200)
    df_hist_daily = compute_daily_returns(df_hist_clean)
    df_current_daily = compute_daily_returns(df_current_raw)

    # Add election cycle labels (historical only)
    df_hist_daily['cycle'] = df_hist_daily['year'].apply(get_election_cycle_label)

    # Build lines
    lines_data = []

    if show_all_years and not df_hist_daily.empty:
        first_year = int(df_hist_daily['year'].min())
        df_hirsch_all = hirsch_style_seasonal_pattern(df_hist_daily)
        df_hirsch_all["category"] = f"All Years ({first_year}-{current_year - 1})"
        lines_data.append(df_hirsch_all)
    else:
        first_year = current_year

    if show_pre:
        df_pre = df_hist_daily[df_hist_daily['cycle'] == "Pre-Election Year"]
        if not df_pre.empty:
            df_hirsch_pre = hirsch_style_seasonal_pattern(df_pre)
            df_hirsch_pre["category"] = "Pre-Election Year"
            lines_data.append(df_hirsch_pre)

    if show_election:
        df_elec = df_hist_daily[df_hist_daily['cycle'] == "Election Year"]
        if not df_elec.empty:
            df_hirsch_elec = hirsch_style_seasonal_pattern(df_elec)
            df_hirsch_elec["category"] = "Election Year"
            lines_data.append(df_hirsch_elec)

    if show_mid:
        df_mid = df_hist_daily[df_hist_daily['cycle'] == "Mid-Term Year"]
        if not df_mid.empty:
            df_hirsch_mid = hirsch_style_seasonal_pattern(df_mid)
            df_hirsch_mid["category"] = "Mid-Term Year"
            lines_data.append(df_hirsch_mid)

    if show_post:
        df_post = df_hist_daily[df_hist_daily['cycle'] == "Post-Election Year"]
        if not df_post.empty:
            df_hirsch_post = hirsch_style_seasonal_pattern(df_post)
            df_hirsch_post["category"] = "Post-Election Year"
            lines_data.append(df_hirsch_post)

    if show_current:
        df_cy = compute_single_year_pattern(df_current_daily, current_year)
        if not df_cy.empty:
            df_cy["category"] = f"Current Year ({current_year} YTD)"
            lines_data.append(df_cy)

    if not lines_data:
        st.warning("No data available to plot for the selected configuration.")
        return

    df_plot = pd.concat(lines_data, ignore_index=True)
    df_plot["date_for_x"] = df_plot["day_of_year"].apply(day_of_year_to_month_date)

    color_map = {
        f"All Years ({first_year}-{current_year - 1})": "#E5E9F0",
        "Pre-Election Year": "#00E676",
        "Election Year": "#F6C445",
        "Mid-Term Year": "#4DD0E1",
        "Post-Election Year": "#80FF80",
        f"Current Year ({current_year} YTD)": "#FF7043",
    }

    if scale_choice.startswith("Linear"):
        y_col = "pct_change_ytd"
        y_label = "Cumulative % Change"
        yaxis_type = "linear"
    else:
        y_col = "cumulative_factor"
        y_label = "Cumulative Factor (log scale)"
        yaxis_type = "log"

    fig = px.line(
        df_plot,
        x="date_for_x",
        y=y_col,
        color="category",
        labels={"date_for_x": "Month", y_col: y_label},
        title=f"Election Cycle Seasonal Chart: {chosen_symbol}"
    )

    current_cycle_label = get_election_cycle_label(current_year)
    all_years_label = f"All Years ({first_year}-{current_year - 1})"
    current_year_label = f"Current Year ({current_year} YTD)"
    allowed_categories = {all_years_label, current_year_label, current_cycle_label}

    for trace in fig.data:
        cat_name = trace.name
        if cat_name in color_map:
            trace.line.color = color_map[cat_name]
        trace.line.width = 2
        if cat_name not in allowed_categories:
            trace.visible = 'legendonly'

    # Month ticks on axis; hover shows full month-day detail
    fig.update_xaxes(tickformat="%b", dtick="M1")

    # Detailed hover with date granularity
    if y_col == "pct_change_ytd":
        hovertemplate = "%{x|%b %d}<br>%{y:.2f}%<extra></extra>"
    else:
        hovertemplate = "%{x|%b %d}<br>%{y:.3f}x<extra></extra>"
    fig.update_traces(hovertemplate=hovertemplate)
    fig.update_layout(
        height=620,
        paper_bgcolor="#050505",
        plot_bgcolor="#050505",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        font=dict(color="#E5E9F0"),
        hovermode="x",
        hoverlabel=dict(bgcolor="#12171A", font_color="#E5E9F0", bordercolor="#1C2428"),
    )
    fig.update_yaxes(type=yaxis_type, gridcolor="#1C2428")
    fig.update_xaxes(gridcolor="#1C2428")

    st.plotly_chart(fig, use_container_width=True)


def render_docs(stock_df: pd.DataFrame):
    st.markdown("<h1>Docs</h1>", unsafe_allow_html=True)
    st.write("This app displays seasonal patterns by election cycle for your selected stock.")
    st.write("Symbols loaded from stocks.csv:")
    st.dataframe(stock_df, use_container_width=True)


def main():
    # Load symbol list once
    stock_df = pd.read_csv("stocks.csv")

    # Sidebar navigation
    st.sidebar.markdown("<h2 style='margin-bottom:0'>Menu</h2>", unsafe_allow_html=True)
    page = st.sidebar.radio("", ["Home", "Docs"], index=0)

    st.sidebar.markdown("---")
    st.sidebar.caption("Created by [Dy.](https://ramadhanep.com)")

    if page == "Home":
        render_home(stock_df)
    else:
        render_docs(stock_df)

if __name__ == "__main__":
    main()
