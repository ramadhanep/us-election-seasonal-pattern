from datetime import datetime, timedelta
import time
import requests
import pandas as pd
import streamlit as st


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
    df = df.copy()
    df['year'] = df['date'].dt.year
    counts = df.groupby('year')['date'].count()
    valid_years = counts[counts >= min_days].index
    return df[df['year'].isin(valid_years)]


def compute_daily_returns(df):
    df = df.copy().sort_values('date')
    df['year'] = df['date'].dt.year
    df['day_of_year'] = df['date'].dt.dayofyear
    df['prev_close'] = df.groupby('year')['adjclose'].shift(1)
    df['daily_return'] = df['adjclose'] / df['prev_close'] - 1
    df['daily_return'] = df['daily_return'].fillna(0.0)
    return df


def get_election_cycle_label(year):
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
    pivot_df = df.pivot(index='day_of_year', columns='year', values='daily_return')
    pivot_df['avg_daily_return'] = pivot_df.mean(axis=1, skipna=True)
    pivot_df = pivot_df.sort_index()
    pivot_df['cumulative_factor'] = (1 + pivot_df['avg_daily_return']).cumprod()
    pivot_df['pct_change_ytd'] = (pivot_df['cumulative_factor'] - 1.0) * 100.0
    out = pivot_df.reset_index()[['day_of_year', 'pct_change_ytd', 'cumulative_factor']]
    return out


def compute_single_year_pattern(df, single_year):
    df = df.copy()
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
    base = datetime(2000, 1, 1)
    return base + timedelta(days=day_of_year - 1)

