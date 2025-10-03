import plotly.express as px
import streamlit as st
import pandas as pd
from datetime import datetime
import time

from core.data import (
    fetch_data_from_yahoo,
    remove_incomplete_years,
    compute_daily_returns,
    get_election_cycle_label,
    hirsch_style_seasonal_pattern,
    compute_single_year_pattern,
    day_of_year_to_month_date,
)
from ui.theme import BACKGROUND_COLOR, TEXT_COLOR, GRID_COLOR


def render_home(stock_df: pd.DataFrame):
    st.markdown("<h1 style='margin-bottom:0'>US Election Cycle Seasonal Pattern</h1>", unsafe_allow_html=True)
    st.caption("Hirsch-style seasonal profile with election-cycle overlays")

    symbol_list = stock_df["symbol"].unique().tolist()
    chosen_symbol = st.selectbox("Select a Symbol", symbol_list, index=0)

    st.write("### Display Settings")
    show_all_years = True
    show_pre = True
    show_election = True
    show_mid = True
    show_post = True
    show_current = True

    scale_choice = st.radio(
        "Y-axis scale",
        ["Linear (% change)", "Logarithmic (cumulative factor)"],
        index=0,
        horizontal=True,
    )

    current_year = datetime.now().year
    start_dt = datetime(1971, 1, 1)
    end_dt = datetime.now()
    start_unix = int(time.mktime(start_dt.timetuple()))
    end_unix = int(time.mktime(end_dt.timetuple()))

    with st.spinner("Fetching data from Yahoo Finance..."):
        df_raw = fetch_data_from_yahoo(chosen_symbol, start_unix, end_unix)

    df_raw['year'] = df_raw['date'].dt.year
    df_hist_raw = df_raw[df_raw['year'] < current_year].copy()
    df_current_raw = df_raw[df_raw['year'] == current_year].copy()

    df_hist_clean = remove_incomplete_years(df_hist_raw, min_days=200)
    df_hist_daily = compute_daily_returns(df_hist_clean)
    df_current_daily = compute_daily_returns(df_current_raw)

    df_hist_daily['cycle'] = df_hist_daily['year'].apply(get_election_cycle_label)

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
        "Pre-Election Year": "#80FF80",
        "Election Year": "#687FE5",
        "Mid-Term Year": "#E62727",
        "Post-Election Year": "#FFC107",
        f"Current Year ({current_year} YTD)": "#FAEAB1",
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

    # Axis ticks and hover
    fig.update_xaxes(tickformat="%b", dtick="M1")
    if y_col == "pct_change_ytd":
        hovertemplate = "%{x|%b %d}<br>%{y:.2f}%<extra></extra>"
    else:
        hovertemplate = "%{x|%b %d}<br>%{y:.3f}x<extra></extra>"
    fig.update_traces(hovertemplate=hovertemplate)

    fig.update_layout(
        height=620,
        width=600,
        paper_bgcolor=BACKGROUND_COLOR,
        plot_bgcolor=BACKGROUND_COLOR,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        font=dict(color=TEXT_COLOR),
        hovermode="x",
        hoverlabel=dict(bgcolor=BACKGROUND_COLOR, font_color=TEXT_COLOR, bordercolor=GRID_COLOR),
    )
    fig.update_yaxes(type=yaxis_type, gridcolor=GRID_COLOR)
    fig.update_xaxes(gridcolor=GRID_COLOR)

    st.plotly_chart(fig, use_container_width=False)
