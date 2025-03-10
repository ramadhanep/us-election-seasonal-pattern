# Election Cycle Seasonal Chart

This repository contains a **Streamlit** app that displays a Hirsch‑Style seasonal chart for a selected stock symbol. The chart shows historical seasonal patterns segmented by U.S. election cycle categories (Pre‑Election, Election, Mid‑Term, Post‑Election) as well as the current year's year‑to‑date (YTD) performance.

## Features

- **Data Fetching:**  
  Retrieves historical stock data from Yahoo Finance starting from 1971 until the present.
  
- **Dynamic Data Handling:**  
  Splits data into historical (years less than the current year) and current-year data (even if incomplete).

- **Daily Returns & Compounding:**  
  Computes daily returns and uses a Hirsch‑style compounding method to produce smooth seasonal patterns.

- **Election Cycle Categorization:**  
  Categorizes historical data into:
  - Election Year (year % 4 == 0)
  - Post‑Election Year (year % 4 == 1)
  - Mid‑Term Year (year % 4 == 2)
  - Pre‑Election Year (year % 4 == 3)

- **Interactive Chart:**  
  Uses Plotly to generate an interactive chart with a dark‑mode friendly color scheme and a legend positioned at the bottom.

- **Dynamic Labeling:**  
  Automatically adjusts the displayed historical range based on the first available year in the data (so if the selected symbol starts later than 1971, the label reflects that).

## Prerequisites

- **Python 3.7+**

### Required Packages

Install the required packages via pip:

```bash
pip install streamlit requests pandas numpy plotly
