"""
# External Indicators

Tuba Opel

This module defines a function that gathers external economic indicators
from FRED (Federal Reserve Economic Data) and saves them to a CSV file. The
indicators are:

- CPIAUCSL: Consumer Price Index for All Urban Consumers: All Items in U.S. City Average.

  The most well-known measure of inflation, the CPI tracks the prices of items that consumers purchase directly. The Bureau of Labor Statistics (BLS) releases monthly CPI figures.

- PCE: Personal Consumption Expenditures price index

  The Federal Reserve's preferred measure of inflation, the PCE tracks the prices of all items consumed by households, including employer-provided medical care. The Bureau of Economic Analysis (BEA) updates the PCE monthly.

- PPIACO: Producer Price Index by Commodity: All Commodities

  Measures inflation at earlier stages of production and marketing.

- ECIALLCIV: Employment Cost Index: Total compensation: All Civilian Measures inflation in the labor market.

- GDPDEF: Gross Domestic Product: Implicit Price Deflator

  Combines the inflation experiences of governments, businesses, and consumers.

- UNRATE: Civilian Unemployment Rate

  An alternative measure of economic slack, which economists believe is a key factor in determining the inflation rate

- MCUMFN: Capacity Utilization: Manufacturing

- SP500: S&P 500
"""

import pandas as pd
import os
from dotenv import load_dotenv
from fredapi import Fred


def save_external_indicators(start_date, end_date):
    load_dotenv()

    api_key = os.getenv("FRED_API_KEY")
    fred = Fred(api_key=api_key)

    METRICS = [
        "CPIAUCSL",
        "PCE",
        "PPIACO",
        "ECIALLCIV",
        "GDPDEF",
        "UNRATE",
        "MCUMFN",
        "SP500",
    ]

    # Create a date range
    date_range = pd.date_range(start_date, end_date, freq="D")

    # Create an empty dataframe with the date range as index
    df = pd.DataFrame(index=date_range)

    # Iterate over all metrics, fetch data, and append to df
    for m in METRICS:
        series_data = fred.get_series(
            m, observation_start=start_date, observation_end=end_date
        )
        df[m] = series_data

    # Some indicators are monthly and some are quarterly
    # Fill the NaN with the last valid value
    df = df.ffill()
    path = "data/financial_data/external_indicators.csv"
    df.to_csv(path, index_label="Date")
    print(f"External indicators saved to {path}")
