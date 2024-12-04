"""
# Company Information

Julie Meunier

This module defines a function that prepares and saves historical stock price data for five companies: Lululemon, Walmart, Walgreens, Ulta, and Dollartree. 
The data is gathered from Yahoo Finance and saved to CSV files. 
"""

import yfinance as yf
import pandas as pd


def get_share_prices(company, start_date, end_date):
    tick = yf.Ticker(company.upper())
    hist = tick.history(period="10y")

    if hist.empty:
        hist = pd.read_csv(f"data/financial_data/{company}.csv")
        print(f"Could not download {company} data:")
        print(f"Loading {company} data from CSV.")

        return hist

    hist = hist.reset_index() if hist.index.name == "Date" else hist
    hist = hist[(hist["Date"] >= start_date) & (hist["Date"] <= end_date)]
    path = f"data/financial_data/{company}.csv"
    hist.to_csv(path, index=False)
    print(f"{company} data saved to {path}.")

    return hist
