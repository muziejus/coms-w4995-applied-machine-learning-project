"""
# Exploratory Data Analysis

Vi Mai and Moacir P. de Sá Pereira

This module prints a simple table summarizing the data for all five
companies.

"""

import pandas as pd


def eda(companies):
    dfs = []
    for company in companies:
        df = pd.read_parquet(f"data/{company}_merged_data.parquet")
        df["company"] = company
        dfs.append(df)

    merged_data = pd.concat(dfs)
    summary_df = merged_data.groupby("company").agg(
        lowest_price=("close", "min"),
        highest_price=("close", "max"),
        average_price=("close", "mean"),
        lowest_sentiment=("weighted_sentiment", "min"),
        highest_sentiment=("weighted_sentiment", "max"),
        average_sentiment=("weighted_sentiment", "mean"),
        articles_analyzed=("analyzed_article_count", "sum"),
        analyzed_bpe_tokens=("analyzed_bpe_tokens", "sum"),
        buy_percentage=("target", lambda x: (x.dropna() == 1).mean()),
        sell_percentage=("target", lambda x: (x.dropna() == 0).mean()),
    )

    return summary_df
