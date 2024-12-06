"""
# Sentiment Analysis

Moacir P. de Sá Pereira

This module defines a function that aggregates the sentiment analysis of
~100,000 news articles about a target company. 

The analysis was done in ProQuest’s TDMStudio environment, so the code is
saved separately in notebooks in the notebooks/ directory. Those notebooks
generated five parquet files, one for each company, with each row devoted
to a single article and with the following columns:

Column | Type | Description
-------|------|------------
`index` | int | A naive index created during aggregation.
`goid` | int | ProQuest’s globally unique identifier for the article in question.
`date`| str | The article’s publication date (`%Y-%m-%d`).
`tokens` | int | A naive (breaking on whitespace) count of tokens in the article.
`corpus` | str | The corpus from which the article comes. Used previously in aggregation.
`daily_article_count` | int | A previously calculated count of all the articles in the corpus from that day.
`daily_token_sum` | int | A sum of all the (naive) tokens from all the articles in the corpus from that day.
`text_sentiment` | float | The overall average sentiment for that article, from (-1, 1) with negative numbers corresponding to negative sentiments and positive with positive.
`text_error` | float | The weighted inverse error in analysis. Higher is more confident in classification.
`text_input_tokens` | int | The number of tokens as analyzed by RoBERTa’s byte-pair encoding tokenizer.

The resulting dataframe returned by this function aggregates the
per-article data into daily data and generalizes the sentiment for each day.

The dataframe has the following columns:

Column | Type | Description
---|---|---
(index)| datetime | The articles’ publication date.
`analyzed_bpe_tokens`| int | The number of tokens as analyzed by RoBERTa’s BPE tokenizer for all the articles analyzed for the day.
`weighted_sentiment` | float | The mean sentiment for the day (from (-1, 1) as above), weighted by each individual article’s length.
`weighted_error` | float | The mean inverse error for the day, weighted by each individual article’s length. Higher is more confident.
`analyzed_naive_tokens` | int | The number of naive tokens (words separated by whitespace) analyzed for the day.
`daily_naive_token_sum` | int | The total number of naive tokens available for the day.
`analyzed_article_count` | int | The number of articles analyzed for the day.
`daily_article_count` | int | The number of available articles for the day.
`sentiment_3d_rolling_mean` | float | Three-day rolling average for sentiment.
`sentiment_7d_rolling_mean` | float | Seven-day rolling average for sentiment.
`sentiment_14d_rolling_mean` | float | Fourteen-day rolling average for sentiment.

"""

import pandas as pd


def weighted_avg(row, value_column, weight_column):
    return (row[value_column] * row[weight_column]).sum() / row[weight_column].sum()


def get_daily_sentiment(company):
    df = pd.read_parquet(f"data/sentiment_data/{company}_sent.parquet")
    agg_sent_df = df.groupby("date").agg(
        analyzed_bpe_tokens=("text_input_tokens", lambda row: row.sum().astype(int)),
        weighted_sentiment=(
            "text_input_tokens",
            lambda row: weighted_avg(
                df.loc[row.index], "text_sentiment", "text_input_tokens"
            ),
        ),
        weighted_error=(
            "text_input_tokens",
            lambda row: weighted_avg(
                df.loc[row.index], "text_error", "text_input_tokens"
            ),
        ),
        analyzed_naive_tokens=("tokens", "sum"),
        daily_naive_token_sum=("daily_token_sum", "first"),
        analyzed_article_count=("index", "count"),
        daily_article_count=("daily_article_count", "first"),
    )
    agg_sent_df.reset_index(inplace=True)
    agg_sent_df["date"] = pd.to_datetime(agg_sent_df.date)
    agg_sent_df.sort_values("date", inplace=True)  # probably redundant
    agg_sent_df.set_index("date", inplace=True)

    for window in [3, 7, 14]:
        agg_sent_df[f"sentiment_{window}d_rolling_mean"] = (
            agg_sent_df.weighted_sentiment.rolling(window).mean()
        )

    return agg_sent_df
