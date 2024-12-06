"""
# Using Sentiment Analysis to Forecast Share Prices

by Ruibin Lyu, Vi Mai, Julie Meunier, Tuba Opel, Moacir P. de Sá Pereira
"""

import sys
import pickle

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from external_indicators import get_external_indicators
from share_prices import get_share_prices
from sentiment import get_daily_sentiment
from expand_financial_data import expand_financial_data
from eda import eda
from models.random_forest_classifier import random_forest_classifier
from models.svm import svm
from models.lstm import lstm_classifier

companies = ["dltr", "lulu", "ulta", "wba", "wmt"]

start_date = "2019-01-01"
end_date = "2024-10-16"
date_range = pd.date_range(start=start_date, end=end_date, freq="D")

rerun = False


def collect_data():
    # Create a dictionary to store the data
    data = {}

    # Collect external indicators
    data["external_indicators"] = get_external_indicators(
        start_date, end_date, date_range
    )

    # Collect share prices and sentiment analyses for our companies.
    # Then merge the data with the external indicators.
    # Save the merged data to a parquet file.
    for company in companies:
        data[company] = get_share_prices(company, start_date, end_date)
        data[f"{company}_sent"] = get_daily_sentiment(company)
        data[company] = expand_financial_data(data[company])

        blank_df = pd.DataFrame(date_range, columns=["date"])
        blank_df.set_index("date", inplace=True)
        merged_df = blank_df.merge(
            data[f"{company}_sent"], how="left", left_index=True, right_index=True
        )
        merged_df = merged_df.merge(
            data["external_indicators"], how="left", left_index=True, right_index=True
        )
        merged_df = merged_df.merge(
            data[company], how="left", left_index=True, right_index=True
        )
        fill_na = {
            "analyzed_bpe_tokens": 0,
            "weighted_sentiment": 0,
            "weighted_error": 0,
            "analyzed_naive_tokens": 0,
            "daily_naive_token_sum": 0,
            "analyzed_article_count": 0,
            "daily_article_count": 0,
        }
        merged_df.fillna(fill_na, inplace=True)
        merged_df.to_parquet(f"data/{company}_merged_data.parquet")
        data[company] = merged_df


# Exploratory Data Analysis
def print_eda():
    # Load the data
    eda_df = eda(companies)
    print(eda_df)


# LSTM model
def lstm_model():
    if rerun:
        results = []
        for company in companies:
            cm, accuracy, f1score = lstm_classifier(company)
            results.append((company, cm, accuracy, f1score))
        with open("models/lstm_results.pkl", "wb") as f:
            pickle.dump(results, f)


# SVM Model
def svm_model():
    if rerun:
        results = []
        for company in companies:
            cm, accuracy, f1score = svm(company)
            results.append((company, cm, accuracy, f1score))
        with open("models/svm_results.pkl", "wb") as f:
            pickle.dump(results, f)


# Random Forest model
def random_forest():
    if rerun:
        results = []
        for company in companies:
            cm, accuracy, f1score = random_forest_classifier(company)
            results.append((company, cm, accuracy, f1score))
        with open("models/random_forest_results.pkl", "wb") as f:
            pickle.dump(results, f)
    else:
        with open("models/random_forest_results.pkl", "rb") as f:
            results = pickle.load(f)
            print(results[0])


if __name__ == "__main__":
    # Check if an argument is passed
    if len(sys.argv) > 2 and sys.argv[2] == "rerun":
        rerun = True
    if len(sys.argv) > 1:
        argument = sys.argv[1]
        if argument == "collect_data":
            collect_data()
        elif argument == "eda":
            print_eda()
        elif argument == "random_forest":
            random_forest()
        elif argument == "svm":
            svm_model()
        elif argument == "lstm":
            lstm_model()
        else:
            print("Invalid argument. Please use 'collect_data'.")
    else:
        print("No argument provided. Please use 'collect_data'.")
