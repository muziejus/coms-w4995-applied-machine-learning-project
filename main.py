"""
# Using Sentiment Analysis to Forecast Share Prices

by Ruibin Lyu, Vi Mai, Julie Meunier, Tuba Opel, Moacir P. de Sá Pereira
"""

import sys
import pickle

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing.external_indicators import get_external_indicators
from preprocessing.share_prices import get_share_prices
from preprocessing.sentiment import get_daily_sentiment
from preprocessing.expand_financial_data import expand_financial_data
from eda import eda
from models.random_forest import random_forest_classifier
from models.svm import svm_classifier
from models.lstm import lstm_classifier
from models.xgboost import xgboost_classifier

from utils.load_data import load_data

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


def rerun_model(company, model_function):
    model, cm, accuracy, f1score = model_function(company)
    return {
        "company": company,
        "model": model,
        "cm": cm,
        "accuracy": accuracy,
        "f1score": f1score,
    }


# LSTM model
def lstm():
    if rerun:
        print("Rerunning LSTM model")
        results = []
        for company in companies:
            results.append(rerun_model(company, lstm_classifier))
        with open("data/model_metadata/lstm_results.pkl", "wb") as f:
            pickle.dump(results, f)


# SVM Model
def svm():
    if rerun:
        print("Rerunning SVM model")
        results = []
        for company in companies:
            results.append(rerun_model(company, svm_classifier))
        with open("data/model_metadata/svm_results.pkl", "wb") as f:
            pickle.dump(results, f)


# Random Forest model
def random_forest():
    if rerun:
        print("Rerunning Random Forest model")
        results = []
        for company in companies:
            results.append(rerun_model(company, random_forest_classifier))
        with open("data/model_metadata/random_forest_results.pkl", "wb") as f:
            pickle.dump(results, f)
    else:
        with open("data/model_metadata/random_forest_results.pkl", "rb") as f:
            results = pickle.load(f)
            df = load_data("dltr")  # Load arbitrary dataset to get column names
            df.insert(0, "Index", range(len(df)))  # Add index column
            columns = df.columns
            dfs = []
            for result in results:
                feat_imps = zip(columns, result["model"].feature_importances_)
                feat_imp_df = pd.DataFrame(feat_imps, columns=["feature", "importance"])
                feat_imp_df["company"] = result["company"]
                feat_imp_df["rank"] = feat_imp_df["importance"].rank(ascending=False)
                dfs.append(feat_imp_df)
            feat_imp_df = pd.concat(dfs)
            avg_feat_imp_df = (
                feat_imp_df.groupby("feature")["rank"].mean().reset_index()
            )
            avg_feat_imp_df = avg_feat_imp_df.sort_values("rank")
            print(avg_feat_imp_df.head(10))
            fig, axes = plt.subplots(1, 1, figsize=(10, 10))
            ax = sns.barplot(data=avg_feat_imp_df, x="rank", y="feature", ax=axes)
            ax.set_title(
                "Average Ranks of Feature Importance for Random Forest Classification"
            )
            ax.set_xlabel("Average Rank")
            ax.set_ylabel("Feature")
            path = "plots/average_rank_feature_importance.png"
            plt.savefig(
                path,
                dpi=300,
                bbox_inches="tight",
            )


# XGBoost model
def xgboost():
    if rerun:
        print("Rerunning XGBoost model")
        results = []
        for company in companies:
            results.append(rerun_model(company, xgboost_classifier))
        with open("data/model_metadata/xgboost_results.pkl", "wb") as f:
            pickle.dump(results, f)


# Compile report on all models
def report():
    with open("data/model_metadata/lstm_results.pkl", "rb") as f:
        lstm_results = pickle.load(f)
        for company in lstm_results:
            company["model_name"] = "LSTM"
    with open("data/model_metadata/svm_results.pkl", "rb") as f:
        svm_results = pickle.load(f)
        for company in svm_results:
            company["model_name"] = "SVM"
    with open("data/model_metadata/random_forest_results.pkl", "rb") as f:
        random_forest_results = pickle.load(f)
        for company in random_forest_results:
            company["model_name"] = "Random Forest"
    with open("data/model_metadata/xgboost_results.pkl", "rb") as f:
        xgboost_results = pickle.load(f)
        for company in xgboost_results:
            company["model_name"] = "XGBoost"

    results = lstm_results + svm_results + random_forest_results + xgboost_results
    results_df = pd.DataFrame(results)
    print(results_df)


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
            svm()
        elif argument == "lstm":
            lstm()
        elif argument == "xgboost":
            xgboost()
        elif argument == "report":
            report()
        else:
            print("Invalid argument. Please use 'collect_data'.")
    else:
        print("No argument provided. Please use 'collect_data'.")
