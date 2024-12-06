"""
# Using Sentiment Analysis to Forecast Share Prices

by Ruibin Lyu, Vi Mai, Julie Meunier, Tuba Opel, Moacir P. de Sá Pereira
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from external_indicators import get_external_indicators
from share_prices import get_share_prices
from sentiment import get_daily_sentiment
from expand_financial_data import expand_financial_data
from eda import eda
from models.random_forest_classifier import random_forest_classifier

companies = ["dltr", "lulu", "ulta", "wba", "wmt"]

start_date = "2019-01-01"
end_date = "2024-10-16"
date_range = pd.date_range(start=start_date, end=end_date, freq="D")


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


# SVM Model


# Random Forest model
def random_forest():
    results = []
    for company in companies:
        confusion_matrix_display, features, importances = random_forest_classifier(
            company
        )
        results.append((company, confusion_matrix_display, features, importances))

    # Plot using subplots
    fig, axes = plt.subplots(2, 5, figsize=(14, 8), constrained_layout=True)
    plt.tight_layout()

    for i, (company, confusion_matrix_display, features, importances) in enumerate(
        results
    ):
        confusion_matrix_display.plot(ax=axes[0, i])
        confusion_matrix_display.ax_.set_title(
            f"{company.upper()} Confusion Matrix", fontsize=12
        )
        confusion_matrix_display.im_.colorbar.remove()
        confusion_matrix_display.ax_.set_xlabel("")
        # cm_ax.set_title()
        # cm_ax.set_xlabel("Predicted")
        # cm_ax.set_ylabel("Actual")

        # Feature Importance subplot (bottom row)
        # sorted_feats_imps = sorted(zip(feats, imps), key=lambda x: x[1], reverse=True)
        # top_feats, top_imps = zip(*sorted_feats_imps[:10])  # Show top 10 features
        # wrapped_feats = ["\n".join(textwrap.wrap(f, width=15)) for f in top_feats]
        ax = sns.barplot(y=list(importances), x=list(features), ax=axes[1, i])
        # ax.set_titleax.tick_params(axis="x", rotation=45)  # Rotate x-axis labels by 45 degrees
        ax.tick_params(axis="x", rotation=90)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
        #     f"{company.upper()} Feature Selection Based on Random Forest Classification"
        # )
        ax.set_ylabel("Importance")
        ax.set_xlabel("Features")

    plt.suptitle(
        "Confusion Matrices and Feature Importances for Companies", fontsize=16
    )
    plt.savefig(
        "plots/random_forest/faceted_random_forest.png",
        dpi=300,
        bbox_inches="tight",
    )


# plt.savefig(
#     f"plots/random_forest/{company}_confusion_matrix.png",
#     dpi=300,
#     bbox_inches="tight",
# )
# plt.figure(figsize=(12, 10))
# plt.tight_layout()
# ax = sns.barplot(x=list(imps), y=list(feats))
# ax.set_title(
#     f"{company.upper()} Feature Selection Based on Random Forest Classification"
# )
# ax.set_xlabel("Importance")
# ax.set_ylabel("Features")
# plt.savefig(
#     f"plots/random_forest/{company}_feature_selection.png",
#     dpi=300,
#     bbox_inches="tight",
# )


if __name__ == "__main__":
    # Check if an argument is passed
    if len(sys.argv) > 1:
        argument = sys.argv[1]
        if argument == "collect_data":
            collect_data()
        elif argument == "eda":
            print_eda()
        elif argument == "random_forest":
            random_forest()
        else:
            print("Invalid argument. Please use 'collect_data'.")
    else:
        print("No argument provided. Please use 'collect_data'.")
