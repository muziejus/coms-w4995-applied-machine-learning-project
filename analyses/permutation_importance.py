"""
# Permutation Importance

Moacir P. de Sá Pereira

This module contains a function that plots the top ten features by permutation importance for all four models.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def permutation_importance(results_df):
    perm_imps = {
        "LSTM": [],
        "SVM": [],
        "Random Forest": [],
        "XGBoost": [],
    }
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    for i, row in results_df.iterrows():
        perm_imps[row["model_name"]].append(row["perm_imp_df"])
    for i, model in enumerate(perm_imps.keys()):
        perm_imp_df = pd.concat(perm_imps[model])
        avg_perm_imp_df = (
            perm_imp_df.groupby("Feature")["Importance"].mean().reset_index()
        )
        avg_perm_imp_df = avg_perm_imp_df.sort_values("Importance", ascending=False)
        ax = sns.barplot(
            data=avg_perm_imp_df.head(10), y="Importance", x="Feature", ax=axes[i]
        )
        ax.set_title(model)
        ax.set_ylabel("Average Importance")
        ax.set_xlabel("Feature")
        ax.tick_params(axis="x", rotation=90)
    plt.suptitle(
        "Top Average Permutation Importance across Four Machine Learning Techniques"
    )
    path = "plots/average_permutation_importance.png"
    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )
    print(f"Saved permutation importance plot to {path}.")
