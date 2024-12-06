"""
# Build confusion matrix and feature importance plot

Vi Mai, Tuba Opel, Moacir P. de Sá Pereira

This module aggregates our team’s work to produce a generalized set of ten
subplots featuring confusion matrices and feature importances for each of
our target companies.
"""

import matplotlib.pyplot as plt
import seaborn as sns


def build_confusion_and_feature_importance_plots(results, technique, feature_count=20):
    company_count = len(results)
    fig, axes = plt.subplots(2, company_count, figsize=(18, 7), constrained_layout=True)
    for i, result in enumerate(results):
        company, confusion_matrix, features, importances, accuracy = result
        features = features[:feature_count]
        importances = importances[:feature_count]
        ax0 = sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Sell (0)", "Buy (1)"],
            yticklabels=["Sell (0)", "Buy (1)"],
            ax=axes[0, i],
        )
        ax0.set_xlabel("Predicted Label")
        ax0.set_ylabel("True Label")
        ax0.set_title(f"{company.upper()} (Accuracy: {accuracy*100:.2f}%)")

        ax1 = sns.barplot(y=list(importances), x=list(features), ax=axes[1, i])
        ax1.tick_params(axis="x", rotation=90)
        ax1.set_xticklabels(ax1.get_xticklabels(), fontsize=6)
        ax1.set_ylabel("Importance")
        ax1.set_xlabel("Features")

    plt.suptitle(
        f"Confusion Matrices and Top 20 Feature Importances Using {technique}"  # , fontsize=16
    )
    path = f"plots/{technique.lower().replace(' ', '_')}_faceted_confusion_matrices_and_feature_importances.png"
    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    print(f"Faceted confusion matrices and feature importances saved to {path}.")
