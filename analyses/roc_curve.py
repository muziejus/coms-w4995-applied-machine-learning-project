"""
# ROC Curve

Ruibin Lyu

"""

from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt


def plot_roc_curves(results_df, companies):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    for i, model in enumerate(["LSTM", "SVM", "Random Forest", "XGBoost"]):
        ax = axes[i]
        for company, color in zip(
            companies, ["blue", "orange", "green", "red", "purple"]
        ):
            company_results = results_df[
                (results_df["model_name"] == model) & (results_df["company"] == company)
            ]
            company_results = company_results.iloc[0]

            fpr, tpr, _ = roc_curve(
                company_results["y_test"].ravel(), company_results["probs"].ravel()
            )

            ax.plot(fpr, tpr, color=color, lw=2, label=f"{company}")
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend(loc="lower right")
        ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        ax.set_title(model)
    plt.suptitle("ROC Curves for Top Companies across Four Machine Learning Techniques")
    path = "plots/roc_curves.png"
    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )
