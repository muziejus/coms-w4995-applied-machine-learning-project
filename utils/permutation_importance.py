"""
# Permutation permutation_importance

Vi Mai

This utility is used to evaluate the importance of features in a dataset.
It is a model-agnostic method. 
"""

from sklearn import inspection
import pandas as pd


def permutation_importance(model, X, y):
    if X.ndim == 3:
        X = X.reshape(X.shape[0], -1)

    def model_predict(X):
        return model.predict(X).flatten()

    result = inspection.permutation_importance(
        model, X, y, n_repeats=10, random_state=42, n_jobs=-1
    )
    feature_names = list(X.columns)
    importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": result.importances_mean}
    ).sort_values(by="Importance", ascending=False)

    return importance_df
