"""
# Random Forest Classifier

Tuba Opel

This module trains a random forest classifier on each of our target
companies.
"""

from utils.load_data import load_data
from utils.split_data import split_data

import seaborn as sns
import matplotlib.pyplot as plt
import time

from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import GridSearchCV


def random_forest_classifier(company):
    df = load_data(company)

    # Add Index as a column for ordinal encoding of days
    df.insert(0, "Index", range(len(df)))

    X_dev, X_test, y_dev, y_test = split_data(df, business_days=True)

    clf = RandomForestClassifier(random_state=42)

    # Hyper Parameter Tuning
    param_grid = {
        "n_estimators": [10, 50, 100, 200, 400],
        "max_depth": [5, 10, 20, 30, 50],
    }

    start_time = time.time()
    model_grid_search = GridSearchCV(
        clf, param_grid=param_grid, cv=5, scoring="f1_micro"
    )

    model_grid_search.fit(X_dev, y_dev)
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Evaluate Model
    print(f"Model Selection time: {elapsed_time} seconds")
    print(f"Best F1 score: {model_grid_search.best_score_}")
    print(f"Best params: {model_grid_search.best_params_}")
    print(f"Test F1 score: {model_grid_search.score(X_test, y_test)}")

    tuned_model = model_grid_search.best_estimator_
    y_pred = tuned_model.predict(X_test)

    # [0, 1]
    class_names = ["Sell", "Buy"]

    # Confusion
    cm = metrics.confusion_matrix(y_test, y_pred)
    disp = metrics.ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=class_names
    )
    disp.plot()
    plt.title(f"{company.upper()} Confusion Matrix")
    plt.show()

    # Feature Importance
    feat_imps = zip(
        X_dev.columns, model_grid_search.best_estimator_.feature_importances_
    )
    feats, imps = zip(
        *(
            sorted(
                list(filter(lambda x: x[1] != 0, feat_imps)),
                key=lambda x: x[1],
                reverse=True,
            )
        )
    )
    ax = sns.barplot(x=list(feats), y=list(imps))
    ax.set_title(f"{company.upper()} Feature Selection Based on RFR")
    ax.set_xlabel("Features")
    ax.set_ylabel("Importance")
