"""
# Random Forest Classifier

Tuba Opel

This module trains a random forest classifier on each of our target
companies. It returns the data to make a confusion matrix and a
feature importance plot.
"""

from utils.load_data import load_data
from utils.split_data import split_data

import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
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

    print(f"\n\n{company.upper()} model selection time: {elapsed_time} seconds")
    print(f"Best F1 score: {model_grid_search.best_score_}")
    print(f"Best params: {model_grid_search.best_params_}")
    print(f"Test F1 score: {model_grid_search.score(X_test, y_test)}")

    tuned_model = model_grid_search.best_estimator_
    y_pred = tuned_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    f1score = f1_score(y_test, y_pred)

    # Confusion
    cm = confusion_matrix(y_test, y_pred)

    # Feature Importance
    # feat_imps = zip(
    #     X_dev.columns, model_grid_search.best_estimator_.feature_importances_
    # )
    # feats, imps = zip(
    #     *(
    #         sorted(
    #             list(filter(lambda x: x[1] != 0, feat_imps)),
    #             key=lambda x: x[1],
    #             reverse=True,
    #         )
    #     )
    # )

    return cm, accuracy, f1score
    # return cm, feats, imps, accuracy, f1score
