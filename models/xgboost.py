"""
# XGBoost Model

Vi Mai

This module trains an XGBoost classifier on each of our target companies.
It returns the model and various metrics.
"""

from utils.load_data import load_data
from utils.split_data import split_data

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


def xgboost_classifier(company):
    df = load_data(company)
    # df = df.dropna()
    X_train, X_test, y_train, y_test = split_data(df, business_days=True)
    analyzer = XGBoostAnalyzer()

    analyzer.fit(X_train, y_train)
    y_pred = analyzer.pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    f1score = f1_score(y_test, y_pred)

    # Confusion
    cm = confusion_matrix(y_test, y_pred)

    xgboost_probs = analyzer.classifier.predict_proba(X_test)[
        :, 1
    ]  # Probabilities for class 1
    roc_auc = roc_auc_score(y_test, xgboost_probs)

    return analyzer.classifier, cm, accuracy, f1score, xgboost_probs, roc_auc, y_test


class XGBoostAnalyzer:
    def __init__(self):
        self.classifier = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            eval_metric="logloss",
        )

    def fit(self, X_train, y_train):
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), X_train.columns),
            ]
        )
        # Create the pipeline with preprocessor and classifier
        self.pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", self.classifier),
            ]
        )
        self.pipeline.fit(X_train, y_train)
