"""
# SVM Model

Vi Mai

This module trains a support vector machine (SVM) on each of our target
companies. It returns the data to make a confusion matrix and a feature
importance plot.
"""

from utils.load_data import load_data
from utils.split_data import split_data

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import shap


def svm(company):
    df = load_data(company)
    df = df.dropna()
    X_train, X_test, y_train, y_test = split_data(df)

    analyzer = SVMAnalyzer()
    analyzer.fit(X_train, y_train)
    y_pred = analyzer.pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    # Confusion
    cm = confusion_matrix(y_test, y_pred)

    # Feature Importance via SHAP
    background_kmeans = shap.kmeans(X_train, 50)
    explainer = shap.KernelExplainer(
        analyzer.classifier.predict_log_proba, background_kmeans
    )
    shap_values = explainer.shap_values(X_test)
    shap_mean_importance = abs(shap_values[1]).mean(
        axis=0
    )  # For class 1 (binary classification)
    shap_feat_imps = zip(X_test.columns, shap_mean_importance)
    feats, imps = zip(
        *sorted(
            ((f, imp) for f, imp in shap_feat_imps if imp != 0),
            key=lambda x: x[1],
            reverse=True,
        )
    )

    return cm, feats, imps, accuracy


class SVMAnalyzer:
    def __init__(self):
        self.classifier = SVC(kernel="rbf", random_state=42, probability=True)

    def fit(self, X_train, y_train):
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), X_train.columns),
            ]
        )
        self.pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", self.classifier),
            ]
        )
        self.pipeline.fit(X_train, y_train)
