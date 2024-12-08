"""
# SVM Model

Vi Mai

This module trains a support vector machine (SVM) on each of our target
companies. It returns the model and various metrics.
"""

from utils.load_data import load_data
from utils.permutation_importance import permutation_importance
from utils.split_data import split_data

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# import shap


def svm_classifier(company):
    df = load_data(company)
    df = df.dropna()
    X_train, X_test, y_train, y_test = split_data(df)

    analyzer = SVMAnalyzer()
    analyzer.fit(X_train, y_train)
    y_pred = analyzer.pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    f1score = f1_score(y_test, y_pred)

    # Confusion
    cm = confusion_matrix(y_test, y_pred)

    svm_probs = analyzer.classifier.predict_proba(X_test)[
        :, 1
    ]  # Probabilities for class 1
    roc_auc = roc_auc_score(y_test, svm_probs)

    perm_imp_df = permutation_importance(analyzer.pipeline, X_test, y_test)

    return (
        analyzer.classifier,
        cm,
        accuracy,
        f1score,
        svm_probs,
        roc_auc,
        y_test,
        perm_imp_df,
    )


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
