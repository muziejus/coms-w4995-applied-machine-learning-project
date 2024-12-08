"""
# LSTM Classifier

Tuba Opel

This module trains an LSTM classifier on each of our target companies.
It returns the data to make a confusion matrix.
"""

from utils.load_data import load_data
from utils.split_data import split_data

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV

from keras.models import Sequential
from keras.layers import LSTM, Dense

from scikeras.wrappers import KerasClassifier


def lstm_classifier(company):
    df = load_data(company)

    # Add Index as a column for ordinal encoding of days
    df.insert(0, "Index", range(len(df)))

    X_dev, X_test, y_dev, y_test = split_data(df, business_days=True)

    df = df.dropna()

    # drop target columns for X
    X = df.drop(["target", "target_price"], axis=1)
    columns = X.columns

    # Label Target
    y = df["target"]

    # Reshape data for LSTM input (samples, timesteps, features)
    timesteps = 30  # Number of timesteps to look back
    features = X.shape[1]
    X = np.array([X[i : i + timesteps] for i in range(len(X) - timesteps)])
    y = y[timesteps:]

    # Structured Train Test Split
    X_dev, X_test = np.split(X, [int(0.8 * len(X))])
    y_dev, y_test = np.split(y, [int(0.8 * len(y))])

    # Build LSTM model
    def create_model():
        model = Sequential()
        model.add(LSTM(units=64, input_shape=(timesteps, features), seed=42))
        model.add(Dense(1, activation="sigmoid"))  # Binary classification
        # Compile the model
        model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        return model

    # Hyper Parameter Tuning
    param_grid = {"epochs": [10, 20], "batch_size": [16, 32]}
    clf = KerasClassifier(model=create_model, verbose=0, random_state=42)
    grid = GridSearchCV(estimator=clf, param_grid=param_grid, n_jobs=-1, cv=5)

    grid.fit(X_dev, y_dev)

    print(f"\n\n{company.upper()} model selection")
    print("Best Train Accuracy score:", grid.best_score_)
    print("Best Train params:", grid.best_params_)
    print("Test Accuracy score:", grid.score(X_test, y_test))

    tuned_model = grid.best_estimator_
    y_pred = tuned_model.predict(X_test)

    tuned_model.model_.save("data/model_metadata/lstm.h5")

    accuracy = accuracy_score(y_test, y_pred)

    f1score = f1_score(y_test, y_pred)

    # Confusion
    cm = confusion_matrix(y_test, y_pred)

    lstm_probs = y_pred.flatten()
    roc_auc = roc_auc_score(y_test, lstm_probs)

    return (
        "model not saveable. See lstm.h5",
        cm,
        accuracy,
        f1score,
        lstm_probs,
        roc_auc,
        y_test,
    )
