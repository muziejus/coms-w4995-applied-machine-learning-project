"""
# Split data into training and testing sets

Tupa Opel

This function splits the data into 80/20 training and testing sets using a
structured split.

If numerical_target = True, it retrieves the target_price column.
Otherwise, it retrieves the target column.

If business_days = True, it drops every row for which there was no close
price (i.e. weekends and holidays).
"""

import pandas as pd
import numpy as np


def split_data(df, business_days=False, numerical_target=False):
    if business_days:
        df = df.dropna(subset=["close"])

    # Drop target columns for X
    X = df.drop(["target", "target_price"], axis=1)

    if numerical_target:
        y = df["target_price"]
    else:
        y = df["target"]

    # Structured Train Test Split
    X_dev, X_test = np.split(X, [int(0.8 * len(X))])
    y_dev, y_test = np.split(y, [int(0.8 * len(y))])

    return X_dev, X_test, y_dev, y_test
