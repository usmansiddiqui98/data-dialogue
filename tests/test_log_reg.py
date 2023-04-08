import os

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.models.sentiment_analysis.log_reg import LogReg


@pytest.fixture
def get_data():
    train_fname = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "test_files", "train_reviews.csv"))
    test_fname = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "test_files", "test_reviews.csv"))

    train_df = pd.read_csv(train_fname).head(100)
    test_df = pd.read_csv(test_fname).head(100)
    X_train = train_df.drop(["sentiment"], axis=1)
    y_train = train_df["sentiment"]
    X_test = test_df.drop(["sentiment"], axis=1)
    y_test = test_df["sentiment"]
    return X_train, y_train, X_test, y_test


def test_fit(get_data):
    X_train, y_train, _, _ = get_data
    model = LogReg(models_path="/test_files")
    model.fit(X_train, y_train)
    assert isinstance(model.model, LogisticRegression)
    assert model.model.get_params() == {
        "C": 1.0,
        "class_weight": None,
        "dual": False,
        "fit_intercept": True,
        "intercept_scaling": 1,
        "l1_ratio": None,
        "max_iter": 1000,
        "multi_class": "auto",
        "n_jobs": None,
        "penalty": "l2",
        "random_state": 4265,
        "solver": "lbfgs",
        "tol": 0.0001,
        "verbose": 0,
        "warm_start": False,
    }
