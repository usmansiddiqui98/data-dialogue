import os

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.models.sentiment_analysis.log_reg import LogReg
from sklearn.metrics import accuracy_score

@pytest.fixture
def model():
    # Load and return the trained model here
    return LogReg(models_path="/test_files")

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


def test_fit(model,get_data):
    X_train, y_train, _, _ = get_data
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

def test_predict(model,get_data):
    X_train, y_train, X_test, y_test = get_data
    model.fit(X_train, y_train)
    result = model.predict(X_test)
    assert isinstance(result, dict)
    assert set(result.keys()) == {"predicted_sentiment", "predicted_sentiment_probability"}
    assert len(result["predicted_sentiment"]) == len(X_test)
    assert len(result["predicted_sentiment_probability"]) == len(X_test)
    assert all(isinstance(x, float) or isinstance(x, int) for x in result["predicted_sentiment_probability"])

def test_accuracy(model,get_data):
    X_train, y_train, X_test, y_test = get_data
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)["predicted_sentiment"]
    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy >= 0.7
