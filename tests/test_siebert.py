import os

import pandas as pd
import pytest
from sklearn.metrics import accuracy_score

from src.models.sentiment_analysis.pre_trained.siebert import Siebert


@pytest.fixture
def model():
    return Siebert(models_path="/test_files")


@pytest.fixture
def get_data():
    test_fname = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "test_files", "test_reviews.csv"))

    test_df = pd.read_csv(test_fname).head(100)
    X_test = test_df.drop(["sentiment"], axis=1)
    y_test = test_df["sentiment"]
    return X_test, y_test


def test_predict(model, get_data):
    X_test, _ = get_data
    result = model.predict(X_test)
    assert isinstance(result, dict)
    assert set(result.keys()) == {"predicted_sentiment", "predicted_sentiment_probability"}
    assert len(result["predicted_sentiment"]) == len(X_test)
    assert len(result["predicted_sentiment_probability"]) == len(X_test)
    assert all(isinstance(x, float) or isinstance(x, int) for x in result["predicted_sentiment_probability"])
def test_accuracy(model, get_data):
    X_test, y_test = get_data
    y_pred = model.predict(X_test)["predicted_sentiment"]
    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy >= 0.7