import os

import pandas as pd
import pytest
import torch
from sklearn.metrics import accuracy_score

from src.models.sentiment_analysis.pre_trained.bert_fine_tuned import BertFineTuned


@pytest.fixture
def model():
    # Load and return the trained model here
    return BertFineTuned(
        models_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "sentiment_analysis"))
    )


@pytest.fixture
def get_data():
    test_fname = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "test_files", "test_reviews.csv"))

    test_df = pd.read_csv(test_fname).head(100)
    X_test = test_df.drop(["sentiment"], axis=1)
    y_test = test_df["sentiment"]
    return X_test, y_test


def test_predict(model, get_data):
    X_test, _ = get_data
    output = model.predict(X_test)
    assert isinstance(output, dict)
    assert "predicted_sentiment" in output
    assert "predicted_sentiment_probability" in output

    predicted_sentiment = output["predicted_sentiment"]
    predicted_sentiment_probability = output["predicted_sentiment_probability"]

    assert isinstance(predicted_sentiment, torch.Tensor)
    assert isinstance(predicted_sentiment_probability, torch.Tensor)

    assert len(predicted_sentiment) == len(predicted_sentiment_probability)
    assert all(x in [0, 1] for x in predicted_sentiment.tolist())

def test_accuracy(model, get_data):
    X_test, y_test = get_data
    y_pred = model.predict(X_test)["predicted_sentiment"]
    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy >= 0.7
