import os

import pandas as pd
import pytest

from src.models.topic_modelling.LDA import LDAGensim


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


def test_get_topics(get_data):
    X_train, y_train, X_test, y_test = get_data
    num_topics = 10
    # Create instance of LSAModel
    model = LDAGensim(X_train, num_topics=num_topics)
    # Call get_topics() method
    topics = model.get_topics()
    # Assert that the returned result is a dictionary with expected keys
    assert isinstance(topics, dict)
    assert len(topics) == num_topics
