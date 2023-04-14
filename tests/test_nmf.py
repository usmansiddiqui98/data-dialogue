import os

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from src.models.topic_modelling.NMF import NMFModel


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


def test_nmf_model(get_data):
    X_train, y_train, X_test, y_test = get_data
    # Initialize NMFModel with sample parameters
    nmf_model = NMFModel(X_train, n_components=2, max_features=100, max_df=0.8)

    # Test fit_transform() method
    nmf_model.fit_transform()
    assert isinstance(nmf_model.vectorizer, TfidfVectorizer)
    assert isinstance(nmf_model.model, NMF)
    assert nmf_model.X is not None
    assert isinstance(nmf_model.document_weights, np.ndarray)

    # Test get_topic_terms() method
    topics_dict = nmf_model.get_topic_terms(n_top_words=5)
    assert isinstance(topics_dict, dict)
    assert len(topics_dict) == 2  # n_components = 2
    assert all(isinstance(val, dict) for val in topics_dict.values())

    # Test get_labels() method
    label_df = nmf_model.get_labels(n_top_words=3)
    assert isinstance(label_df, pd.DataFrame)
    assert "Topic_idx" in label_df.columns
    assert "Top_Topic_Terms" in label_df.columns
    assert len(label_df) == len(X_train)
