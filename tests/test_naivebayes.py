import os

import pandas as pd
from sklearn import naive_bayes
from sklearn.metrics import accuracy_score

from src.models.sentiment_analysis.naive_bayes import Naivebayes


def model():
    # Load and return the trained model here
    return Naivebayes(models_path="/test_files")


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


def test_fit(model, get_data):
    X_train, y_train, _, _ = get_data
    model.fit(X_train, y_train)
    assert isinstance(model.model, naive_bayes.MultinomialNB)


def test_predict(model, get_data):
    X_train, y_train, X_test, y_test = get_data
    model.fit(X_train, y_train)
    result = model.predict(X_test)
    assert isinstance(result, dict)
    assert set(result.keys()) == {"predicted_sentiment", "predicted_sentiment_probability"}
    assert len(result["predicted_sentiment"]) == len(X_test)
    assert len(result["predicted_sentiment_probability"]) == len(X_test)
    assert all(isinstance(x, float) or isinstance(x, int) for x in result["predicted_sentiment_probability"])


def test_accuracy(model, get_data):
    X_train, y_train, X_test, y_test = get_data
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)["predicted_sentiment"]
    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy >= 0.7
