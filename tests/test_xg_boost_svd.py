import os
import pandas as pd
import pytest
from xgboost import XGBClassifier

from src.models.sentiment_analysis.xg_boost_svd import XgBoostSvd
from sklearn.metrics import accuracy_score

@pytest.fixture
def model():
    return XgBoostSvd(models_path="/test_files")

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
    assert isinstance(model.model, XGBClassifier)
    # assert model.model.get_params() == {
    #     "colsample_bytree": 0.7,
    #     "learning_rate": 0.1,
    #     "max_depth": 6,
    #     "min_child_weight": 1,
    #     "n_estimators": 500,
    #     "subsample": 0.5,
    #     "eval_metric": "mlogloss",
    #     "random_state": 4265
    # }


def test_predict(model,get_data):
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