import os
import pickle

import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier

from src.models.sentiment_analysis.base_model import BaseModel


class XgBoostSvd(BaseModel):
    """
    XgBoostSvd is a class that implements sentiment analysis using XGBoost and TruncatedSVD dimensionality reduction.

    Attributes:
        vectorizer (TfidfVectorizer): TfidfVectorizer object for text vectorization.
        dim_reduce (TruncatedSVD): TruncatedSVD object for dimensionality reduction.
        tuned_parameters (dict): Dictionary of hyperparameters for XGBoost classifier.
        model (XGBClassifier): XGBoost classifier object.

    Methods:
        fit(X_train, y_train): Fit the XGBoostSvd model to the training data.
        save(model_name): Save the trained model, vectorizer, and dim_reduce objects to pickle files.
        load(model_name): Load the trained model, vectorizer, and dim_reduce objects from pickle files.
        predict(X_test): Predict sentiment labels and probabilities for the given test data.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the XgBoostSvd model.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.vectorizer = TfidfVectorizer()
        self.dim_reduce = TruncatedSVD(n_components=100, random_state=4265)
        self.tuned_parameters = {
            "colsample_bytree": 0.7,
            "learning_rate": 0.1,
            "max_depth": 6,
            "min_child_weight": 1,
            "n_estimators": 500,
            "subsample": 0.5,
        }
        self.model = XGBClassifier(eval_metric="mlogloss", **self.tuned_parameters, random_state=4265)

    def fit(self, X_train, y_train):
        """
        Fit the XgBoostSvd model to the training data.

        Args:
            X_train (pd.DataFrame): Training data.
            y_train (pd.Series): Target labels.

        Returns:
            None
        """
        vectorizer = self.vectorizer
        dim_reduce = self.dim_reduce
        X_train.reset_index(drop=True, inplace=True)
        X_train_tfidf = vectorizer.fit_transform(X_train["cleaned_text"])
        X_train_tfidf = pd.DataFrame(X_train_tfidf.toarray(), columns=vectorizer.get_feature_names_out())
        X_train_svd = dim_reduce.fit_transform(X_train_tfidf)
        X_train_svd = pd.DataFrame(X_train_svd, columns=[f"svd_{i}" for i in range(100)])
        X_train_clean = X_train.drop(["cleaned_text", "text"], axis=1)
        X_train_concat = pd.concat([X_train_clean, X_train_svd], axis=1)
        X_train_concat = X_train_concat.loc[:, ~X_train_concat.columns.duplicated()].copy()

        self.model.fit(X_train_concat, y_train)

    def save(self, model_name):
        """
        Save the trained model, vectorizer, and dim_reduce objects to pickle files.

        Args:
            model_name (str): Name of the model to be saved.

        Returns:
            None
        """

        self.model_dir = os.path.join(self.models_path, model_name)

        with open(os.path.join(self.model_dir, "model.pkl"), "wb") as f:
            pickle.dump(self.model, f)
        with open(os.path.join(self.model_dir, "vectorizer.pkl"), "wb") as f:
            pickle.dump(self.vectorizer, f)
        with open(os.path.join(self.model_dir, "dim_reduce.pkl"), "wb") as f:
            pickle.dump(self.dim_reduce, f)

    def load(self, model_name):
        """
        Load a trained model, vectorizer, and dimensionality reduction model from disk.

        Args:
            model_name (str): The name of the model to load.

        Returns:
            None
        """
        self.model_dir = os.path.join(self.models_path, model_name)

        with open(os.path.join(self.model_dir, "model.pkl"), "rb") as f:
            self.model = pickle.load(f)
        with open(os.path.join(self.model_dir, "vectorizer.pkl"), "rb") as f:
            self.vectorizer = pickle.load(f)
        with open(os.path.join(self.model_dir, "dim_reduce.pkl"), "rb") as f:
            self.dim_reduce = pickle.load(f)

    def predict(self, X_test):
        """
        Perform sentiment classification on test data.

        Args:
            X_test (DataFrame): Test data as a pandas DataFrame.

        Returns:
            dict: A dictionary containing predicted sentiment labels and their corresponding probabilities.
                - predicted_sentiment (list): List of predicted sentiment labels (0 for negative, 1 for positive).
                - predicted_sentiment_probability (list): List of predicted sentiment probabilities.
        """
        vectorizer = self.vectorizer
        dim_reduce = self.dim_reduce
        X_test.reset_index(drop=True, inplace=True)
        X_test_tfidf = vectorizer.transform(X_test["cleaned_text"])
        X_test_tfidf = pd.DataFrame(X_test_tfidf.toarray(), columns=vectorizer.get_feature_names_out())
        X_test_svd = dim_reduce.transform(X_test_tfidf)
        X_test_svd = pd.DataFrame(X_test_svd, columns=[f"svd_{i}" for i in range(100)])
        X_test_clean = X_test.drop(["cleaned_text", "text"], axis=1)
        X_test_concat = pd.concat([X_test_clean, X_test_svd], axis=1)
        X_test_concat = X_test_concat.loc[:, ~X_test_concat.columns.duplicated()].copy()
        probability = self.model.predict_proba(X_test_concat)[:, 0].tolist()  # probability of negative sentiment
        results = [0 if prob > 0.5 else 1 for prob in probability]
        probability = [prob if prob > 0.5 else 1 - prob for prob in probability]

        return {"predicted_sentiment": results, "predicted_sentiment_probability": probability}
