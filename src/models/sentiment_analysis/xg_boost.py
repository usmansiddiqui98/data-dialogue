import os
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier

from src.models.sentiment_analysis.base_model import BaseModel


class XgBoost(BaseModel):
    """
    XgBoost class for sentiment analysis using XGBoost classifier.

    Parameters:
    ----------
    *args : tuple
        Positional arguments to be passed to the parent class.
    **kwargs : dict
        Keyword arguments to be passed to the parent class.

    Attributes:
    ----------
    vectorizer : TfidfVectorizer
        TfidfVectorizer object for text vectorization.
    model : XGBClassifier
        XGBoost classifier object for sentiment analysis.

    Methods:
    -------
    fit(X_train, y_train)
        Fit the model to the training data.
    save(model_name)
        Save the trained model and vectorizer to pickle files.
    load(model_name)
        Load the trained model and vectorizer from pickle files.
    predict(X_test)
        Predict sentiment labels for test data.

    """

    def __init__(self, *args, **kwargs):
        """
        Initialize XgBoost model.

        Parameters:
        ----------
        *args : tuple
            Positional arguments to be passed to the parent class.
        **kwargs : dict
            Keyword arguments to be passed to the parent class.

        """
        super().__init__(*args, **kwargs)
        self.vectorizer = TfidfVectorizer()
        self.model = XGBClassifier(eval_metric="mlogloss", random_state=4265)

    def fit(self, X_train, y_train):
        """
        Fit the XgBoost model to the training data.

        Parameters:
        ----------
        X_train : DataFrame
            Training data containing text and label columns.
        y_train : Series or array-like
            Labels for the training data.

        """
        vectorizer = self.vectorizer
        X_train.reset_index(drop=True, inplace=True)
        X_train_tfidf = vectorizer.fit_transform(X_train["cleaned_text"])
        X_train_tfidf = pd.DataFrame(X_train_tfidf.toarray(), columns=vectorizer.get_feature_names_out())
        X_train_clean = X_train.drop(["cleaned_text", "text"], axis=1)
        X_train_concat = pd.concat([X_train_clean, X_train_tfidf], axis=1)
        X_train_concat = X_train_concat.loc[:, ~X_train_concat.columns.duplicated()].copy()

        self.model.fit(X_train_concat, y_train)

    def save(self, model_name):
        """
        Save the trained XgBoost model and vectorizer to pickle files.

        Parameters:
        ----------
        model_name : str
            Name of the model to be saved.

        """
        self.model_dir = os.path.join(self.models_path, model_name)

        with open(os.path.join(self.model_dir, "model.pkl"), "wb") as f:
            pickle.dump(self.model, f)
        with open(os.path.join(self.model_dir, "vectorizer.pkl"), "wb") as f:
            pickle.dump(self.vectorizer, f)

    def load(self, model_name):
        """
        Load the trained XgBoost model and vectorizer from pickle files.

        Parameters:
        ----------
        model_name : str
            Name of the model to be loaded.

        """
        self.model_dir = os.path.join(self.models_path, model_name)

        with open(os.path.join(self.model_dir, "model.pkl"), "rb") as f:
            self.model = pickle.load(f)
        with open(os.path.join(self.model_dir, "vectorizer.pkl"), "rb") as f:
            self.vectorizer = pickle.load(f)

    def predict(self, X_test):
        """
        Predict sentiment labels and probabilities for a given test dataset using the trained model.

        Parameters:
            X_test (pd.DataFrame): Test dataset containing features for prediction.

        Returns:
            dict: A dictionary containing predicted sentiment labels and their corresponding probabilities.
                {
                    "predicted_sentiment": list of int: List of predicted sentiment labels (0 or 1) for each sample,
                    "predicted_sentiment_probability": list of float: List of predicted sentiment probabilities for each sample
                }
        """
        vectorizer = self.vectorizer
        X_test.reset_index(drop=True, inplace=True)
        X_test_tfidf = vectorizer.transform(X_test["cleaned_text"])
        X_test_tfidf = pd.DataFrame(X_test_tfidf.toarray(), columns=vectorizer.get_feature_names_out())
        X_test_clean = X_test.drop(["cleaned_text", "text"], axis=1)
        X_test_concat = pd.concat([X_test_clean, X_test_tfidf], axis=1)
        X_test_concat = X_test_concat.loc[:, ~X_test_concat.columns.duplicated()].copy()
        probability = self.model.predict_proba(X_test_concat)[:, 0].tolist()  # probability of negative sentiment
        results = [0 if prob > 0.5 else 1 for prob in probability]
        probability = [prob if prob > 0.5 else 1 - prob for prob in probability]

        return {"predicted_sentiment": results, "predicted_sentiment_probability": probability}
