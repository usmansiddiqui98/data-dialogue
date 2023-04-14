import os
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.models.sentiment_analysis.base_model import BaseModel


class LogReg(BaseModel):
    """
    Logistic Regression Model for sentiment analysis.

    Attributes
    ----------

    vectorizer (TfidfVectorizer):
        TF-IDF vectorizer for text feature extraction.
    model (LogisticRegression):
        Logistic regression model for classification.
    """

    def __init__(self, *args, **kwargs):
        """
        Constructor for LogReg class.

        Parameters:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        """
        super().__init__(*args, **kwargs)
        self.vectorizer = TfidfVectorizer()
        self.scaler = StandardScaler()
        self.model = LogisticRegression(random_state=4265, max_iter=1000)

    def fit(self, X_train, y_train):
        """
        Fits the logistic regression model to the training data.

        Parameters:
            X_train (pandas.DataFrame): The input data consisting of review texts and feature engineered features.
            y_train (pandas.DataFrame): The sentiment of X_train.

        """
        X_train_bow = self.vectorizer.fit_transform(X_train["cleaned_text"])
        X_train_bow = pd.DataFrame(X_train_bow.toarray(), columns=self.vectorizer.get_feature_names_out())
        X_train_clean = X_train.drop(["cleaned_text", "text"], axis=1)
        X_train_concat = pd.concat([X_train_clean, X_train_bow], axis=1)
        X_train_concat = X_train_concat.loc[:, ~X_train_concat.columns.duplicated()].copy()
        X_train_scaled = self.scaler.fit_transform(X_train_concat)
        self.model.fit(X_train_scaled, y_train)

    def save(self, model_name):
        """
        Saves the trained model and vectorizer to disk.

        Parameters:
            model_name (str): Name of the model to be saved.

        """
        self.model_dir = os.path.join(self.models_path, model_name)

        with open(os.path.join(self.model_dir, "model.pkl"), "wb") as f:
            pickle.dump(self.model, f)

        with open(os.path.join(self.model_dir, "vectorizer.pkl"), "wb") as f:
            pickle.dump(self.vectorizer, f)

        with open(os.path.join(self.model_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(self.scaler, f)

    def load(self, model_name):
        """
        Loads the trained model and vectorizer from disk.

        Parameters:
            model_name (str): Name of the model to be loaded.

        """
        self.model_dir = os.path.join(self.models_path, model_name)

        with open(os.path.join(self.model_dir, "model.pkl"), "rb") as f:
            self.model = pickle.load(f)

        with open(os.path.join(self.model_dir, "vectorizer.pkl"), "rb") as f:
            self.vectorizer = pickle.load(f)

        with open(os.path.join(self.model_dir, "scaler.pkl"), "rb") as f:
            self.scaler = pickle.load(f)

    def predict(self, X_test):
        """
        Predicts sentiment labels for test data.

        Parameters:
            X_test (DataFrame): Test data containing cleaned text.

        Returns:
            dict: Dictionary containing predicted sentiment labels and probabilities.
                {"predicted_sentiment": List of predicted sentiment labels,
                 "predicted_sentiment_probability": List of predicted sentiment probabilities}

        """
        X_test_bow = self.vectorizer.transform(X_test["cleaned_text"])
        X_test_bow = pd.DataFrame(X_test_bow.toarray(), columns=self.vectorizer.get_feature_names_out())
        X_test_clean = X_test.drop(["cleaned_text", "text"], axis=1)
        X_test_concat = pd.concat([X_test_clean, X_test_bow], axis=1)
        X_test_concat = X_test_concat.loc[:, ~X_test_concat.columns.duplicated()].copy()

        X_test_scaled = self.scaler.transform(X_test_concat)
        probability = self.model.predict_proba(X_test_scaled)[:, 0].tolist()  # probability of negative sentiment
        results = [0 if prob > 0.5 else 1 for prob in probability]
        probability = [prob if prob > 0.5 else 1 - prob for prob in probability]
        return {"predicted_sentiment": results, "predicted_sentiment_probability": probability}
