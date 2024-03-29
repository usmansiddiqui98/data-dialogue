import os
import pickle

import pandas as pd
from sklearn import model_selection, naive_bayes
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

from src.models.sentiment_analysis.base_model import BaseModel


class Naivebayes(BaseModel):
    """
    A class to train and predict positive and negative sentiment of texts using (Multinomial) Naive Bayes model.
    Naive Bayes classifiers are a collection of classification algorithms based on Bayes Theorem.

    Parameters:
        *args: The variable arguments
        **kwargs: The arbitrary keyword arguments

    Attributes
    ----------

    vectorizer (sklearn.feature_extraction.text.TfidfVectorizer):
        Initialises TfidfVectorizer to convert a collection of raw text to a matrix of TF-IDF features.
    scalar (sklearn.preprocessing.MinMaxScaler):
        Transform features by scaling each feature to a range of 0 to 1 as Naive Bayes does not take in negative values.
    model (sklearn.naive_bayes.MultinomialNB):
        Naive Bayes classifier that takes in tuned parameters.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vectorizer = TfidfVectorizer()
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.tuned_parameters = {"alpha": 0.5, "class_prior": None, "fit_prior": True}
        self.model = naive_bayes.MultinomialNB(**self.tuned_parameters)

    def fit(self, X_train, y_train):
        """
        Function to generate the TF-IDF features by taking in 'cleaned_text' in X_train.
        Scales the feature engineered features to values 0 to 1 using MinMaxScaler.
        Fits the TF-IDF and feature engineered features into the SVM model and trains it.

        Parameters:
            X_train (pandas.DataFrame): The input data consisting of review texts and feature engineered features.
            y_train (pandas.DataFrame): The sentiment of X_train.
        """
        X_train_bow = self.vectorizer.fit_transform(X_train["cleaned_text"])
        X_train_bow = pd.DataFrame(X_train_bow.toarray(), columns=self.vectorizer.get_feature_names_out())
        X_train_clean = X_train.drop(["cleaned_text", "text"], axis=1)
        X_train_clean_scaled = self.scaler.fit_transform(X_train_clean)
        X_train_clean_scaled = pd.DataFrame(X_train_clean_scaled, columns=X_train_clean.columns)
        X_train_concat = pd.concat([X_train_clean_scaled, X_train_bow], axis=1)
        X_train_concat = X_train_concat.loc[:, ~X_train_concat.columns.duplicated()].copy()

        self.model.fit(X_train_concat, y_train)

    def save(self, model_name):
        """
        Save Naive Bayes model and the vectorizer as pickle files.

        Parameters:
            model_name (str): The name of the Naive Bayes model to be saved.
        """
        self.model_dir = os.path.join(self.models_path, model_name)

        with open(os.path.join(self.model_dir, "model.pkl"), "wb") as f:
            pickle.dump(self.model, f)

        with open(os.path.join(self.model_dir, "vectorizer.pkl"), "wb") as f:
            pickle.dump(self.vectorizer, f)

    def load(self, model_name):
        """
        Load Naive Bayes model and the vectorizer pickle files.

        Parameters:
            model_name (str): The name of the Naive Bayes model to be loaded.
        """
        self.model_dir = os.path.join(self.models_path, model_name)

        with open(os.path.join(self.model_dir, "model.pkl"), "rb") as f:
            self.model = pickle.load(f)

        with open(os.path.join(self.model_dir, "vectorizer.pkl"), "rb") as f:
            self.vectorizer = pickle.load(f)

    def predict(self, X_test):
        """
        Generates the TF-IDF features by taking in 'cleaned_text' in X_test.
        Naive Bayes model predicts sentiment & probability of sentiment on unseen data (TF-IDF and scaled feature engineered features of X_test).

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
        X_test_clean_scaled = self.scaler.fit_transform(X_test_clean)
        X_test_clean_scaled = pd.DataFrame(X_test_clean_scaled, columns=X_test_clean.columns)
        X_test_concat = pd.concat([X_test_clean_scaled, X_test_bow], axis=1)
        X_test_concat = X_test_concat.loc[:, ~X_test_concat.columns.duplicated()].copy()
        probability = self.model.predict_proba(X_test_concat)[:, 0].tolist()  # probability of negative sentiment
        results = [0 if prob > 0.5 else 1 for prob in probability]
        probability = [prob if prob > 0.5 else 1 - prob for prob in probability]
        return {"predicted_sentiment": results, "predicted_sentiment_probability": probability}
