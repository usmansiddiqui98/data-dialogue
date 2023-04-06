import os
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier

from src.models.sentiment_analysis.base_model import BaseModel


class XgBoost(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vectorizer = TfidfVectorizer()
        self.model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=4265)

    def fit(self, X_train, y_train):
        # X_train_processed = self.fit_transform(X_train)

        vectorizer = self.vectorizer
        X_train.reset_index(drop=True, inplace=True)
        X_train_tfidf = vectorizer.fit_transform(X_train["cleaned_text"])
        X_train_tfidf = pd.DataFrame(X_train_tfidf.toarray(), columns=vectorizer.get_feature_names_out())
        X_train_clean = X_train.drop(["cleaned_text", "text"], axis=1)
        X_train_concat = pd.concat([X_train_clean, X_train_tfidf], axis=1)
        X_train_concat = X_train_concat.loc[:, ~X_train_concat.columns.duplicated()].copy()

        self.model.fit(X_train_concat, y_train)

    def save(self, model_name):
        # Save the model, vectorizer to pickle files
        self.model_dir = os.path.join(self.models_path, model_name)

        with open(os.path.join(self.model_dir, "model.pkl"), "wb") as f:
            pickle.dump(self.model, f)
        with open(os.path.join(self.model_dir, "vectorizer.pkl"), "wb") as f:
            pickle.dump(self.vectorizer, f)

    def load(self, model_name):
        # Load model, vectorizer
        self.model_dir = os.path.join(self.models_path, model_name)

        with open(os.path.join(self.model_dir, "model.pkl"), "rb") as f:
            self.model = pickle.load(f)
        with open(os.path.join(self.model_dir, "vectorizer.pkl"), "rb") as f:
            self.vectorizer = pickle.load(f)

    def predict(self, X_test):
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
