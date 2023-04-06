import os

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


class BaseModel:
    def __init__(self, models_path):
        self.model_dir = None
        self.models_path = models_path

    def save(self, model_name):
        self.model_dir = os.path.join(self.models_path, model_name)

    def load(self, model_name):
        self.model_dir = os.path.join(self.models_path, model_name)

    def fit(self, X_train, y_train):
        raise NotImplementedError("Subclasses should implement this method")

    def predict(self, X):
        # "predicted_sentiment_probability", "predicted_sentiment"
        raise NotImplementedError("Subclasses should implement this method")
