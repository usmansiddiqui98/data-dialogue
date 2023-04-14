import os

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


class BaseModel:
    """
    A class to be inherited by each model for Sentiment Analysis.

    Parameters:
        models_path (str): Path location of all stored models.

    Attributes
    ----------

    model_dir (str):
        Path location of stored model.
    models_path (str):
        Path location of all stored models.
    """

    def __init__(self, models_path):
        self.model_dir = None
        self.models_path = models_path

    def save(self, model_name):
        """
        Save the fitted model as model name. Implemented in model subclass.

        Parameters:
            model_name (str): The name of the model to be saved.
        """
        self.model_dir = os.path.join(self.models_path, model_name)

    def load(self, model_name):
        """
        Load the fitted model with model name. Implemented in model subclass.

        Parameters:
            model_name (str): The name of the model to be loaded.
        """
        self.model_dir = os.path.join(self.models_path, model_name)

    def fit(self, X_train, y_train):
        """
        Fits the model. Implemented in model subclass.

        Parameters:
            X_train (pandas.DataFrame): The input data consisting of review texts and feature engineered features.
            y_train (pandas.DataFrame): The sentiment of X_train.

        Raises
        NotImplementedError
            Raised if fit method in subclass is not implemented.
        """
        raise NotImplementedError("Subclasses should implement this method")

    def predict(self, X):
        """
        Generates model prediction. Implemented in model subclass.

        Parameters:
            X (pandas.DataFrame): The test data consisting of review texts and feature engineered features.

        Raises
        NotImplementedError
            Raised if predict method in subclass is not implemented.
        """
        raise NotImplementedError("Subclasses should implement this method")
