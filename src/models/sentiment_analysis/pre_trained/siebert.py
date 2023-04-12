import torch
from transformers import pipeline

from src.models.sentiment_analysis.base_model import BaseModel


class Siebert(BaseModel):
    """
    Class implementing Sentiment analysis using Siebert's RoBERTa model.

    Attributes
    ----------
    models_path : str
        Path to directory where the model should be saved or loaded from.

    device : int
        The device where the model should be loaded. If CUDA is available, the device is set to 0, otherwise it is set to -1.
    """

    def __init__(self, models_path):
        """
        Parameters
        ----------
        models_path : str
            Path to directory where the model should be saved or loaded from.
        """
        super().__init__(models_path)
        if torch.cuda.is_available():
            self.device = 0
        else:
            self.device = -1

    def save(self, model_name):
        """
        Save the model under the given model_name in the models_path directory.
        This method is not implemented since Siebert uses a pre-trained model.

        Parameters
        ----------
        model_name : str
            Name of the model to be saved.

        Returns
        -------
        None
        """
        pass

    def load(self, model_name):
        """
        Load the model from the models_path directory. This method is not implemented since Siebert uses a pre-trained model.

        Parameters
        ----------
        model_name : str
            Name of the model to be loaded.

        Returns
        -------
        None
        """
        pass

    def fit(self, X_train, y_train):
        """
        Fit method is not implemented since Siebert uses a pre-trained model.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training data, a Pandas DataFrame containing text data to be used for training.

        y_train : pd.Series
            Labels corresponding to the training data.

        Returns
        -------
        None
        """
        pass

    def predict(self, x_test):
        """
        Given a Pandas DataFrame of text data, this method returns a dictionary containing the predicted sentiment and
        the corresponding probabilities for each sample in the input.

        Parameters
        ----------
        x_test : pd.DataFrame
            Test data, a Pandas DataFrame containing text data to be used for predicting sentiment.

        Returns
        -------
        dict
            A dictionary containing the predicted sentiment and the corresponding probabilities for each sample in the input.
        """
        x_test = x_test.text.to_list()
        x_test = [x[:512] if len(x) > 512 else x for x in x_test]
        sentiment_analysis = pipeline(
            "sentiment-analysis", model="siebert/sentiment-roberta-large-english", device=self.device
        )
        results = sentiment_analysis(x_test)
        labels = [result["label"] for result in results]
        predicted_sentiment = [1 if label == "POSITIVE" else 0 for label in labels]
        predicted_sentiment_probability = [result["score"] for result in results]
        return {
            "predicted_sentiment": predicted_sentiment,
            "predicted_sentiment_probability": predicted_sentiment_probability,
        }
