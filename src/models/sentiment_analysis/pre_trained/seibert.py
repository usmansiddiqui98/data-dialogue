from transformers import pipeline
import torch
from src.models.sentiment_analysis.base_model import BaseModel


class Seibert(BaseModel):
    def __init__(self, models_path):
        super().__init__(models_path)
        if torch.cuda.is_available():
            self.device = 0
        else:
            self.device = -1

    def save(self, model_name):
        pass

    def load(self, model_name):
        pass

    def fit(self, X_train, y_train):
        pass

    def predict(self, x_test):
        x_test = x_test.text.to_list()
        x_test = [x[:512] if len(x) > 512 else x for x in x_test]
        sentiment_analysis = pipeline('sentiment-analysis', model="siebert/sentiment-roberta-large-english",
                                      device=self.device)
        results = sentiment_analysis(x_test)
        labels = [result['label'] for result in results]
        predicted_sentiment = [1 if label == 'POSITIVE' else 0 for label in labels]
        predicted_sentiment_probability = [result['score'] for result in results]
        return {
            'predicted_sentiment': predicted_sentiment,
            'predicted_sentiment_probability': predicted_sentiment_probability
        }