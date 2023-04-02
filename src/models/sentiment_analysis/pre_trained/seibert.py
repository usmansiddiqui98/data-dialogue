from transformers import pipeline
import torch
import pandas as pd


class Seibert:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = 0
        else:
            self.device = -1

    def predict(self, x_test):
        x_test = x_test.text.to_list()
        x_test = [x[:512] if len(x) > 512 else x for x in x_test]
        sentiment_analysis = pipeline('sentiment-analysis', model="siebert/sentiment-roberta-large-english", device=self.device)
        results = sentiment_analysis(x_test)
        labels = [result['label'] for result in results]
        y_pred = [1 if label == 'POSITIVE' else 0 for label in labels]
        return y_pred
