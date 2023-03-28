import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class BaseModel:
    def __init__(self):
        self.model = None

    def train(self, X_train, y_train):
        raise NotImplementedError

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        f1 = f1_score(y_test, self.predict(X_test), average="weighted")
        precision = precision_score(y_test, self.predict(X_test), average="weighted")
        recall = recall_score(y_test, self.predict(X_test), average="weighted")
        accuracy = accuracy_score(y_test, self.predict(X_test))
        return f1, precision, recall, accuracy
