from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier

from src.models.sentiment_analysis.base_model import BaseModel


class XgBoost(BaseModel):
    def __init__(self, vectorizer, dim_reducer=None, *args, **kwargs):
        super().__init__(vectorizer=vectorizer, dim_reducer=dim_reducer, *args, **kwargs)
        self.model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")

    def fit(self, X_train, y_train):
        X_train_processed = self.process(X_train)
        self.model.fit(X_train_processed, y_train)
