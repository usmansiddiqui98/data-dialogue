from base_model import BaseModel


class LogisticRegression(BaseModel):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = LogisticRegression(**kwargs)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
