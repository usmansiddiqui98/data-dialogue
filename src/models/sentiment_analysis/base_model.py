class BaseModel:
    def __init__(self):
        self.model = None

    def train(self, X_train, y_train):
        raise NotImplementedError

    def predict(self, X_test):
        return self.model.predict(X_test)
