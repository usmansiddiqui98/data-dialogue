from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


class BaseModel:
    def __init__(self, vectorizer=None, dim_reducer=None):
        self.vectorizer = vectorizer or TfidfVectorizer()
        self.dim_reducer = dim_reducer

    def fit_transform(self, X):
        X_vectorized = self.vectorizer.fit_transform(X)
        if self.dim_reducer is not None:
            return self.dim_reducer.fit_transform(X_vectorized)
        else:
            return X_vectorized

    def process(self, X):
        X_vectorized = self.vectorizer.transform(X)
        if self.dim_reducer is not None:
            return self.dim_reducer.transform(X_vectorized)
        else:
            return X_vectorized

    def fit(self, X_train, y_train):
        raise NotImplementedError("Subclasses should implement this method")

    def predict(self, X):
        raise NotImplementedError("Subclasses should implement this method")

    def evaluate(self, X_test, y_test):
        raise NotImplementedError("Subclasses should implement this method")
