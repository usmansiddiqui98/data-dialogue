import os
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

from src.data.make_dataset import main as make_dataset
from src.models.sentiment_analysis.xg_boost import XgBoost


def train_models(models, X_train, y_train, models_path):
    for model_name, model_instance in models.items():
        # Train the model
        model_instance.fit(X_train, y_train)

        # Create the model's directory if it doesn't exist
        model_dir = os.path.join(models_path, model_name)
        os.makedirs(model_dir, exist_ok=True)

        # Save the model, vectorizer, and dimensionality reducer to pickle files
        with open(os.path.join(model_dir, "model.pkl"), "wb") as f:
            pickle.dump(model_instance.model, f)
        with open(os.path.join(model_dir, "vectorizer.pkl"), "wb") as f:
            pickle.dump(model_instance.vectorizer, f)
        if model_instance.dim_reducer is not None:
            with open(os.path.join(model_dir, "dim_reducer.pkl"), "wb") as f:
                pickle.dump(model_instance.dim_reducer, f)


def find_best_model(models_path, X_test, y_test):
    best_accuracy = 0
    best_model = None
    best_vectorizer = None
    best_dim_reducer = None
    best_model_name = None

    # Iterate through the model directories
    for model_name in os.listdir(models_path):
        model_dir = os.path.join(models_path, model_name)

        if os.path.isdir(model_dir):
            # Load model, vectorizer, and dimension reducer
            with open(os.path.join(model_dir, "model.pkl"), "rb") as f:
                model = pickle.load(f)
            with open(os.path.join(model_dir, "vectorizer.pkl"), "rb") as f:
                vectorizer = pickle.load(f)
            with open(os.path.join(model_dir, "dim_reducer.pkl"), "rb") as f:
                dim_reducer = pickle.load(f)

            # Preprocess the test data
            X_test_vectorized = vectorizer.transform(X_test)
            if dim_reducer is not None:
                X_test_reduced = dim_reducer.transform(X_test_vectorized)
            else:
                X_test_reduced = X_test_vectorized

            # Predict and evaluate the model
            y_pred = model.predict(X_test_reduced)
            accuracy = accuracy_score(y_test, y_pred)

            # Check if the model has the highest accuracy
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_vectorizer = vectorizer
                best_dim_reducer = dim_reducer
                best_model_name = model_name

    return best_model, best_vectorizer, best_dim_reducer, best_model_name, best_accuracy


if __name__ == "__main__":
    # Load the data
    X_train, X_test, y_train, y_test = make_dataset("../../../data/raw/reviews.csv")
    models = {
        "xg_boost": XgBoost(vectorizer=TfidfVectorizer()),
        # Add other model instances here
    }
    models_path = "../../../models/sentiment_analysis"
    # Train the models and save them
    train_models(models, X_train, y_train, models_path)
    best_model, best_vectorizer, best_dim_reducer, best_model_name, best_accuracy = find_best_model(
        models_path, X_test, y_test
    )
    print(f"Best model: {best_model_name}")
    print(f"Accuracy: {best_accuracy}")
