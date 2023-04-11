import os
from sys import platform

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from tqdm.auto import tqdm

from src.data.make_dataset import main as make_dataset
from src.models.sentiment_analysis.log_reg import LogReg
from src.models.sentiment_analysis.pre_trained.siebert import Siebert
from src.models.sentiment_analysis.xg_boost import XgBoost
from src.models.sentiment_analysis.xg_boost_svd import XgBoostSvd


def train_models(models, X_train, y_train, models_path):
    for model_name, model_instance in tqdm(models.items(), desc="Training models"):
        # Train the model
        model_instance.fit(X_train, y_train)

        # Create the model's directory if it doesn't exist
        model_dir = os.path.join(models_path, model_name)
        os.makedirs(model_dir, exist_ok=True)

        model_instance.save(model_name)


def find_best_model(models, models_path, X_test, y_test):
    best_accuracy = 0
    best_model = None
    best_model_name = None

    # Iterate through the model directories
    for model_name, model_instance in models.items():
        model_dir = os.path.join(models_path, model_name)
        if os.path.isdir(model_dir):
            model_instance.load(model_name)
            print("_" * 80)
            print(f"Loaded model: {model_name}")
            pred = model_instance.predict(X_test)
            y_pred = pred["predicted_sentiment"]
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred)
            print(f"Accuracy: {accuracy}")
            print(f"Precision: {precision}")
            print(f"Recall: {recall}")
            print(f"F1 Score: {f1}")
            print(f"ROC AUC: {roc_auc}")
            print("_" * 80)

            # Check if the model has the highest accuracy
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model_instance
                best_model_name = model_name

        # Create the model's directory if it doesn't exist
        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        best_model_dir = os.path.join(BASE_DIR, "models/sentiment_analysis/best_model")
        os.makedirs(best_model_dir, exist_ok=True)
        # Write the best model name to a txt file
        with open(os.path.join(best_model_dir, "best_model_name.txt"), "w") as f:
            f.write(best_model_name)

    return best_model, best_model_name, best_accuracy


if __name__ == "__main__":
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

    # Load the data
    train_filepath = os.path.join(BASE_DIR, "data/processed/train_final_processed_reviews.csv")

    test_filepath = os.path.join(BASE_DIR, "data/processed/test_final_processed_reviews.csv")

    if os.path.exists(train_filepath) and os.path.exists(test_filepath):
        train = pd.read_csv(train_filepath, index_col="Unnamed: 0")
        test = pd.read_csv(test_filepath, index_col="Unnamed: 0")
        X_train = train.drop("sentiment", axis=1)
        X_test = test.drop("sentiment", axis=1)
        y_train = train.sentiment.tolist()
        y_test = test.sentiment.tolist()
    else:
        data = pd.read_csv(os.path.join(BASE_DIR, "data/raw/reviews.csv"))
        X_train, X_test, y_train, y_test = make_dataset(
            data,
            train_split_output_filepath=train_filepath,
            test_split_output_filepath=test_filepath,
        )

    if platform == "win32":
        models_path = "..\\..\\..\\models\\sentiment_analysis"
    else:
        models_path = os.path.join(BASE_DIR, "models/sentiment_analysis")
    models = {
        "xg_boost": XgBoost(models_path),
        "xg_boost_svd": XgBoostSvd(models_path),
        "log_reg": LogReg(models_path),
        # "siebert": Siebert(models_path)
        # Add other model instances here
    }

    # Train the models and save them
    train_models(models, X_train, y_train, models_path)
    best_model, best_model_name, best_accuracy = find_best_model(models, models_path, X_test, y_test)
    print(f"Best model: {best_model_name}")
    print(f"Accuracy: {best_accuracy}")
