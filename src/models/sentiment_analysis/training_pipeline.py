import os
from sys import platform

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from src.data.make_dataset import main as make_dataset
from src.models.sentiment_analysis.pre_trained.seibert import Seibert
from src.models.sentiment_analysis.xg_boost import XgBoost
from src.models.sentiment_analysis.xg_boost_svd import XgBoostSvd


def train_models(models, X_train, y_train, models_path):
    for model_name, model_instance in models.items():
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

            # Check if the model has the highest accuracy
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model_instance
                best_model_name = model_name

    return best_model, best_model_name, best_accuracy


if __name__ == "__main__":
    # Load the data

    train = pd.read_csv("../../../data/processed/train_final_processed_reviews.csv", index_col="Unnamed: 0")
    test = pd.read_csv("../../../data/processed/test_final_processed_reviews.csv", index_col="Unnamed: 0")
    X_train = train.drop("sentiment", axis=1)
    X_test = test.drop("sentiment", axis=1)
    y_train = train.sentiment.tolist()
    y_test = test.sentiment.tolist()

    # X_train, X_test, y_train, y_test = make_dataset(
    #     "../../../data/raw/reviews.csv",
    #     train_split_output_filepath="../../../data/processed/train_final_processed_reviews.csv",
    #     test_split_output_filepath="../../../data/processed/test_final_processed_reviews.csv",
    # )

    if platform == "win32":
        models_path = "..\\..\\..\\models\\sentiment_analysis"
    else:
        models_path = "../../../models/sentiment_analysis"
    models = {
        "xg_boost": XgBoost(models_path),
        "xg_boost_svd": XgBoostSvd(models_path),
        "seibert": Seibert(models_path)
        # Add other model instances here
    }

    # Train the models and save them
    train_models(models, X_train, y_train, models_path)
    best_model, best_model_name, best_accuracy = find_best_model(models, models_path, X_test, y_test)
    print(f"Best model: {best_model_name}")
    print(f"Accuracy: {best_accuracy}")
