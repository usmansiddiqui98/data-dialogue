import os
import time
from sys import platform

import pandas as pd

from src.data.feature_engineering import FeatureEngineer
from src.data.preprocess import Preprocessor
from src.models.sentiment_analysis.log_reg import LogReg
from src.models.sentiment_analysis.pre_trained.siebert import Siebert
from src.models.sentiment_analysis.xg_boost import XgBoost
from src.models.sentiment_analysis.xg_boost_svd import XgBoostSvd


def run_scoring_pipeline(input_df):
    """
    Execute the sentiment analysis scoring pipeline on the input DataFrame.

    The pipeline consists of the following steps:
    1. Preprocessing the input data.
    2. Feature engineering.
    3. Model selection and loading.
    4. Prediction using the best model.
    5. Saving the predicted sentiment and sentiment probabilities in an output DataFrame.

    Parameters
    input_df : pandas.DataFrame
        The input DataFrame containing the raw reviews data.

    Returns
    output : pandas.DataFrame
        The output DataFrame containing the predicted sentiment and sentiment probabilities for each review.
    """

    # ________CHANGE THIS TO CHANGE MODEL_______
    with open("models/sentiment_analysis/best_model/best_model_name.txt") as f:
        best_model_name = f.readlines()
    best_model = best_model_name[0].strip()
    # ________CHANGE THIS TO CHANGE MODEL_______
    if platform == "win32":
        models_path = "models\\sentiment_analysis"
    else:
        models_path = "models/sentiment_analysis"

    model_classes = {
        "xg_boost": XgBoost,
        "xg_boost_svd": XgBoostSvd,
        "log_reg": LogReg,
        # "siebert": Siebert,
        # Add other model instances here
    }
    # Use the best_model variable to create the corresponding model object
    model = model_classes[best_model](models_path)

    start = time.time()
    preprocessor = Preprocessor(input_df)
    preprocessor.clean_test_csv()
    pre_processed_df = preprocessor.clean_df
    feature_engineer = FeatureEngineer(pre_processed_df)
    feature_engineer.add_features()
    feature_engineered_df = feature_engineer.feature_engineered_df
    fe_end = time.time()
    total_time_fe = fe_end - start

    print("\n" + "Preprocessing and Feature Engineering finished in " + str(round(total_time_fe)) + "s")

    time_col = feature_engineered_df.time
    X_test = feature_engineered_df.drop(["time"], axis=1)
    model.load(best_model)
    pred = model.predict(X_test)
    end = time.time()
    total_time = end - start

    print(f"Prediction Done using {best_model.title()} in " + str(round(total_time)) + "s")

    # The output file should be named "reviews_test_predictions_<your_group_name>.csv ,
    # and it should have columns - "Text", Time", "predicted_sentiment_probability", "predicted_sentiment".

    output = pd.DataFrame(
        {
            "Text": X_test.text,
            "Time": time_col,
            "predicted_sentiment_probability": pred["predicted_sentiment_probability"],
            "predicted_sentiment": pred["predicted_sentiment"],
        }
    )

    return output


if __name__ == "__main__":
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    # Load the data
    input_df = pd.read_csv("data/raw/reviews_test.csv")
    # Run the pipeline
    output_df = run_scoring_pipeline(input_df)
    # Save the output
    output_df.to_csv("data/predictions/reviews_test_predictions_data-dialogue.csv", index=False)
