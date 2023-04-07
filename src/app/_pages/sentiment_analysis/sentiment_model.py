import time
from sys import platform

import pandas as pd
import streamlit as st

from src.data.feature_engineering import FeatureEngineer
from src.data.preprocess import Preprocessor
from src.models.sentiment_analysis.pre_trained.seibert import Seibert

best_model = "seibert"

if "output_df" not in st.session_state:
    st.session_state.output_df = None


def run_scoring_pipeline(input_df):
    """Run the scoring pipeline."""
    progress_bar = st.progress(0)
    for i in range(0, 10):
        time.sleep(0.1)
        progress_bar.progress(i + 1, text="Preprocessing...")
    preprocessor = Preprocessor(input_df)
    preprocessor.clean_test_csv()
    progress_bar.progress(20, text="Preprocessing done!")
    progress_bar.progress(30, text="Feature Engineering in progress...")
    pre_processed_df = preprocessor.clean_df
    feature_engineer = FeatureEngineer(pre_processed_df)
    feature_engineer.add_features()
    feature_engineered_df = feature_engineer.feature_engineered_df
    progress_bar.progress(60, text="Feature Engineering Done!")

    time_col = feature_engineered_df.time
    X_test = feature_engineered_df.drop(["time"], axis=1)

    if platform == "win32":
        models_path = "..\\..\\..\\models\\sentiment_analysis"
    else:
        print("entering else block")
        models_path = "models/sentiment_analysis"

    model = Seibert(models_path)
    progress_bar.progress(70, text="Loading Model...")
    model.load(best_model)
    progress_bar.progress(80, text="Model Loaded!")
    pred = model.predict(X_test)
    progress_bar.progress(100, text="Prediction Done!")
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

    # output.to_csv("reviews_test_predictions_data-dialogue.csv", index=False)

    return output


def display():
    st.title("Sentiment Analysis Scoring Pipeline")

    uploaded_file = st.file_uploader("Upload a CSV file with columns 'Time' and 'Text'", type="csv")

    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)

        # Check if uploaded file has "Time" and "Text" columns
        cols = list(input_df.columns)
        if cols != ["Time", "Text"]:
            st.error("Invalid CSV format. Required columns: Time, Text")
        else:
            if "output_df" not in st.session_state:
                output_df = run_scoring_pipeline(input_df)
                st.session_state.output_df = output_df
                st.success("Scoring pipeline completed. Here is your output.")
            else:
                output_df = st.session_state.output_df

            st.dataframe(output_df)

            dl = st.download_button(
                label="Download output file",
                data=output_df.to_csv(index=False).encode(),
                file_name="reviews_test_predictions_data-dialogue.csv",
                mime="text/csv",
            )

            if dl:
                del st.session_state.output_df
                st.success("File downloaded successfully!")
