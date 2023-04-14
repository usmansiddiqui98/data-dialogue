import pandas as pd
import streamlit as st

from src.models.sentiment_analysis.scoring_pipeline import run_scoring_pipeline

if "output_df" not in st.session_state:
    st.session_state.output_df = None


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
                st.session_state.output_df = None

            output_df = st.session_state.output_df

            if output_df is None:
                with st.spinner("Running scoring pipeline..."):
                    output_df = run_scoring_pipeline(input_df)
                    st.session_state.output_df = output_df
                    st.success("Scoring pipeline completed. Here is your output.")

            if output_df is not None:
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
