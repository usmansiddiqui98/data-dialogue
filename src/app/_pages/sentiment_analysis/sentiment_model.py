import pandas as pd
import plotly.express as px
import streamlit as st

from src.models.sentiment_analysis.pre_trained.predict_bert import apply_pre_process_text, apply_predict_sentiment


def display():
    st.title("Sentiment Analysis")
    st.markdown("This page is under construction. Please check back later.")

    # Upload file and ensure correct format
    file = st.file_uploader("Upload a file", type=["csv", "xlsx", "xls"])

    if file is not None:
        # Read file
        df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
        st.dataframe(df)

        # Preprocess text and predict sentiment for each row (Scoring Pipeline)
        df = apply_pre_process_text(df)
        df = apply_predict_sentiment(df)

        # Create summary of sentiment scores
        summary = df["sentiment_score"].describe()
        st.write("Summary of Sentiment Scores:")
        st.write(summary)

        # Create sentiment analysis plot
        fig = px.histogram(df, x="sentiment_score", nbins=10, color_discrete_sequence=["blue"])
        fig.update_layout(title="Sentiment Analysis")
        st.plotly_chart(fig)
