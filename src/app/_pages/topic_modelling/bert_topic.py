import pandas as pd
import streamlit as st

from src.data.preprocess import Preprocessor

if "output_df" not in st.session_state:
    st.session_state.output_df = None


def display():
    st.title("Human Evaluator for topic assignment")
    st.markdown("This page is under construction. Please check back later.")

    uploaded_file = st.file_uploader("Upload a CSV file for topic modelling", type="csv")

    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        st.dataframe(input_df)
