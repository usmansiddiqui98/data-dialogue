import os
import sys

import pandas as pd
import streamlit as st

from src.models.topic_modelling.training_pipeline import run_training_pipeline

sys.path.append(os.path.abspath("../../models/topic_modelling/"))


if "output_df" not in st.session_state:
    st.session_state.output_df = None


def display():
    st.title("Human Evaluator for topic modelling")

    with st.expander("What is topic modelling?"):
        st.markdown(
            "Topic modeling is a broad term. It encompasses a number of specific statistical learning methods. "
            "These methods do the following: explain documents in terms of a set of topics and those topics in terms of "
            "the a set of words. Two very commonly used methods are Latent Dirichlet Allocation (LDA), Non-Negative "
            "Matrix Factorization (NMF),  Latent Semantic Analysis (LSA) and Bertopic for instance."
        )
    st.markdown("#### This is the data we will be using for topic modelling:")
    pre_processed_df = pd.read_csv("data/processed/clean_reviews_w_topics.csv", parse_dates=["time"])
    st.dataframe(pre_processed_df.head())
    with st.form(key="topic_modelling_form"):
        model_choice = st.selectbox("Choose which topic model to run", ("LDA", "LSA", "NMF", "Bertopic"))
        submit_button = st.form_submit_button(label="Run Topic Modelling")

    if submit_button:
        with st.spinner("Running topic modelling..."):
            topics_df = run_training_pipeline(model_choice.lower(), pre_processed_df)
            st.success("Done!")
        st.session_state.output_df = topics_df

    if st.session_state.output_df is not None:
        st.markdown("Here are the topic modelling results:")
        st.dataframe(st.session_state.output_df)
        dl = st.download_button(
            label="Download output file",
            data=st.session_state.output_df.to_csv(index=False).encode(),
            file_name="topic_models_data-dialogue.csv",
            mime="text/csv",
        )

        if dl:
            del st.session_state.output_df
            st.success("File downloaded successfully!")
