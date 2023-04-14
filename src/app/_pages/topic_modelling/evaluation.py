import os
import sys

import pandas as pd
import streamlit as st
import pyLDAvis.gensim_models
import streamlit.components.v1 as components
from wordcloud import WordCloud

from src.data.preprocess import Preprocessor
from src.models.topic_modelling.training_pipeline import run_training_pipeline, topics_dict_to_df

sys.path.append(os.path.abspath("../../models/topic_modelling/"))

if "pre_processed_df" not in st.session_state:
    st.session_state.pre_processed_df = None

def run_preprocessing_and_topic_modelling(model_choice, num_topics=None, tags=None):
    if st.session_state.pre_processed_df is None:
        with st.spinner("Preprocessing Data for topic modelling"):
            data = pd.read_csv("data/raw/reviews.csv", parse_dates=["Time"])
            preprocessor = Preprocessor(data)
            preprocessor.clean_csv()
            pre_processed_df = preprocessor.clean_df
            st.session_state.pre_processed_df = pre_processed_df

    if st.session_state.pre_processed_df is not None:
        with st.spinner("Running topic modelling..."):
            topics_dict = run_training_pipeline(model_choice.lower(), st.session_state.pre_processed_df,
                                                num_topics=num_topics, tags=tags)
            topics_df = topics_dict_to_df(model_choice.lower(), topics_dict)
            st.success("Done! Here is the topic modelling results:")
            st.dataframe(topics_df)
        return topics_dict

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

    model_choice = st.selectbox("Choose which topic model to run", (None, "LDA", "LSA", "NMF", "Bertopic"))

    if not model_choice:
        st.write('Choose a Model to Continue ...')
        st.stop()

    num_topics, tags = None, None
    if model_choice in ["LDA", "LSA"]:
        num_topics = st.number_input("Number of Topics", min_value=0, max_value=20, step=1)
        tags = st.multiselect("Part-of-Speech Tags", options=["NOUN", "VERB", "ADJ"])

        if num_topics is not None and not tags:
            st.write('Choose a Tag to Continue ...')
            st.stop()

    if st.button("Run Topic Modelling"):
        topics_dict = run_preprocessing_and_topic_modelling(model_choice, num_topics, tags)

        with st.expander("Topic Word-Weighted Summaries"):
            for topic_idx, topic in topics_dict.items():
                topic_summary = " + ".join([f"{prob:.3f} * {word}" for word, prob in topic.items()])
                st.markdown(f"**Topic {topic_idx}**: _{topic_summary}_")

        with st.expander("Top N Topic Keywords Wordclouds"):
            cols = st.columns(3)
            for index, topic in topics_dict.items():
                wc = WordCloud(width=700, height=600, background_color="white", prefer_horizontal=1.0)
                with cols[index % 3]:
                    wc.generate_from_frequencies(topic)
                    st.image(wc.to_image(), caption=f"Topic #{index}", use_column_width=True)