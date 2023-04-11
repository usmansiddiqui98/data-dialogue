import os
import sys

import pandas as pd
import streamlit as st

from src.data.preprocess import Preprocessor
from src.models.topic_modelling.training_pipeline import run_training_pipeline
from src.models.topic_modelling.training_pipeline import topics_dict_to_df
from wordcloud import WordCloud

sys.path.append(os.path.abspath("../../models/topic_modelling/"))

# if "output_df" not in st.session_state:
#    st.session_state.output_df = None

if "pre_processed_df" not in st.session_state:
    st.session_state.pre_processed_df = None


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
    if st.session_state.pre_processed_df is None:
        # BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        # filepath = os.path.join(BASE_DIR, "data/raw/reviews.csv")
        data = pd.read_csv("data/raw/reviews.csv", parse_dates=["Time"])
        preprocessor = Preprocessor(data)
        preprocessor.clean_csv()
        pre_processed_df = preprocessor.clean_df
        st.session_state.pre_processed_df = pre_processed_df
    st.dataframe(st.session_state.pre_processed_df.head())

    with st.form(key="topic_modelling_form"):
        model_choice = st.selectbox("Choose which topic model to run", ("LDA", "LSA", "NMF", "Bertopic"))
        submit_button = st.form_submit_button(label="Run Topic Modelling")

    if submit_button:
        with st.spinner("Running topic modelling..."):
            topics_dict = run_training_pipeline(model_choice.lower(), st.session_state.pre_processed_df)
            topics_df = topics_dict_to_df(model_choice.lower(), topics_dict)
            st.success("Done!")
            st.dataframe(topics_df)

        if st.success:
            topics_dict = run_training_pipeline(model_choice.lower(), st.session_state.pre_processed_df)
            with st.expander('Topic Word-Weighted Summaries'):
                for topic_idx, topic in topics_dict.items():
                    topic_summary = ' + '.join([f'{prob:.3f} * {word}' for word, prob in topic.items()])
                    st.markdown(f'**Topic {topic_idx}**: _{topic_summary}_')

            with st.expander('Top N Topic Keywords Wordclouds'):
                cols = st.columns(3)
                for index, topic in enumerate(topics):
                    wc = WordCloud(font_path=WORDCLOUD_FONT_PATH, width=700, height=600,
                                   background_color='white', collocations=collocations, prefer_horizontal=1.0,
                                   color_func=lambda *args, **kwargs: colors[index])
                    with cols[index % 3]:
                        wc.generate_from_frequencies(dict(topic[1]))
                        st.image(wc.to_image(), caption=f'Topic #{index}', use_column_width=True)

