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

# if "output_df" not in st.session_state:
#    st.session_state.output_df = None

if "pre_processed_df" not in st.session_state:
    st.session_state.pre_processed_df = None

if "lda_lsa_params" not in st.session_state:
    st.session_state.lda_lsa_params = None


def clear_session_state():
    if "lda_lsa_params" in st.session_state:
        del st.session_state["lda_lsa_params"]


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

    model_choice = st.selectbox("Choose which topic model to run", (None, "LDA", "LSA", "NMF", "Bertopic"),
                                on_change=clear_session_state)
    if model_choice in ["LDA", "LSA"]:
        with st.form(key="choose params"):
            num_topics = st.number_input("Number of Topics", min_value=0, max_value=20, step=1)
            tags = st.multiselect("Part-of-Speech Tags", options=["NOUN", "VERB", "ADJ"])

            if num_topics is not None and not tags:
                st.write('Choose a Tag to Continue ...')
                st.stop()

            st.session_state.lda_lsa_params = [num_topics, tags]

            submit_button = st.form_submit_button(label="Submit")

        if submit_button:
            if st.session_state.pre_processed_df is None:
                with st.spinner("Preprocessing Data for topic modelling"):
                    # BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
                    # filepath = os.path.join(BASE_DIR, "data/raw/reviews.csv")
                    data = pd.read_csv("data/raw/reviews.csv", parse_dates=["Time"])
                    preprocessor = Preprocessor(data)
                    preprocessor.clean_csv()
                    pre_processed_df = preprocessor.clean_df
                    st.session_state.pre_processed_df = pre_processed_df

            if st.session_state.pre_processed_df is not None:
                with st.spinner("Running topic modelling..."):
                    if model_choice in ["LDA", "LSA"]:
                        topics_dict = run_training_pipeline(model_choice.lower(), st.session_state.pre_processed_df,
                                                            num_topics=num_topics, tags=tags)
                    else:
                        topics_dict = run_training_pipeline(model_choice.lower(), st.session_state.pre_processed_df)
                    topics_df = topics_dict_to_df(model_choice.lower(), topics_dict)
                    st.success("Done! Here is the topic modelling results:")
                    st.dataframe(topics_df)

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

                if model_choice.lower() == "lda":
                    if st.button('Generate pyLDAvis'):
                        with st.spinner('Creating pyLDAvis Visualization ...'):
                            id2word = lda_model.id2word
                            corpus = lda_model.corpus
                            lda_vis = pyLDAvis.gensim_models.prepare(lda_model.lda_model, corpus, id2word)
                            lda_vis_html = pyLDAvis.prepared_data_to_html(lda_vis)
                        with st.expander('pyLDAvis', expanded=True):
                            components.html(lda_vis_html, width=1300, height=800)

    if not model_choice:
        st.write('Choose a Model to Continue ...')
        st.stop()

    if st.session_state.pre_processed_df is None:
        with st.spinner("Preprocessing Data for topic modelling"):
            # BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
            # filepath = os.path.join(BASE_DIR, "data/raw/reviews.csv")
            data = pd.read_csv("data/raw/reviews.csv", parse_dates=["Time"])
            preprocessor = Preprocessor(data)
            preprocessor.clean_csv()
            pre_processed_df = preprocessor.clean_df
            st.session_state.pre_processed_df = pre_processed_df

    if st.session_state.pre_processed_df is not None:
        with st.spinner("Running topic modelling..."):
            if model_choice in ["LDA", "LSA"]:
                topics_dict = run_training_pipeline(model_choice.lower(), st.session_state.pre_processed_df,
                                                    num_topics=num_topics, tags=tags)
            else:
                topics_dict = run_training_pipeline(model_choice.lower(), st.session_state.pre_processed_df)
            topics_df = topics_dict_to_df(model_choice.lower(), topics_dict)
            st.success("Done! Here is the topic modelling results:")
            st.dataframe(topics_df)

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
