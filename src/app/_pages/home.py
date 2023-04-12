import streamlit as st


def display():
    # Explain what the application is about
    st.header("Welcome to Data Dialogue!")
    st.subheader("What is Data Dialogue?")
    st.markdown(
        "Whether for a B2B or B2C company,"
        " relevance and longevity in the industry"
        " depends on how well the products answer"
        " the needs of the customers."
        " However, when the time comes for the companies"
        " to demonstrate that understanding —"
        " during a sales conversation, customer service interaction,"
        " or through the product itself"
        " — how can companies"
        " evaluate how they measure up?"
    )

    st.markdown("---")
    st.subheader("Spaces:")
    st.markdown("#### Sentiment Analysis")
    st.markdown("We have developed a tool to predict the sentiment of the given reviews of your products.")
    st.markdown("#### Topic Modelling")
    st.markdown("We also have a tool to predict the topics of the given reviews of your products.")
    st.markdown("---")
    # instructions
    st.markdown("To continue, please select a **space** from the sidebar...")
