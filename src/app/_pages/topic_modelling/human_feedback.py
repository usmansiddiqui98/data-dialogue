import pandas as pd
import streamlit as st


def upload_file():
    file = st.file_uploader("Upload reviews file", type=["csv", "xlsx", "xls"])
    if file is not None:
        # Read file
        df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
        return df


def get_random_sample(dataset, n=50):
    return dataset.sample(n)


def display():
    st.title("Human Evaluator for topic assignment")
    st.markdown("This page is under construction. Please check back later.")
    dataset = upload_file()
    if st.button("Load Random Sample"):
        sample = get_random_sample(dataset)
        for idx, row in sample.iterrows():
            st.subheader(f"Review {idx}")
            st.write(row["Text"])
            topic_input = st.text_input(f"Enter topics for Review {idx}", "")
            row["human_topic"] = topic_input
        if st.button("Save Annotated Sample"):
            sample.to_csv("annotated_sample.csv", index=False)
            st.success("Annotated sample saved successfully!")
