import os

import pandas as pd

from src.data.preprocess import Preprocessor
from src.models.topic_modelling.bert_topic import BertTopic
from src.models.topic_modelling.LDA import LDAGensim
from src.models.topic_modelling.LSA import LSAModel
from src.models.topic_modelling.NMF import NMFModel


def run_training_pipeline(model_choice, pre_processed_df):
    if model_choice:
        if model_choice == "lda":
            lda_model = LDAGensim(pre_processed_df, tags=["NOUN"])
            print("Running LDA Model...")
            topics_dict = lda_model.get_topics()
        elif model_choice == "lsa":
            lsa_model = LSAModel(pre_processed_df, tags=["NOUN"])
            print("Running LSA Model...")
            topics_dict = lsa_model.get_topics()
        elif model_choice == "nmf":
            nmf_model = NMFModel(pre_processed_df)
            print("Running NMF Model...")
            nmf_model.fit_transform()
            topics_dict = nmf_model.get_topic_terms()
        # elif model_choice == "bertopic":
        #     bertopic_model = BertTopic(pre_processed_df)
        #     print("Running BertTopic Model...")
        #     bertopic_model.prepare_embeddings()
        #     bertopic_model.run_bertopic()
        #     topics_dict = bertopic_model.get_topics()
    else:
        raise ValueError("Please specify a model to run.")

    return topics_dict


def topics_dict_to_df(model_choice, topics_dict):
    topics_df = pd.DataFrame.from_dict(
        {(i, j): topics_dict[i][j] for i in topics_dict.keys() for j in topics_dict[i].keys()},
        orient="index",
        columns=["value"],
    )

    # split the row index into two separate columns
    topics_df.index = pd.MultiIndex.from_tuples(topics_df.index, names=["topic", "word"])

    # reset the index to turn the MultiIndex into columns
    topics_df = topics_df.reset_index()

    if model_choice == "lda":
        score_col = "probability"
    elif model_choice == "lsa":
        score_col = "svd_score"
    elif model_choice == "nmf":
        score_col = "tfidf_score"
    # elif model_choice == "bertopic":
    #     score_col = "probability"

    topics_df = topics_df.rename(columns={"value": score_col})

    pivoted_df = topics_df.pivot_table(
        index="topic",
        columns=topics_df.groupby(["topic"]).cumcount() + 1,
        values=["word", score_col],
        aggfunc="first",
    ).reset_index()

    pivoted_df.columns = ["_".join(map(str, col)).strip() for col in pivoted_df.columns.values]
    pivoted_df = pivoted_df.rename(columns={"topic_": "topic_id"})

    column_order = ["topic_id"]
    for i in range(1, len(pivoted_df.columns[2:]), 2):
        column_order += [f"word_{i // 2 + 1}", f"{score_col}_{i // 2 + 1}"]

    pivoted_df = pivoted_df.reindex(columns=column_order)

    return pivoted_df


if __name__ == "__main__":
    model_choice = input("Choose which topic model to run (lda, lsa, nmf, bertopic): ")

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

    filepath = os.path.join(BASE_DIR, "data/raw/reviews.csv")
    data = pd.read_csv(filepath, parse_dates=["Time"])
    preprocessor = Preprocessor(data)
    preprocessor.clean_csv()
    pre_processed_df = preprocessor.clean_df

    # pre_processed_df = pd.read_csv("../../../data/processed/clean_reviews_w_topics.csv", parse_dates=["time"])
    topics_dict = run_training_pipeline(model_choice, pre_processed_df)
    pivoted_df = topics_dict_to_df(model_choice, topics_dict)
    pivoted_df.to_csv(f"topic_modelling_{model_choice}.csv", index=False)
