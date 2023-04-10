import argparse
import os

import pandas as pd

from src.data.preprocess import Preprocessor
from src.models.topic_modelling.LDA import LDAGensim
from src.models.topic_modelling.LSA import LSAModel
from src.models.topic_modelling.NMF import NMFModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Topic Modelling Pipeline")
    parser.add_argument(
        "--model", type=str, choices=["lda", "lsa", "nmf"], help="Choose which topic model to run: lda, lsa, or nmf"
    )
    args = parser.parse_args()

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

    filepath = os.path.join(BASE_DIR, "data/raw/reviews.csv")
    data = pd.read_csv(filepath, parse_dates=["Time"])
    preprocessor = Preprocessor(data)
    preprocessor.clean_csv()
    pre_processed_df = preprocessor.clean_df

    if args.model:
        if args.model == "lda":
            lda_model = LDAGensim(pre_processed_df, tags=["NOUN"])
            print("Running LDA Model...")
            topics_dict = lda_model.get_topics()
        elif args.model == "lsa":
            lsa_model = LSAModel(pre_processed_df, tags=["NOUN"])
            print("Running LSA Model...")
            topics_dict = lsa_model.get_topics()
        elif args.model == "nmf":
            nmf_model = NMFModel(pre_processed_df)
            print("Running NMF Model...")
            nmf_model.fit_transform()
            topics_dict = nmf_model.get_topic_terms()
    else:
        raise ValueError("Please specify a model to run.")

    topics_df = pd.DataFrame.from_dict(
        {(i, j): topics_dict[i][j] for i in topics_dict.keys() for j in topics_dict[i].keys()},
        orient="index",
        columns=["value"],
    )

    # split the row index into two separate columns
    topics_df.index = pd.MultiIndex.from_tuples(topics_df.index, names=["topic", "word"])

    # reset the index to turn the MultiIndex into columns
    topics_df = topics_df.reset_index()

    if args.model in ["lda", "lsa", "nmf"]:
        if args.model == "lda":
            score_col = "probability"
        elif args.model == "lsa":
            score_col = "svd_score"
        else:
            score_col = "tfidf_score"

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
        for i in range(1, len(pivoted_df.columns[2:]), 1):
            column_order += [f"word_{i // 2 + 1}", f"{score_col}_{i // 2 + 1}"]

    pivoted_df = pivoted_df.reindex(columns=column_order)
    pivoted_df.to_csv(f"topic_modelling_{args.model}.csv", index=False)
