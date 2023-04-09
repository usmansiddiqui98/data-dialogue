import argparse

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

    data = pd.read_csv("../../../data/raw/reviews.csv", parse_dates=["Time"])
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

    if args.model == "lda":
        topics_df = topics_df.rename(columns={"value": "probability"})
    elif args.model == "lsa":
        topics_df = topics_df.rename(columns={"value": "svd_score"})
    elif args.model == "nmf":
        topics_df = topics_df.rename(columns={"value": "tfidf_score"})

    topics_df.to_csv(f"topic_modelling_{args.model}.csv", index=False)
