import os

import pandas as pd

import src.data.feature_engineering as feature_engineering

fname = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "processed", "clean_reviews_w_topics.csv")
)


def test_add_features():
    df_full = pd.read_csv(fname)
    df = df_full.head(25)
    df = feature_engineering.add_features(df)
    assert df["Lowercase Count"][22] == 351
    assert df["Uppercase Count"][22] == 2
    assert df["Uppercase Words"][22] == "GOT, KIDDING"
    assert df["Uppercase Ratio"][22] == 0.005525
    assert df["Punc Count"][22] == 60
    assert df["num_digits"][22] == 4
    assert df["num_verbs"][22] == 21
    assert df["num_nouns"][22] == 57
    assert df["num_tokens_cleaned"][22] == 134
    assert df["num_tokens_raw"][22] == 362
    assert df["num_words_misspelled"][22] == 12
    assert df["polarity"][22] == 0.9956
    assert df["subjectivity"][22] == 0.489659
    assert df["num_pos_words"][22] == 3
    assert df["num_neg_words"][22] == 0
