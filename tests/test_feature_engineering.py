import os

import pandas as pd

from src.data.feature_engineering import FeatureEngineer
from src.data.preprocess import Preprocessor

fname = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "reviews.csv"))
input_df = pd.read_csv(fname).head(25)
preprocessor = Preprocessor(input_df)
preprocessor.clean_csv()
pre_processed_df = preprocessor.clean_df
feature_engineer = FeatureEngineer(pre_processed_df)
feature_engineer.add_features()
feature_engineered_df = feature_engineer.feature_engineered_df


def test_add_features():
    assert feature_engineered_df["lowercase_count"][22] == 351
    assert feature_engineered_df["uppercase_count"][22] == 2
    assert feature_engineered_df["uppercase_ratio"][22] == 0.005525
    assert feature_engineered_df["punc_count"][22] == 60
    assert feature_engineered_df["num_digits"][22] == 0
    assert feature_engineered_df["num_verbs"][22] == 21
    assert feature_engineered_df["num_nouns"][22] == 52
    assert feature_engineered_df["num_tokens_cleaned"][22] == 152
    assert feature_engineered_df["num_tokens_raw"][22] == 362
    assert feature_engineered_df["num_words_misspelled"][22] == 12
    assert feature_engineered_df["polarity"][22] == 0.9933
    assert feature_engineered_df["subjectivity"][22] == 0.511613
    assert feature_engineered_df["num_pos_words"][22] == 3
    assert feature_engineered_df["num_neg_words"][22] == 0
