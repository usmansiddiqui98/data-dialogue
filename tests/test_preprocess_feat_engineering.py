import os

from src.data.feature_engineering import FeatureEngineer
from src.data.preprocess import Preprocessor

fname = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "reviews.csv"))

preprocessor = Preprocessor(fname)
preprocessor.clean_csv()
pre_processed_df = preprocessor.clean_df.head(25)
feature_engineer = FeatureEngineer(pre_processed_df)
feature_engineer.add_features()
feature_engineered_df = feature_engineer.feature_engineered_df


def test_clean():
    text = Preprocessor.clean_sentence(
        "This is a very healthy dog food. Good for their digestion. Also good for small puppies. "
        "My dog eats her required amount at every feeding."
    )
    assert text == "healthy dog food good digestion also good small puppy dog eats require amount every feeding"


def test_clean_csv():
    assert (
        pre_processed_df["cleaned_text"][1] == "pleased natural balance dog food dog issue dog food past someone "
        "recommend natural balance grain free since possible allergic grain since switch issue "
        "also helpful different kibble size size dog"
    )


def test_add_features():
    assert feature_engineered_df["lowercase_count"][22] == 351
    assert feature_engineered_df["uppercase_count"][22] == 2
    assert feature_engineered_df["uppercase_words"][22] == "GOT, KIDDING"
    assert feature_engineered_df["uppercase_ratio"][22] == 0.005525
    assert feature_engineered_df["punc_count"][22] == 60
    assert feature_engineered_df["num_digits"][22] == 4
    assert feature_engineered_df["num_verbs"][22] == 21
    assert feature_engineered_df["num_nouns"][22] == 57
    assert feature_engineered_df["num_tokens_cleaned"][22] == 134
    assert feature_engineered_df["num_tokens_raw"][22] == 362
    assert feature_engineered_df["num_words_misspelled"][22] == 12
    assert feature_engineered_df["polarity"][22] == 0.9956
    assert feature_engineered_df["subjectivity"][22] == 0.489659
    assert feature_engineered_df["num_pos_words"][22] == 3
    assert feature_engineered_df["num_neg_words"][22] == 0
