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
print(feature_engineered_df.iloc[22, :])

stopwords = [
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "hence",
    "her",
    "hers",
    "him",
    "his",
    "if",
    "in",
    "is",
    "it",
    "its",
    "may",
    "might",
    "must",
    "of",
    "on",
    "or",
    "shall",
    "should",
    "since",
    "so",
    "some",
    "such",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "to",
    "was",
    "we",
    "were",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "whose",
    "will",
    "with",
    "would",
    "yet",
    "you",
    "your",
    "yours",
    "about",
    "above",
    "across",
    "after",
    "against",
    "along",
    "among",
    "around",
    "before",
    "behind",
    "below",
    "beneath",
    "beside",
    "between",
    "beyond",
    "during",
    "inside",
    "into",
    "near",
    "outside",
    "over",
    "through",
    "under",
    "upon",
    "within",
    "without",
    "been",
    "having",
    "once",
    "other",
    "until",
    "more",
    "less",
    "own",
    "also",
    "each",
    "every",
    "any",
    "all",
    "some",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "many",
    "several",
    "few",
    "less",
    "more",
    "most",
    "several",
    "how",
    "anyway",
    "however",
    "just",
    "quite",
]


def test_clean():
    text = Preprocessor.clean_sentence(
        "This is a very healthy dog food. Good for their digestion. Also good for small puppies. "
        "My dog eats her required amount at every feeding.",
        stopwords
    )
    assert text == 'very healthy dog food good digestion good small puppy my dog eats require amount feeding'


def test_clean_csv():
    assert (
        pre_processed_df["cleaned_text"][1] == 'very pleased natural balance dog food our dog issue dog food past '
                                               'someone recommend natural balance grain free possible allergic grain '
                                               'switch not issue helpful different kibble size large small sized dog'
    )


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
    assert feature_engineered_df["polarity"][22] == 0.9959
    assert feature_engineered_df["subjectivity"][22] == 0.488526
    assert feature_engineered_df["num_pos_words"][22] == 3
    assert feature_engineered_df["num_neg_words"][22] == 0
