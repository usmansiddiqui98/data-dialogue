import nltk
from pandarallel import pandarallel

from src.utils.feature_engineering_helpers import (
    compound_polarity_score,
    count_lower,
    count_pos_neg_neutral,
    count_punc,
    count_upper,
    get_subjectivity,
    num_typos,
    pos_tags,
    uppercase_ratio,
)


class FeatureEngineer:
    """Class for performing feature engineering on pre-processed text data.

    Parameters
    ----------
    pre_processed_df : pandas.DataFrame
        The pre-processed text data.

    Attributes
    ----------
    pre_processed_df : pandas.DataFrame
        The pre-processed text data.
    feature_engineered_df : pandas.DataFrame or None
        The feature-engineered data. Initialized to None.

    """
    pandarallel.initialize(progress_bar=False, verbose=0)

    def __init__(self, pre_processed_df):
        self.pre_processed_df = pre_processed_df
        self.feature_engineered_df = None

    @staticmethod
    def pos_tag_count(pre_processed_df):
        """Count the number of verbs, nouns, and cardinal digits in each review.

               Parameters
               ----------
               pre_processed_df : pandas.DataFrame
                   The pre-processed text data.

               Returns
               -------
               pandas.DataFrame
                   The data with three new columns, indicating the number of verbs, nouns, and cardinal digits in each review.

               """
        # extract verbs
        df = pre_processed_df.copy()
        df["verbs"] = df["pos_tags"].parallel_apply(
            lambda x: [word for word, tag in x if tag in ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]]
        )
        # extract nouns
        df["nouns"] = df["pos_tags"].parallel_apply(
            lambda x: [word for word, tag in x if tag in ["NN", "NNS", "NNP", "NNPS"]]
        )
        # extract the numbers or cardinal digits from the text
        df["cardinal_digits"] = df["pos_tags"].parallel_apply(lambda x: [word for word, tag in x if tag in ["CD"]])

        df["num_digits"] = df["cardinal_digits"].str.len()
        df["num_verbs"] = df["verbs"].str.len()
        df["num_nouns"] = df["nouns"].str.len()

        to_drop = ["pos_tags", "nouns", "verbs", "cardinal_digits"]
        df.drop(to_drop, axis=1, inplace=True)
        print("[FE] finished pos tag count...")
        return df

    @staticmethod
    def tokenized_untokenized_count(df):
        """Count the number of tokens in the cleaned and raw text of each review.

        Parameters
        ----------
        df : pandas.DataFrame
            The data containing the pre-processed and cleaned text data.

        Returns
        -------
        pandas.DataFrame
            The data with two new columns, indicating the number of tokens in the cleaned and raw text of each review.

        """
        df["num_tokens_cleaned"] = df["cleaned_text"].str.lower().parallel_apply(nltk.word_tokenize).str.len()
        df["num_tokens_raw"] = df["text"].str.lower().parallel_apply(nltk.word_tokenize).str.len()
        print("[FE] finished tokenized_untokenized count...")
        return df

    @staticmethod
    def add_pos_neg_count(df):
        """
        Adds columns for the number of positive, negative, and neutral words of each review to the given DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            A DataFrame.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with added columns for the number of positive, negative, and neutral words.
        """

        df["num_pos_neg_neutral_words"] = df["cleaned_text"].parallel_apply(count_pos_neg_neutral)
        df["num_pos_words"] = df["num_pos_neg_neutral_words"].str[0]
        df["num_neg_words"] = df["num_pos_neg_neutral_words"].str[1]
        df.drop(["num_pos_neg_neutral_words"], axis=1, inplace=True)
        print("[FE] finished pos neg count...")
        return df

    # main function that adds feature to the df
    def add_features(self):
        """
        Adds new features to the pre-processed DataFrame and creates a new feature-engineered DataFrame.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with added features.
        """
        new_df = self.pre_processed_df.copy()
        new_df["Lowercase Count"] = new_df["text"].parallel_apply(count_lower)
        print("[FE] finished lowercase count...")
        new_df["Uppercase Count"] = new_df["text"].parallel_apply(count_upper)
        print("[FE] finished uppercase count...")
        new_df["Uppercase Ratio"] = new_df["text"].parallel_apply(uppercase_ratio)
        print("[FE] finished uppercase ratio...")
        new_df["Punc Count"] = new_df["text"].parallel_apply(count_punc)
        print("[FE] finished punc count...")
        new_df["pos_tags"] = new_df["cleaned_text"].parallel_apply(pos_tags)
        print("[FE] finished pos tags...")
        new_df = FeatureEngineer.pos_tag_count(new_df)
        new_df = FeatureEngineer.tokenized_untokenized_count(new_df)
        new_df["num_words_misspelled"] = new_df["text"].parallel_apply(num_typos)
        print("[FE] finished num words misspelled...")
        new_df["polarity"] = new_df["text"].parallel_apply(compound_polarity_score)
        print("[FE] finished polarity...")
        new_df["subjectivity"] = new_df["text"].parallel_apply(get_subjectivity)
        print("[FE] finished subjectivity...")
        new_df = self.add_pos_neg_count(new_df)
        # lower case all column names
        new_df.columns = [x.lower().replace(" ", "_") for x in new_df.columns]
        self.feature_engineered_df = new_df
