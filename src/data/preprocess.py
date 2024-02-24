import re

import contractions
import nltk
import pandas as pd
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from pandarallel import pandarallel

# necessary package downloads
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)

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
    # "may",
    # "might",
    # "must",
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
    "i",
]

stopwords = list(stopwords)


class Preprocessor:
    """
    A class for preprocessing text data.

    Attributes:
    -----------

    dirty_df (pandas DataFrame):
        A DataFrame containing the dirty text data to be cleaned.

    clean_df (pandas DataFrame):
        A DataFrame containing the cleaned text data.

    """

    pandarallel.initialize(progress_bar=False, verbose=0)

    def __init__(self, dirty_df):
        """
        Initialize the Preprocessor object.

        Parameters:
            dirty_df (pandas DataFrame): A DataFrame containing the dirty text data to be cleaned.
        """
        self.dirty_df = dirty_df

    @staticmethod
    def clean_sentence(sentence, stop_words):
        """
        Clean a single string by removing tags, resolving contractions, removing digits and special characters,
        tokenizing, changing to lower case and removing punctuations, removing stop words, finding the POS tag
        for each token, and lemmatizing each token.

        Parameters:
            sentence (str):A single string to be cleaned.

            stop_words (list): A list of stop words to be removed from the string.

        Returns:
            str: The cleaned string.
        """

        def pos_tagger(nltk_tag):
            if nltk_tag.startswith("J"):
                return wordnet.ADJ
            elif nltk_tag.startswith("V"):
                return wordnet.VERB
            elif nltk_tag.startswith("N"):
                return wordnet.NOUN
            elif nltk_tag.startswith("R"):
                return wordnet.ADV
            else:
                return None

        lemmatizer = WordNetLemmatizer()
        sentence = re.sub(r"<.*?>|Length::\d+:\d+Mins", " ", sentence)  # remove tags
        sentence = contractions.fix(sentence)  # resolve contractions
        sentence = re.sub(r"[^a-zA-Z\s]", " ", sentence)  # remove digits and special characters
        words = word_tokenize(sentence)  # tokenize
        words = [word.lower() for word in words if word.isalpha()]  # change to lower case and remove punctuations
        words = [word for word in words if word not in stop_words]  # remove stop words
        pos_tagged = nltk.pos_tag(words)  # find the POS tag for each token
        # we use our own pos_tagger function to make things simpler to understand.
        wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
        lemmatized_sentence = []
        for word, tag in wordnet_tagged:
            if tag is None:
                # if there is no available tag, append the token as is
                lemmatized_sentence.append(word)
            else:
                # else use the tag to lemmatize the token
                lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
        lemmatized_sentence = " ".join(lemmatized_sentence)
        return lemmatized_sentence
        # return stemmed_sentence

    def truncate_to_512(sentence):
        """
        Truncate a string to contain the first 512 nouns and adjectives.

        Parameters:
            sentence (str): A string to be truncated.

        Returns:
            str: The truncated string.
        """
        words = word_tokenize(sentence)
        pos_tagged = nltk.pos_tag(words)
        nouns_adjectives = [word for word, tag in pos_tagged if tag.startswith("N") or tag.startswith("J")]
        remaining = [word for word in words if word not in nouns_adjectives]
        if len(nouns_adjectives) <= 512:
            combined = nouns_adjectives + remaining[: 512 - len(nouns_adjectives)]
        else:
            combined = nouns_adjectives[:512]
        return " ".join(combined)

    def clean_csv(self):
        """
        Clean the text data in the dirty_df DataFrame, adding a new column of cleaned text and a new column of
        sentiment labels.

        Parameters:
            self (object): Object of Preprocessor class with 'dirty_df' attribute.

        Returns:
            None

        The 'dirty_df' attribute is copied to 'clean_df' attribute with cleaned 'Text' column and
        binary 'Sentiment' column.
        """
        new_df = self.dirty_df.copy()
        new_df["cleaned_transcript"] = new_df["transcript"].parallel_apply(lambda x: Preprocessor.clean_sentence(x, stopwords))
        new_df["Had Timing Objection"] = new_df["Had Timing Objection"].parallel_apply(lambda x: 1 if x == "True" else 0)
        # lower case all column names
        new_df.columns = [x.lower().replace(" ", "_") for x in new_df.columns]
        self.clean_df = new_df

    def clean_test_csv(self):
        """
        Clean the text data in the dirty_df DataFrame, adding a new column of cleaned text.

        Parameters:
            self (object): Object of Preprocessor class with 'dirty_df' attribute.

        Returns:
            None

        The 'dirty_df' attribute is copied to 'clean_df' attribute with cleaned 'Text' column.
        """
        new_df = self.dirty_df.copy()
        new_df["cleaned_transcript"] = new_df["transcript"].parallel_apply(lambda x: Preprocessor.clean_sentence(x, stopwords))
        # lower case all column names
        new_df.columns = [x.lower().replace(" ", "_") for x in new_df.columns]
        self.clean_df = new_df
