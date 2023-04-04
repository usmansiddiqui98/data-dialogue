import re

import contractions
import nltk
import pandas as pd
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# necessary package downloads
nltk.download("averaged_perceptron_tagger")
nltk.download("punkt")
nltk.download("wordnet")

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
    "did",
    "do",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
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
    "own",
    "each",
    "every",
    "any",
    "all",
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
    "how",
    "anyway",
    "however",
    "just",
    "my",
]

stopwords = list(stopwords)


class Preprocessor:
    def __init__(self, dirty_file_path):
        self.dirty_file_path = dirty_file_path
        self.dirty_df = pd.read_csv(dirty_file_path)
        self.clean_df = None

    @staticmethod
    def clean_sentence(sentence, stop_words):  # takes in single string, returns a cleaned string
        stemmer = PorterStemmer()
        sentence = re.sub(r"<.*?>|Length::\d+:\d+Mins", " ", sentence)  # remove tags
        sentence = contractions.fix(sentence)  # resolve contractions
        sentence = re.sub(r"[^a-zA-Z\s]", " ", sentence)  # remove digits and special characters
        words = word_tokenize(sentence)  # tokenize
        words = [word.lower() for word in words if word.isalpha()]  # change to lower case and remove punctuations
        words = [word for word in words if word not in stop_words]  # remove stop words
        stemmed_sentence = [stemmer.stem(word) for word in words]
        stemmed_sentence = " ".join(stemmed_sentence)
        return stemmed_sentence

    def add_cleaned_text_512(self):
        def truncate_to_512(sentence):
            words = word_tokenize(sentence)
            pos_tagged = nltk.pos_tag(words)
            nouns_adjectives = [word for word, tag in pos_tagged if tag.startswith("N") or tag.startswith("J")]

            truncated = (
                words[:512] if len(nouns_adjectives) >= 512 else nouns_adjectives + words[len(nouns_adjectives) : 512]
            )
            return " ".join(truncated)

        self.clean_df["cleaned_text_512"] = self.clean_df["cleaned_text"].apply(truncate_to_512)

    def clean_csv(self):
        new_df = self.dirty_df.copy()
        new_df["Cleaned Text"] = new_df["Review"].apply(lambda x: Preprocessor.clean_sentence(x, stopwords))
        # lower case all column names
        new_df.columns = [x.lower().replace(" ", "_") for x in new_df.columns]
        self.clean_df = new_df
