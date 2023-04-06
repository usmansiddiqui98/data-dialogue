import re

import contractions
import nltk
import pandas as pd
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
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
    "i",
]

stopwords = list(stopwords)


class Preprocessor:
    def __init__(self, dirty_df):
        self.dirty_df = dirty_df

    @staticmethod
    def clean_sentence(sentence, stop_words):  # takes in single string, returns a cleaned string
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
        new_df["cleaned_text"] = new_df["Text"].apply(lambda x: Preprocessor.clean_sentence(x, stopwords))
        new_df["Sentiment"] = new_df["Sentiment"].apply(lambda x: 1 if x == "positive" else 0)
        # lower case all column names
        new_df.columns = [x.lower().replace(" ", "_") for x in new_df.columns]
        self.clean_df = new_df

    def clean_test_csv(self):
        new_df = self.dirty_df.copy()
        new_df["cleaned_text"] = new_df["Text"].apply(lambda x: Preprocessor.clean_sentence(x, stopwords))
        # lower case all column names
        new_df.columns = [x.lower().replace(" ", "_") for x in new_df.columns]
        self.clean_df = new_df
