import re

import contractions
import nltk
import pandas as pd
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# necessary package downloads
nltk.download("averaged_perceptron_tagger")
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")


class Preprocessor:
    def __init__(self, dirty_file_path):
        self.dirty_file_path = dirty_file_path
        self.dirty_df = pd.read_csv(dirty_file_path)
        self.clean_df = None

    @staticmethod
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

    @staticmethod
    def clean_sentence(sentence):  # takes in single string, returns a cleaned string
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words("english"))

        sentence = re.sub(r"<.*?>|Length::\d+:\d+Mins", " ", sentence)  # remove tags
        sentence = contractions.fix(sentence)  # resolve contractions
        words = word_tokenize(sentence)  # tokenize
        words = [word.lower() for word in words if word.isalpha()]  # change to lower case and remove punctuations
        words = [word for word in words if word not in stop_words]  # remove stop words
        pos_tagged = nltk.pos_tag(words)  # find the POS tag for each token
        # we use our own pos_tagger function to make things simpler to understand.
        wordnet_tagged = list(map(lambda x: (x[0], Preprocessor.pos_tagger(x[1])), pos_tagged))
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

    def clean_csv(self):
        new_df = self.dirty_df.copy()
        new_df["Cleaned Text"] = new_df["Text"].apply(Preprocessor.clean_sentence)
        # lower case all column names
        new_df.columns = [x.lower().replace(" ", "_") for x in new_df.columns]
        self.clean_df = new_df
