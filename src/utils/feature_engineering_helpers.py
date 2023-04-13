import string

import contractions
import nltk
from spellchecker import SpellChecker
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

nltk.download("averaged_perceptron_tagger", quiet=True)


def num_typos(sentence):
    """
    Count the number of typos in a sentence.

    Parameters
    ----------
    sentence : str
        The sentence to be checked for typos.

    Returns
    -------
    int
        The number of typos found in the sentence.
    """
    # remove punctuations
    s = sentence.translate(str.maketrans("", "", string.punctuation))
    # expand contractions
    expanded_words = []
    for word in s.split():
        expanded_words.append(contractions.fix(word))  # using contractions.fix to expand the shortened words
    spell = SpellChecker()
    amount_miss = len(list(spell.unknown(expanded_words)))
    return amount_miss


def compound_polarity_score(sentence):
    """
        Calculate the compound polarity score of a sentence using VaderSentiment.

        Parameters
        ----------
        sentence : str
            The sentence to be analyzed.

        Returns
        -------
        float
            The compound polarity score of the sentence.
        """
    sid_obj = SentimentIntensityAnalyzer()
    score = sid_obj.polarity_scores(sentence)["compound"]
    return score


def get_subjectivity(sentence):
    """
        Calculate the subjectivity score of a sentence using TextBlob.

        Parameters
        ----------
        sentence : str
            The sentence to be analyzed.

        Returns
        -------
        float
            The subjectivity score of the sentence.
        """
    return round(TextBlob(sentence).sentiment.subjectivity, 6)


def count_pos_neg_neutral(sentence):
    """
       Count the number of positive, negative, and neutral words in a sentence using VaderSentiment.

       Parameters
       ----------
       sentence : str
           The sentence to be analyzed.

       Returns
       -------
       list
           A list of three integers representing the count of positive, negative, and neutral words in the sentence.
       """

    text_split = sentence.split()
    sid = SentimentIntensityAnalyzer()
    pos_word_list = []
    neu_word_list = []
    neg_word_list = []

    for word in text_split:
        if (sid.polarity_scores(word)["compound"]) >= 0.5:
            pos_word_list.append(word)
        elif (sid.polarity_scores(word)["compound"]) <= -0.5:
            neg_word_list.append(word)
        else:
            neu_word_list.append(word)
    return [len(pos_word_list), len(neg_word_list), len(neu_word_list)]

    # Define function to count number of lowercase


def count_lower(sentence):
    """
       Count the number of lowercase words in a sentence.

       Parameters
       ----------
       sentence : str
           The sentence to be analyzed.

       Returns
       -------
       int
           The number of lowercase words in the sentence.
       """
    words = nltk.word_tokenize(sentence)
    count = 0
    for word in words:
        if not word.isupper():  # eg Real is considered lowercase
            count += 1
    return count


# Define function to count number of uppercase
def count_upper(sentence):
    """
        Count the number of uppercase words in a sentence.

        Parameters
        ----------
        sentence : str
            The sentence to be analyzed.

        Returns
        -------
        int
            The number of uppercase words in the sentence.
        """
    words = nltk.word_tokenize(sentence)
    count = 0
    for word in words:
        if word.isupper():
            if len(word) > 1:  # exclude one letter words eg 'I'
                count += 1
    return count


# Define function to list uppercase words
def uppercase_list(sentence):
    """
    Get a list of uppercase words in a sentence.

    Parameters
    ----------
    sentence : str
        Input sentence.

    Returns
    -------
    str
        A comma-separated string of uppercase words in the sentence.
    """
    words = nltk.word_tokenize(sentence)
    uppercase = []
    for word in words:
        if word.isupper():
            if len(word) > 1:  # exclude one letter words eg 'I'
                uppercase.append(word)
    uppercase = ", ".join(uppercase)
    return uppercase


# Define function to get uppercase:total tokens ratio
def uppercase_ratio(sentence):
    """
        Get the ratio of uppercase words to total tokens in a sentence.

        Parameters
        ----------
        sentence : str
            Input sentence.

        Returns
        -------
        float
            The ratio of uppercase words to total tokens, rounded to 6 decimal places.
        """
    words = nltk.word_tokenize(sentence)
    count = 0
    for word in words:
        if word.isupper():
            if len(word) > 1:  # exclude 'I'
                count += 1
    ratio = count / len(words)
    ratio = round(ratio, 6)
    return ratio


# Define function to count number of punctuations
def count_punc(sentence):
    """
       Count the number of punctuations in a sentence.

       Parameters
       ----------
       sentence : str
           Input sentence.

       Returns
       -------
       int
           The number of punctuations in the sentence.
       """
    words = nltk.word_tokenize(sentence)
    count = 0
    for word in words:
        if word in string.punctuation:
            count += 1
    return count


def pos_tags(sentence):
    """
       Get the part-of-speech (POS) tags of the words in a sentence.

       Parameters
       ----------
       sentence : str
           Input sentence.

       Returns
       -------
       list of tuples
           A list of tuples, where each tuple contains a word and its corresponding POS tag.
       """
    tokenized_sentence = nltk.word_tokenize(sentence.lower())
    tagged = nltk.pos_tag(tokenized_sentence)
    return tagged
