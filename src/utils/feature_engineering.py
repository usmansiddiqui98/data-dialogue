import string
import contractions
import nltk
from spellchecker import SpellChecker
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

nltk.download("averaged_perceptron_tagger")

def num_typos(sentence):
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
    sid_obj = SentimentIntensityAnalyzer()
    score = sid_obj.polarity_scores(sentence)["compound"]
    return score


def get_subjectivity(sentence):
    return round(TextBlob(sentence).sentiment.subjectivity, 6)


def count_pos_neg_neutral(sentence):
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
    words = nltk.word_tokenize(sentence)
    count = 0
    for word in words:
        if not word.isupper():  # eg Real is considered lowercase
            count += 1
    return count


# Define function to count number of uppercase
def count_upper(sentence):
    words = nltk.word_tokenize(sentence)
    count = 0
    for word in words:
        if word.isupper():
            if len(word) > 1:  # exclude one letter words eg 'I'
                count += 1
    return count


# Define function to list uppercase words
def uppercase_list(sentence):
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
    words = nltk.word_tokenize(sentence)
    count = 0
    for word in words:
        if word in string.punctuation:
            count += 1
    return count


def pos_tags(sentence):
    # print("CALLING POS TAGS METHOD ON SENTENCE")
    tokenized_sentence = nltk.word_tokenize(sentence.lower())
    tagged = nltk.pos_tag(tokenized_sentence)
    return tagged
