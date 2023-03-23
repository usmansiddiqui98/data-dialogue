import string

import contractions
import nltk
from spellchecker import SpellChecker
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

nltk.download("averaged_perceptron_tagger")


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
    tokenized_sentence = nltk.word_tokenize(sentence.lower())
    tagged = nltk.pos_tag(tokenized_sentence)
    return tagged


def pos_tag_count(df):
    # extract verbs
    df["verbs"] = df["pos_tags"].apply(
        lambda x: [word for word, tag in x if tag in ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]]
    )
    # extract nouns
    df["nouns"] = df["pos_tags"].apply(lambda x: [word for word, tag in x if tag in ["NN", "NNS", "NNP", "NNPS"]])
    # extract the numbers or cardinal digits from the text
    df["cardinal_digits"] = df["pos_tags"].apply(lambda x: [word for word, tag in x if tag in ["CD"]])

    df["num_digits"] = df["cardinal_digits"].str.len()
    df["num_verbs"] = df["verbs"].str.len()
    df["num_nouns"] = df["nouns"].str.len()

    to_drop = ["pos_tags", "nouns", "verbs", "cardinal_digits"]
    df.drop(to_drop, axis=1, inplace=True)
    return df


def tokenized_untokenized_count(df):
    df["num_tokens_cleaned"] = df["cleaned_text"].str.lower().apply(nltk.word_tokenize).str.len()
    df["num_tokens_raw"] = df["text"].str.lower().apply(nltk.word_tokenize).str.len()
    return df


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


def getSubjectivity(sentence):
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


def add_pos_neg_count(df):
    df["num_pos_neg_neutral_words"] = df["cleaned_text"].apply(count_pos_neg_neutral)
    df["num_pos_words"] = df["num_pos_neg_neutral_words"].str[0]
    df["num_neg_words"] = df["num_pos_neg_neutral_words"].str[1]
    df.drop(["num_pos_neg_neutral_words"], axis=1, inplace=True)
    return df


def add_features(df):
    new_df = df.copy()
    new_df["Lowercase Count"] = new_df["text"].apply(count_lower)
    new_df["Uppercase Count"] = new_df["text"].apply(count_upper)
    new_df["Uppercase Words"] = new_df["text"].apply(uppercase_list)
    new_df["Uppercase Ratio"] = new_df["text"].apply(uppercase_ratio)
    new_df["Punc Count"] = new_df["text"].apply(count_punc)
    new_df["pos_tags"] = new_df["cleaned_text"].apply(pos_tags)
    new_df = pos_tag_count(new_df)
    new_df = tokenized_untokenized_count(new_df)
    new_df["num_words_misspelled"] = new_df["text"].apply(num_typos)
    new_df["polarity"] = new_df["cleaned_text"].apply(compound_polarity_score)
    new_df["subjectivity"] = new_df["cleaned_text"].apply(getSubjectivity)
    new_df = add_pos_neg_count(new_df)
    return new_df
