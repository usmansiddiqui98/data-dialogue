import string

from nltk.tokenize import word_tokenize


# Define function to count number of lowercase
def count_lower(sentence):
    words = word_tokenize(sentence)
    count = 0
    for word in words:
        if not word.isupper():  # eg Real is considered lowercase
            count += 1
    return count


# Define function to count number of uppercase
def count_upper(sentence):
    words = word_tokenize(sentence)
    count = 0
    for word in words:
        if word.isupper():
            if len(word) > 1:  # exclude one letter words eg 'I'
                count += 1
    return count


# Define function to list uppercase words
def uppercase_list(sentence):
    words = word_tokenize(sentence)
    uppercase = []
    for word in words:
        if word.isupper():
            if len(word) > 1:  # exclude one letter words eg 'I'
                uppercase.append(word)
    uppercase = ", ".join(uppercase)
    return uppercase


# Define function to get uppercase:total tokens ratio
def uppercase_ratio(sentence):
    words = word_tokenize(sentence)
    count = 0
    for word in words:
        if word.isupper():
            if len(word) > 1:  # exclude 'I'
                count += 1
    ratio = count / len(words)
    return ratio


# Define function to count number of punctuations
def count_punc(sentence):
    words = word_tokenize(sentence)
    count = 0
    for word in words:
        if word in string.punctuation:
            count += 1
    return count


def add_features(df):
    new_df = df.copy()
    new_df["Lowercase Count"] = new_df["text"].apply(count_lower)
    new_df["Uppercase Count"] = new_df["text"].apply(count_upper)
    new_df["Uppercase Words"] = new_df["text"].apply(uppercase_list)
    new_df["Uppercase Ratio"] = new_df["text"].apply(uppercase_ratio)
    new_df["Punc Count"] = new_df["text"].apply(count_punc)
    return new_df
