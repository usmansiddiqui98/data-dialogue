import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


def clean(sentence: str):  # takes in single string, returns a cleaned string
    lemmatizer = nltk.WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    words = word_tokenize(sentence)  # tokenize
    words = [word.lower() for word in words if word.isalpha()]  # change to lower case and remove non-alphabetic tokens
    words = [lemmatizer.lemmatize(word, 'n') for word in words if word.isalpha()]  # lemmatize nouns
    words = [lemmatizer.lemmatize(word, 'v') for word in words if word.isalpha()]  # lemmatize verbs
    words = [lemmatizer.lemmatize(word, 'a') for word in words if word.isalpha()]  # lemmatize adjectives
    words = [word for word in words if not word in stop_words]  # remove stop words
    return " ".join(words)


def clean_df(df):  # takes in a pandas df with a 'Text' column, returns df with additional 'Cleaned Text' column
    new_df = df.copy()
    new_df['Cleaned Text'] = new_df['Text'].apply(clean)
    return new_df


def clean_csv(path):  # takes in a csv file path with a 'Text' column, returns df with additional 'Cleaned Text' column
    df = pd.read_csv(path)
    return clean_df(df)
