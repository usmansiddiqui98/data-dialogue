import spacy
from nltk.tokenize import word_tokenize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


class LSAModel:
    """
    A class to represent Latent Semantic Analysis model.

    Attributes
    ----------

    df (pandas.DataFrame):
        The input dataframe containing text data to perform LSA.
    num_topics (int, optional, default: 10):
        The number of topics to extract.
    tags (list, optional, default: None):
        A list of part-of-speech tags to use for lemmatization.
    vectorizer (sklearn.feature_extraction.text.TfidfVectorizer):
        The TfidfVectorizer object used to convert text data to a matrix of TF-IDF features.
    X (scipy.sparse.csr_matrix):
        The matrix of TF-IDF features.
    svd_model (sklearn.decomposition.TruncatedSVD):
        The TruncatedSVD object used to perform dimensionality reduction on the matrix of TF-IDF features.
    terms (list):
        The list of terms (i.e., words) extracted from the input text data.
    """

    def __init__(self, df, num_topics=10, tags=None):
        """
        Initializes the LDAGensim object.

        Parameters:
            df (pandas.DataFrame): The input data as a pandas DataFrame.
            num_topics (int, optional, default: 10): The number of topics to extract.
            tags (list, optional, default: None): A list of part-of-speech tags to use for lemmatization.
        """
        self.df = df
        self.num_topics = num_topics
        self.tags = tags
        self.vectorizer = None
        self.X = None
        self.svd_model = None
        self.terms = None

    @staticmethod
    def lemmatization(texts, tags):
        """
        Lemmatizes input texts using spaCy.

        Parameters:
            texts (list): A list of tokenized texts to lemmatize.
            tags (list): A list of part-of-speech tags to use for lemmatization.

        Returns:
            list: A list of lemmatized texts.
        """
        output = []
        nlp = spacy.load("en_core_web_sm")
        for sent in texts:
            doc = nlp(" ".join(sent))
            output.append([token.lemma_ for token in doc if token.pos_ in tags])
        return output

    def get_topics(self):
        """
        Perform LSA on the input text data.

        Returns:
            dict: A dictionary of topic words and their weights for each topic.
        """
        if not self.tags:
            reviews = [word_tokenize(review) for review in self.df["cleaned_text"]]
        else:
            reviews = self.lemmatization([word_tokenize(review) for review in self.df["cleaned_text"]], tags=self.tags)

        reviews = [item for sublist in reviews for item in sublist]
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=1000, max_df=0.5, smooth_idf=True)
        self.X = self.vectorizer.fit_transform(reviews)
        self.svd_model = TruncatedSVD(
            n_components=self.num_topics, algorithm="randomized", n_iter=100, random_state=122
        )
        self.svd_model.fit(self.X)
        self.terms = self.vectorizer.get_feature_names_out()

        topics = {}
        for i, comp in enumerate(self.svd_model.components_):
            topic = {}
            terms_comp = zip(self.terms, comp)
            sorted_terms = sorted(terms_comp, key=lambda x: x[1], reverse=True)[: self.num_topics]
            for term, weight in sorted_terms:
                topic[term] = weight
            topics[i] = topic

        return topics
