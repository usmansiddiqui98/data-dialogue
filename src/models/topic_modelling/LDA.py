import gensim
import spacy
from gensim import corpora
from nltk.tokenize import word_tokenize


class LDAGensim:
    """
    A class for performing LDA topic modeling using the Gensim library.

    Attributes:
        df (pandas.DataFrame): The input dataframe containing text data to perform LSA.
        num_topics (int): The number of topics to extract (default: 10).
        tags (list): A list of part-of-speech tags to use for lemmatization (default: None).
        id2word (gensim.corpora.Dictionary): The Gensim dictionary object representing the corpus vocabulary.
        corpus (list): The Gensim corpus object representing the tokenized documents.
        lda_model (gensim.models.ldamodel.LdaModel): The Gensim LDA model object.

    Methods:
        lemmatization(texts, tags): Lemmatizes input texts using spaCy.
        get_topics(passes): Perform Latent Dirichlet Allocation on the input text data and return a dictionary of topics.
    """

    def __init__(self, df, num_topics=10, tags=None):
        """
        Initializes the LDAGensim object.

        Args:
            df (pandas.DataFrame): The input data as a pandas DataFrame.
            num_topics (int): The number of topics to extract (default: 10).
            tags (list): A list of part-of-speech tags to use for lemmatization (default: None).
        """
        self.df = df
        self.num_topics = num_topics
        self.tags = tags
        self.id2word = None
        self.corpus = None
        self.lda_model = None

    @staticmethod
    def lemmatization(texts, tags):
        """
        Lemmatizes input texts using spaCy.

        Args:
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

    def get_topics(self, passes=10):
        """
        Extracts LDA topics from the input data using the Gensim library.

        Args:
            passes (int): The number of passes to use for training the LDA model (default: 10).

        Returns:
            dict: A dictionary of topic words and their weights for each topic.
        """
        if not self.tags:
            reviews = [word_tokenize(review) for review in self.df["cleaned_text"]]
        else:
            reviews = self.lemmatization([word_tokenize(review) for review in self.df["cleaned_text"]], tags=self.tags)
        self.id2word = corpora.Dictionary(reviews)
        self.corpus = [self.id2word.doc2bow(tokens) for tokens in reviews]
        self.lda_model = gensim.models.ldamodel.LdaModel(
            corpus=self.corpus, id2word=self.id2word, num_topics=self.num_topics, passes=passes, random_state=4263
        )
        topics = {}
        for topic, words in self.lda_model.show_topics(num_topics=self.num_topics, formatted=False):
            topics[topic] = {}
            for word, weight in words:
                topics[topic][word] = weight

        return topics
