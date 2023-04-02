import spacy

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize


class LSAModel:
    def __init__(self, df, tags=None):
        self.df = df
        self.tags = tags
        self.vectorizer = None
        self.X = None
        self.svd_model = None
        self.terms = None

    @staticmethod
    def lemmatization(texts, tags):
        output = []
        nlp = spacy.load("en_core_web_sm")
        for sent in texts:
            doc = nlp(" ".join(sent))
            output.append([token.lemma_ for token in doc if token.pos_ in tags])
        return output

    def print_topics(self, num_topics):
        if not self.tags:
            reviews = [word_tokenize(review) for review in self.df["cleaned_text"]]
        else:
            reviews = self.lemmatization([word_tokenize(review) for review in self.df["cleaned_text"]], tags=self.tags)

        reviews = [item for sublist in reviews for item in sublist]
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=1000, max_df=0.5, smooth_idf=True)
        self.X = self.vectorizer.fit_transform(reviews)
        self.svd_model = TruncatedSVD(n_components=num_topics, algorithm="randomized", n_iter=100, random_state=122)
        self.svd_model.fit(self.X)
        self.terms = self.vectorizer.get_feature_names_out()

        for i, comp in enumerate(self.svd_model.components_):
            terms_comp = zip(self.terms, comp)
            sorted_terms = sorted(terms_comp, key=lambda x: x[1], reverse=True)[:num_topics]
            print("Topic " + str(i) + ": ")
            print(sorted_terms)
