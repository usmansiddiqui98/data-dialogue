import spacy
from nltk.tokenize import word_tokenize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


class LSAModel:
    def __init__(self, df, num_topics=10, tags=None):
        self.df = df
        self.num_topics = num_topics
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

    def get_topics(self):
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
