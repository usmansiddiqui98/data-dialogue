import gensim
import spacy
from gensim import corpora
from nltk.tokenize import word_tokenize


class LDAGensim:
    def __init__(self, df, tags=None):
        self.df = df
        self.tags = tags
        self.id2word = None
        self.corpus = None
        self.lda_model = None

    @staticmethod
    def lemmatization(texts, tags):
        output = []
        nlp = spacy.load("en_core_web_sm")
        for sent in texts:
            doc = nlp(" ".join(sent))
            output.append([token.lemma_ for token in doc if token.pos_ in tags])
        return output

    def print_topics(self, num_topics, passes=10):
        if not self.tags:
            reviews = [word_tokenize(review) for review in self.df["cleaned_text"]]
        else:
            reviews = self.lemmatization([word_tokenize(review) for review in self.df["cleaned_text"]], tags=self.tags)
        self.id2word = corpora.Dictionary(reviews)
        self.corpus = [self.id2word.doc2bow(tokens) for tokens in reviews]
        self.lda_model = gensim.models.ldamodel.LdaModel(
            corpus=self.corpus, id2word=self.id2word, num_topics=num_topics, passes=passes, random_state=4263
        )
        for element in self.lda_model.print_topics():
            print("Topic " + str(element[0]))
            print(element[1])
