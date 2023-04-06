import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer


class NMFModel:
    def __init__(self, df, n_components=10, max_features=1000, max_df=0.5):
        self.df = df
        self.n_components = n_components
        self.max_features = max_features
        self.max_df = max_df
        self.vectorizer = None
        self.model = None
        self.topic_df = None
        self.labels = None
        self.X = None
        self.document_weights = None

    # Define method to fit and transform the data
    def fit_transform(self):
        self.vectorizer = TfidfVectorizer(stop_words='english',
                                          max_features=self.max_features,
                                          max_df=self.max_df,
                                          smooth_idf=True)
        self.X = self.vectorizer.fit_transform(self.df["cleaned_text"])
        self.model = NMF(n_components=self.n_components, random_state=5)
        self.model.fit(self.X)
        self.document_weights = self.model.transform(self.df["cleaned_text"])

    # Define method to get the top terms for each topic
    def get_topic_terms(self, n_top_words=10):
        words = self.vectorizer.get_feature_names_out()
        components_df = pd.DataFrame(self.model.components_, columns=words)
        for topic in range(components_df.shape[0]):
            tmp = components_df.iloc[topic]
            print(f'For topic {topic + 1} the words with the highest value are:')
            print(tmp.nlargest(n_top_words))
            print('\n')

    # Define method to label the topics for each data point
    def get_labels(self, n_top_words=5):
        words = self.vectorizer.get_feature_names()
        topic_terms = {}
        for i, topic_vec in enumerate(self.model.components_):
            topic_descr = ''
            for fid in topic_vec.argsort()[-1:-n_top_words - 1:-1]:
                topic_descr = topic_descr + words[fid] + " "
            topic_terms[i] = topic_descr
        self.topic_df = pd.DataFrame({'Top_Topic_Terms': topic_terms})
        self.labels = self.document_weights.argmax(axis=1)
        self.labels = pd.merge(self.labels.to_frame(name='Topic_idx'),
                               self.topic_df,
                               left_on='Topic_idx',
                               right_index=True,
                               how='left')
        return self.labels
