import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer


class NMFModel:
    """
    A class to represent a NMF (Non-negative matrix factorization) model for topic modeling.

    Attributes
    ----------
    df : pd.DataFrame
        The data that the model will be trained on.
    n_components : int, optional
        The number of topics that will be generated, default is 10.
    max_features : int, optional
        The maximum number of features that will be used to represent the text, default is 1000.
    max_df : float, optional
        The maximum document frequency of the words that will be used to represent the text, default is 0.5.
    vectorizer : TfidfVectorizer
        A vectorizer used to convert text to a matrix of TF-IDF features.
    model : NMF
        A model for topic modeling using non-negative matrix factorization.
    topic_df : pd.DataFrame
        A dataframe of top terms for each topic.
    labels : list
        A list of labels for each topic.
    X : array-like or sparse matrix, shape (n_samples, n_features)
        The matrix of the TF-IDF features for the text.
    document_weights : array-like, shape (n_samples, n_components)
        The weights of each document on each topic.
    label_df : pd.DataFrame
        A dataframe with the original data and topic labels.

    """

    def __init__(self, df, n_components=10, max_features=1000, max_df=0.5):
        """
        Parameters
        ----------
        df : pd.DataFrame
            The data that the model will be trained on.
        n_components : int, optional
            The number of topics that will be generated, default is 10.
        max_features : int, optional
            The maximum number of features that will be used to represent the text, default is 1000.
        max_df : float, optional
            The maximum document frequency of the words that will be used to represent the text, default is 0.5.
        """
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
        self.label_df = None

    # Define method to fit and transform the data
    def fit_transform(self):
        """
        Fit and transform the data to get the document weights.
        """
        self.vectorizer = TfidfVectorizer(
            stop_words="english", max_features=self.max_features, max_df=self.max_df, smooth_idf=True
        )
        self.X = self.vectorizer.fit_transform(self.df["cleaned_text"])
        self.model = NMF(n_components=self.n_components, random_state=5)
        self.model.fit(self.X)
        self.document_weights = self.model.transform(self.vectorizer.transform(self.df["cleaned_text"]))

    # Define method to get the top terms for each topic
    def get_topic_terms(self, n_top_words=10):
        """
        Get the top terms for each topic.

        Parameters:
        -----------
        n_top_words : int, default=10
            The number of top words to display for each topic.

        Returns:
        --------
        topics_dict : dict
            A dictionary where each key corresponds to a topic index, and each value is another dictionary
            containing the top words/terms for that topic and their corresponding weights.
        """
        words = self.vectorizer.get_feature_names_out()
        components_df = pd.DataFrame(self.model.components_, columns=words)
        topics_dict = {}
        for topic in range(components_df.shape[0]):
            tmp = components_df.iloc[topic]
            topic_dict = {}
            for i, val in tmp.nlargest(n_top_words).items():
                topic_dict[i] = val
            topics_dict[topic] = topic_dict
        return topics_dict

    # Define method to label the topics for each data point
    def get_labels(self, n_top_words=5):
        """
        Label the topics for each data point in the input DataFrame.

        Parameters
        ----------
        n_top_words : int, optional
            The number of top words to use in the topic description. Defaults to 5.

        Returns
        -------
        pandas.DataFrame
            A new DataFrame with an additional column `Topic_idx` that contains
            the index of the topic with the highest weight for each data point, and
            another additional column `Top_Topic_Terms` that contains the
            description of the top words in the corresponding topic.

        Raises
        ------
        ValueError
            If `fit_transform` method hasn't been called yet.

        Notes
        -----
        This method uses the `document_weights` attribute that is computed by the
        `fit_transform` method. If the `fit_transform` method hasn't been called
        yet, a `ValueError` is raised.

        The topic description for each data point is computed as the concatenation
        of the top `n_top_words` words with the highest weights in the corresponding
        topic. The resulting descriptions are stored in a new DataFrame, along with
        the corresponding topic index.

        """
        words = self.vectorizer.get_feature_names_out()
        topic_terms = {}
        for i, topic_vec in enumerate(self.model.components_):
            topic_descr = ""
            for fid in topic_vec.argsort()[-1 : -n_top_words - 1 : -1]:
                topic_descr = topic_descr + words[fid] + " "
            topic_terms[i] = topic_descr
        self.topic_df = pd.DataFrame({"Top_Topic_Terms": topic_terms})
        self.df["Topic_idx"] = self.document_weights.argmax(axis=1)
        self.label_df = pd.merge(self.df, self.topic_df, left_on="Topic_idx", right_index=True, how="left")
        return self.label_df
