import os
import pickle

from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from umap import UMAP


class BertTopic:
    """
    A class for generating topic models using BERT embeddings.

    Attributes:
    -----------

    data (pandas.DataFrame):
        The input data as a pandas dataframe.
    preprocessor (Preprocessor):
        An instance of Preprocessor class used for text pre-processing.
    embeddings (numpy.ndarray):
        An array of BERT embeddings for each document.
    topics (list):
        A list of topic labels for each document.
    probabilities (numpy.ndarray):
        An array of probabilities for each topic in each document.
    topic_model (BERTopic):
        An instance of BERTopic model for generating topics.
    """

    def __init__(self, data):
        """
        Constructs all necessary attributes for the BertTopic object.

        Parameters:
            data (pandas.DataFrame):
                The input data as a pandas dataframe.
        """
        self.data = data
        self.embeddings = None
        self.topics = None
        self.probabilities = None
        self.topic_model = None

    def prepare_embeddings(self):
        """
        Generates BERT embeddings for input data and saves them to a file for future use.
        """
        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        embeddings_file = os.path.join(BASE_DIR, "data/embeddings/BERTopic_embeddings.pickle")

        if os.path.exists(embeddings_file):
            print("Loading existing embeddings...")
            with open(embeddings_file, "rb") as pkl:
                self.embeddings = pickle.load(pkl)
        else:
            print("Creating new embeddings...")
            sentence_model = SentenceTransformer("all-MiniLM-L12-v2")
            self.embeddings = sentence_model.encode(self.df["cleaned_text"], show_progress_bar=True)

            # save embeddings
            with open(embeddings_file, "wb") as pkl:
                pickle.dump(self.embeddings, pkl)

    def run_bertopic(self):
        """
        Runs BERTopic model to generate topics and probabilities.
        """
        umap_model = UMAP(n_neighbors=100, n_components=3, min_dist=0.0, metric="cosine", random_state=4263)
        hdbscan_model = HDBSCAN(min_cluster_size=50, min_samples=20, metric="euclidean", prediction_data=True)
        self.topic_model = BERTopic(
            hdbscan_model=hdbscan_model,
            umap_model=umap_model,
            language="english",
            calculate_probabilities=True,
            nr_topics="auto",
        )
        self.topics, self.probabilities = self.topic_model.fit_transform(self.df["cleaned_text"], self.embeddings)

    def get_topics(self):
        """
        Returns a dictionary of topics with top words and probabilities.
        """
        topics = self.topic_model.get_topic_info()
        topics_dict = {}
        for index, row in topics.iterrows():
            topic_id = row["Topic"]
            if topic_id != -1:
                topics_dict[topic_id] = {}
                for word, prob in zip(row["Words"].split(","), row["WordScores"]):
                    topics_dict[topic_id][word] = prob
        return topics_dict
