import os
import pickle

from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from umap import UMAP


class BertTopic:
    def __init__(self, data, preprocessor):
        self.data = data
        self.preprocessor = preprocessor
        self.embeddings = None
        self.topics = None
        self.probabilities = None
        self.topic_model = None

    def prepare_embeddings(self):
        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        embeddings_file = os.path.join(BASE_DIR, "models/topic_modelling/bert_topic/BERTopic_embeddings.pickle")

        if os.path.exists(embeddings_file):
            print("Loading existing embeddings...")
            with open(embeddings_file, "rb") as pkl:
                self.embeddings = pickle.load(pkl)
        else:
            print("Creating new embeddings...")
            sentence_model = SentenceTransformer("all-MiniLM-L12-v2")
            self.embeddings = sentence_model.encode(self.pre_processed_df["cleaned_text"], show_progress_bar=True)

            # save embeddings
            with open(embeddings_file, "wb") as pkl:
                pickle.dump(self.embeddings, pkl)

    def run_bertopic(self):
        umap_model = UMAP(n_neighbors=100, n_components=3, min_dist=0.0, metric="cosine", random_state=4263)
        hdbscan_model = HDBSCAN(min_cluster_size=50, min_samples=20, metric="euclidean", prediction_data=True)
        self.topic_model = BERTopic(
            hdbscan_model=hdbscan_model,
            umap_model=umap_model,
            language="english",
            calculate_probabilities=True,
            nr_topics="auto",
        )
        self.topics, self.probabilities = self.topic_model.fit_transform(
            self.preprocessor.clean_df["cleaned_text"], self.embeddings
        )

    def get_topics(self):
        topics = self.topic_model.get_topic_info()
        topics_dict = {}
        for index, row in topics.iterrows():
            topic_id = row["Topic"]
            if topic_id != -1:
                topics_dict[topic_id] = {}
                for word, prob in zip(row["Words"].split(","), row["WordScores"]):
                    topics_dict[topic_id][word] = prob
        return topics_dict
