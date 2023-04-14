import logging
import os

import torch
import transformers.utils.logging
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer

from src.models.sentiment_analysis.base_model import BaseModel

logging.basicConfig(level=logging.ERROR)
transformers.utils.logging.set_verbosity_error()


class BERTDataset:
    """
    A custom dataset class for loading and pre-processing text data using BERT tokenizer.

    Attributes
    ----------
    reviews : list
        The reviews in the dataset.
    tokenizer : transformers.BertTokenizer
        The BERT tokenizer used to process the text data.
    max_len : int
        The maximum length for the tokenized text.
    """

    def __init__(self, reviews, tokenizer, max_len):
        """
        Initialize the BERTDataset object.

        Parameters
        ----------
        reviews : list
            The reviews in the dataset.
        tokenizer : transformers.BertTokenizer
            The BERT tokenizer used to process the text data.
        max_len : int
            The maximum length for the tokenized text.
        """
        self.reviews = reviews
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        """
        Get the number of items in the dataset.

        Returns
        -------
        int
            The number of items in the dataset.
        """
        return len(self.reviews)

    def __getitem__(self, item):
        """
        Get an item from the dataset by index.

        Parameters
        ----------
        item : int
            The index of the item to get.

        Returns
        -------
        dict
            A dictionary containing the tokenized review text, input IDs, and attention mask.
        """
        review = str(self.reviews[item])

        # Encoded format to be returned
        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            # pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors="pt",
            truncation=True,
        )

        return {
            "review_text": review,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }


class SentimentClassifier(nn.Module):
    """
    A class for sentiment classification using a fine-tuned BERT model.

    Attributes
    ----------
    bert : transformers.BertModel
        The pre-trained BERT model.
    drop : torch.nn.Dropout
        The dropout layer.
    out : torch.nn.Linear
        The output layer.
    """

    def __init__(self, n_classes):
        """
        Initialize the SentimentClassifier object.

        Parameters
        ----------
        n_classes : int
            The number of output classes.
        """
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        """
        Forward propagation through the model.

        Parameters
        ----------
        input_ids : torch.Tensor
            The input IDs of the tokenized text.
        attention_mask : torch.Tensor
            The attention mask for the tokenized text.

        Returns
        -------
        torch.Tensor
            The output tensor after passing through the model.
        """
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        #  Add a dropout layer
        output = self.drop(pooled_output)
        return self.out(output)


class BertFineTuned(BaseModel):
    """
    A class for fine-tuning BERT models for sentiment classification tasks.

    Attributes
    ----------
    device : torch.device
        The device on which the model will run (either CPU or GPU).
    tokenizer : transformers.BertTokenizer
        The BERT tokenizer used to process the text data.
    saved_model : torch.nn.Module
        The fine-tuned BERT model.
    batch_size : int
        The batch size for the DataLoader.
    model_dir : str
        The directory path to the saved model.
    model_path : str
        The file path to the saved model.
    """

    def __init__(self, models_path):
        """
        Initialize the BertFineTuned object.

        Parameters
        ----------
        models_path : str
            The path to the directory containing the saved models.
        """
        super().__init__(models_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        self.saved_model = None
        self.batch_size = 16
        self.model_dir = os.path.join(self.models_path, "bert_fine_tuned")
        self.model_path = os.path.join(self.model_dir, "bert_state_dict_new_raw.pt")

    def save(self, model_name):
        """
         Save the model under the given model_name in the models_path directory.
         This method is not implemented since Siebert uses a pre-trained model.

         Parameters
         ----------
         model_name : str
             Name of the model to be saved.

         Returns
         -------
         None
         """
        pass

    def load(self, model_name):
        """
        Load the model from the models_path directory. This method is not implemented since Siebert uses a pre-trained model.

        Parameters
        ----------
        model_name : str
            Name of the model to be loaded.

        Returns
        -------
        None
        """
        pass

    def fit(self, x_train, y_train):
        """
        Fit method is not implemented since Siebert uses a pre-trained model.

        Parameters
        ----------
        x_train : pd.DataFrame
            Training data, a Pandas DataFrame containing text data to be used for training.

        y_train : pd.Series
            Labels corresponding to the training data.

        Returns
        -------
        None
        """
        pass

    def predict(self, x_test):
        """
        Predict the sentiment of the given text data using the fine-tuned BERT model.

        Parameters
        ----------
        x_test : pandas.DataFrame or pandas.Series
            The test data containing the text.

        Returns
        -------
        dict
            A dictionary containing the predicted sentiment and predicted sentiment probability.
        """
        # Build a BERT based tokenizer
        x_test = x_test.text.to_list()
        test_dataset = BERTDataset(x_test, self.tokenizer, 512)

        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=0)

        class_names = ["0", "1"]
        self.saved_model = SentimentClassifier(len(class_names))
        self.saved_model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.saved_model = self.saved_model.to(self.device)

        self.saved_model.eval()

        review_texts = []
        predictions = []
        prediction_probs = []

        with torch.no_grad():
            for d in test_dataloader:
                texts = d["review_text"]
                input_ids = d["input_ids"].to(self.device)
                attention_mask = d["attention_mask"].to(self.device)

                # Get outputs
                outputs = self.saved_model(input_ids=input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs, dim=1)

                review_texts.extend(texts)
                predictions.extend(preds)
                prediction_probs.extend(outputs)

        predictions = torch.stack(predictions).cpu()
        prediction_probs = torch.stack(prediction_probs).cpu()

        return {
            "predicted_sentiment": predictions,
            "predicted_sentiment_probability": prediction_probs,
        }
