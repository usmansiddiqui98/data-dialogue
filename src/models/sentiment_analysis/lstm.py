from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.sentiment_analysis.base_model import BaseModel


class SentimentLSTM(nn.Module):
    """
    A PyTorch module for sentiment analysis using LSTM (Long Short-Term Memory).

    Args:
        no_layers (int): Number of LSTM layers.
        vocab_size (int): Size of the vocabulary.
        hidden_dim (int): Number of hidden dimensions in LSTM.
        embedding_dim (int): Dimension of word embeddings.
        output_dim (int): Dimension of output.

    Attributes:
        no_layers (int): Number of LSTM layers.
        output_dim (int): Dimension of output.
        hidden_dim (int): Number of hidden dimensions in LSTM.
        vocab_size (int): Size of the vocabulary.
        embedding (nn.Embedding): Embedding layer for word embeddings.
        lstm (nn.LSTM): LSTM layer for sequence modeling.
        dropout (nn.Dropout): Dropout layer for regularization.
        fc (nn.Linear): Fully connected layer for output.
        sig (nn.Sigmoid): Sigmoid activation function.

    """

    def __init__(self, no_layers, vocab_size, hidden_dim, embedding_dim, output_dim):
        super(SentimentLSTM, self).__init__()
        self.no_layers = no_layers
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim, hidden_size=self.hidden_dim, num_layers=no_layers, batch_first=True
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        """
        Forward pass of the LSTM model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len).
            hidden (tuple): Tuple containing the hidden state and cell state of LSTM.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
            tuple: Updated hidden state of LSTM.

        """
        batch_size = x.size(0)
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        out = self.dropout(lstm_out)
        out = self.fc(out)
        sig_out = self.sig(out)
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1]
        return sig_out, hidden

    def init_hidden(self, batch_size):
        """
        Initialize hidden state of LSTM.

        Args:
            batch_size (int): Batch size.

        Returns:
            tuple: Initial hidden state of LSTM.

        """
        hidden_state = torch.zeros((self.no_layers, batch_size, self.hidden_dim))
        cell_state = torch.zeros((self.no_layers, batch_size, self.hidden_dim))
        hidden = (hidden_state, cell_state)
        return hidden


class BasicLSTM(BaseModel):
    """A basic LSTM model for sentiment analysis."""

    onehot_dict = {}  # type: dict[str, int]

    def __init__(self, models_path):
        """
        Initialize the BasicLSTM model.

        Args:
            models_path (str): The path to save the trained model.
        """
        super().__init__(models_path)
        self.model = SentimentLSTM(no_layers=2, vocab_size=2001, hidden_dim=256, embedding_dim=64, output_dim=1)

    def data_preparation(self, x_train):
        """
        Prepare the training data.

        Args:
            x_train (list): List of sentences in the training data.

        Returns:
            np.ndarray: The prepared training data as a numpy array.
        """
        if len(self.onehot_dict) == 0:
            global onehot_dict
            # Getting unique words in x_train
            word_list = []
            for sent in x_train:
                for word in sent:
                    if word != "" and word not in word_list:
                        word_list.append(word)
            # Tokenization
            corpus = Counter(word_list)
            corpus_ = sorted(corpus.items(), key=lambda x: x[1], reverse=True)[:2000]
            self.onehot_dict = {w[0]: i + 1 for i, w in enumerate(corpus_)}
        final_list_train = []
        for sent in x_train:
            final_list_train.append(
                [self.onehot_dict[word] for word in sent.lower().split() if word in self.onehot_dict.keys()]
            )
        return np.array(final_list_train, dtype="object")

    def padding(self, sents, seq_len=50):
        """
        Pad the sentences to a fixed sequence length.

        Args:
            sents (list): List of sentences.
            seq_len (int, optional): The sequence length to pad the sentences to. Defaults to 50.

        Returns:
            np.ndarray: The padded sentences as a numpy array.
        """
        features = np.zeros((len(sents), seq_len), dtype=int)
        for i, rev in enumerate(sents):
            if len(rev) != 0:
                features[i, -len(rev) :] = np.array(rev, dtype="object")[:seq_len]
        return features

    def lstmdata(self, x, y):
        """
        Prepare the LSTM data.

        Args:
            x (np.ndarray): The input data.
            y (np.ndarray): The target labels.

        Returns:
            TensorDataset: The LSTM data as a TensorDataset object.
        """
        x_pad = self.padding(x)
        return TensorDataset(torch.from_numpy(x_pad), torch.from_numpy(np.asarray(y)))

    def fit(self, train_data, train_labels, num_epochs=1, batch_size=50, learning_rate=0.001):
        """
        Fits the LSTM model to the training data.

        Args:
            train_data (pd.Series): Training data with cleaned text.
            train_labels (np.ndarray): Training labels.
            num_epochs (int, optional): Number of epochs to train the model. Defaults to 1.
            batch_size (int, optional): Batch size for training. Defaults to 50.
            learning_rate (float, optional): Learning rate for optimizer. Defaults to 0.001.
        """
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        train = self.data_preparation(train_data["cleaned_text"])
        dataset = self.lstmdata(train, train_labels)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        for epoch in range(num_epochs):
            h = self.model.init_hidden(batch_size)
            self.model.train()
            for inputs, labels in data_loader:
                h = tuple([each.data for each in h])
                self.model.zero_grad()
                outputs, h = self.model(inputs, h)
                loss = criterion(outputs.squeeze(), labels.float())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

    def predict(self, X_test):
        """
        Predicts sentiment labels and probabilities for the test data.

        Args:
            X_test (pd.Series): Test data with cleaned text.

        Returns:
            dict: Dictionary containing predicted sentiment labels and probabilities.
                {"predicted_sentiment": List of predicted sentiment labels,
                 "predicted_sentiment_probability": List of predicted sentiment probabilities}
        """
        prob = []
        results = []
        word_seq = self.data_preparation(X_test["cleaned_text"])
        for text in word_seq:
            text = np.expand_dims(text, axis=0)
            pad = torch.from_numpy(self.padding(text))
            inputs = pad
            batch_size = 1
            h = self.model.init_hidden(batch_size)
            output, h = self.model(inputs, h)
            pred = output.item()
            label = 1 if pred > 0.5 else 0
            results.append(label)
            prob.append(pred)
        return {"predicted_sentiment": results, "predicted_sentiment_probability": prob}
