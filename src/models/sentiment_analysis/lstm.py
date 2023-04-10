from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.sentiment_analysis.base_model import BaseModel


class SentimentLSTM(nn.Module):
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
        hidden_state = torch.zeros((self.no_layers, batch_size, self.hidden_dim))
        cell_state = torch.zeros((self.no_layers, batch_size, self.hidden_dim))
        hidden = (hidden_state, cell_state)
        return hidden


class BasicLSTM(BaseModel):
    onehot_dict = {}  # type: dict[str, int]

    def __init__(self, models_path):
        super().__init__(models_path)
        self.model = SentimentLSTM(no_layers=2, vocab_size=2001, hidden_dim=256, embedding_dim=64, output_dim=1)

    def data_preparation(self, x_train):
        if len(self.onehot_dict) == 0:
            global onehot_dict
            # Getting unique words in x_train
            print("Creating a one hot dictionary containing top 2000 words based on frequency")
            word_list = []
            for sent in x_train:
                for word in sent:
                    if word != "" and word not in word_list:
                        word_list.append(word)
            # Tokenization
            corpus = Counter(word_list)
            corpus_ = sorted(corpus.items(), key=lambda x: x[1], reverse=True)[:2000]
            self.onehot_dict = {w[0]: i + 1 for i, w in enumerate(corpus_)}
            print("Done creating dictionary")
        else:
            print("Using existing one hot dictionary to tokenise words")
        final_list_train = []
        for sent in x_train:
            final_list_train.append(
                [self.onehot_dict[word] for word in sent.lower().split() if word in self.onehot_dict.keys()]
            )
        return np.array(final_list_train, dtype="object")

    def padding(self, sents, seq_len=50):
        features = np.zeros((len(sents), seq_len), dtype=int)
        for i, rev in enumerate(sents):
            if len(rev) != 0:
                features[i, -len(rev) :] = np.array(rev, dtype="object")[:seq_len]
        return features

    def lstmdata(self, x, y):
        x_pad = self.padding(x)
        return TensorDataset(torch.from_numpy(x_pad), torch.from_numpy(np.asarray(y)))

    def fit(self, train_data, train_labels, num_epochs=5, batch_size=50, learning_rate=0.001):
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        train = self.data_preparation(train_data["cleaned_text"])
        dataset = self.lstmdata(train, train_labels)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        print("Training begins")
        for epoch in range(num_epochs):
            print("Epoch: " + str(epoch))
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
