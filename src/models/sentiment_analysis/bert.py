import numpy as np
import torch
from base_model import BaseModel
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import AdamW, BertForSequenceClassification, BertTokenizer


class BERTDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class BERTModel(BaseModel):
    def __init__(self, model_name="bert-base-uncased", learning_rate=0.002, epochs=2):
        super().__init__()
        self.learning_rate = learning_rate
        self.model_name = model_name
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)

    def train(self, X_train, y_train):
        train_encodings = self.tokenizer(X_train, truncation=True, padding=True, max_length=256, return_tensors="pt")
        train_dataset = BERTDataset(train_encodings, y_train)

        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=16)

        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)

        self.model.train()

        for epoch in range(self.epochs):
            for batch in train_dataloader:
                inputs = {key: value.to(self.device) for key, value in batch.items() if key != "labels"}
                labels = batch["labels"].to(self.device)

                outputs = self.model(**inputs, labels=labels)
                loss = outputs.loss
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

    def predict(self, X_test):
        test_encodings = self.tokenizer(X_test, truncation=True, padding=True, max_length=256, return_tensors="pt")
        test_dataset = BERTDataset(test_encodings, np.zeros(len(X_test)))

        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=16)

        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch in test_dataloader:
                inputs = {key: value.to(self.device) for key, value in batch.items() if key != "labels"}
                outputs = self.model(**inputs)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                predictions.extend(preds)

        return predictions
