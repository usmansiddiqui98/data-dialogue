import logging
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer

from src.models.sentiment_analysis.base_model import BaseModel

logging.basicConfig(level=logging.ERROR)


class BERTDataset:
    # Constructor Function
    def __init__(self, reviews, tokenizer, max_len):
        self.reviews = reviews
        self.tokenizer = tokenizer
        self.max_len = max_len

    # Length magic method
    def __len__(self):
        return len(self.reviews)

    # get item magic method
    def __getitem__(self, item):
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
            truncation=True
        )

        return {
            "review_text": review,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }


class SentimentClassifier(nn.Module):
    # Constructor class
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    # Forward propagation class
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        #  Add a dropout layer
        output = self.drop(pooled_output)
        return self.out(output)


class BertFineTuned(BaseModel):
    def __init__(self, models_path):
        super().__init__(models_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.PATH = os.path.join(models_path, "bert_state_dict.pt")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        self.saved_model = None
        self.batch_size = 16

    def save(self, model_name):
        pass

    def load(self, model_name):
        pass

    def fit(self, x_train, y_train):
        pass

    def predict(self, x_test):
        # Build a BERT based tokenizer
        x_test = x_test.text.to_list()
        test_dataset = BERTDataset(x_test, self.tokenizer, 216)

        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=0)

        class_names = ["0", "1"]
        self.saved_model = SentimentClassifier(len(class_names))
        self.saved_model.load_state_dict(torch.load(self.PATH, map_location=self.device))
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
