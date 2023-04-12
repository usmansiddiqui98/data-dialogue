# Import necessary libraries
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from src.models.sentiment_analysis.bert import BERTDataset

# Load Saved Model
from transformers import (
    BertModel,
    BertTokenizer
)


class BertFineTuned:
    def __init__(self, path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.PATH = path
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        self.saved_model = None

    def predict(self, x_test):
        # Build a BERT based tokenizer
        test_encodings = self.tokenizer(x_test, truncation=True, padding=True, max_length=256, return_tensors="pt")
        test_dataset = BERTDataset(test_encodings, np.zeros(len(x_test)))
        test_dataloader = DataLoader(test_dataset, batch_size=16, num_workers=0)

        class_names = ["0", "1"]
        self.saved_model = SentimentClassifier(len(class_names))
        self.saved_model.load_state_dict(torch.load(self.PATH, map_location=self.device))
        self.saved_model = self.saved_model.to(self.device)

        self.saved_model.eval()
        predictions = []

        with torch.no_grad():
            for batch in test_dataloader:
                inputs = {key: value.to(self.device) for key, value in batch.items() if key != "labels"}
                outputs = self.saved_model(**inputs)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                predictions.extend(preds)

        return predictions


# class GPReviewDataset(Dataset):
#     # Constructor Function
#     def __init__(self, reviews, targets, tokenizer, max_len):
#         self.reviews = reviews
#         self.targets = targets
#         self.tokenizer = tokenizer
#         self.max_len = max_len
#
#     # Length magic method
#     def __len__(self):
#         return len(self.reviews)
#
#     # get item magic method
#     def __getitem__(self, item):
#         review = str(self.reviews[item])
#         target = self.targets[item]
#
#         # Encoded format to be returned
#         encoding = self.tokenizer.encode_plus(
#             review,
#             add_special_tokens=True,
#             max_length=self.max_len,
#             return_token_type_ids=False,
#             pad_to_max_length=True,
#             return_attention_mask=True,
#             return_tensors="pt",
#         )
#
#         return {
#             "review_text": review,
#             "input_ids": encoding["input_ids"].flatten(),
#             "attention_mask": encoding["attention_mask"].flatten(),
#             "targets": torch.tensor(target, dtype=torch.long),
#         }
#
#
# def create_data_loader(df, tokenizer, max_len, batch_size):
#     ds = GPReviewDataset(
#         reviews=df.cleaned_text.to_numpy(),
#         # targets=df.Sentiment.to_numpy(),
#         tokenizer=tokenizer,
#         max_len=max_len,
#     )
#
#     return DataLoader(ds, batch_size=batch_size, num_workers=0)


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
