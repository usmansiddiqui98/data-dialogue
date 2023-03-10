import re

import pandas as pd
import torch
from transformers import BertForSequenceClassification, BertTokenizer

# Load pre-trained BERT model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

# Define label mappings
label_map = {0: "negative", 1: "neutral", 2: "positive"}


def pre_process_text(text):
    """Preprocesses the text in the 'reviews' column of a Pandas DataFrame."""
    text = text.lower()  # Convert text to lowercase
    text = re.sub("[^\w\s]", "", text)  # Remove punctuation
    text = re.sub("\d+", "", text)  # Remove numbers
    text = text.strip()  # Remove leading/trailing whitespaces
    return text


def predict_sentiment(text):
    """Predicts the sentiment of the text using a pre-trained BERT model."""
    encoded_text = tokenizer.encode_plus(text, padding=True, truncation=True, return_tensors="pt")
    input_ids = encoded_text["input_ids"]
    attention_mask = encoded_text["attention_mask"]
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs[0]
        _, predicted = torch.max(logits, dim=1)
        sentiment_scores = torch.softmax(logits, dim=1)
    sentiment_scores = sentiment_scores.tolist()
    predicted_label = label_map[predicted.item()]  # Convert tensor to Python integer
    sentiment_score = sentiment_scores[0][predicted]
    return (predicted_label, sentiment_score)


def apply_pre_process_text(df):
    """Applies pre_process_text function to 'reviews' column of Pandas DataFrame."""
    df["reviews"] = df["reviews"].apply(pre_process_text)
    return df


def apply_predict_sentiment(df):
    """Applies predict_sentiment function to 'reviews' column of Pandas DataFrame."""
    df[["sentiment", "sentiment_score"]] = df["reviews"].apply(predict_sentiment).apply(pd.Series)
    return df
