import os.path
from pathlib import Path

import nlpaug.augmenter.word as naw
import numpy as np
import pandas as pd
import tqdm as tqdm
from dotenv import find_dotenv, load_dotenv
from sklearn.model_selection import train_test_split


def bert_aug(Xy_train, TOPK=20, ACT="insert"):
    # count the number of samples to match majority class
    samples = abs(Xy_train.Sentiment.value_counts()[0] - Xy_train.Sentiment.value_counts()[1])
    aug_bert = naw.ContextualWordEmbsAug(
        model_path="distilbert-base-uncased",
        action=ACT,
        top_k=TOPK,
    )
    aug_bert.aug_p = 0.2
    new_text = []
    # selecting the minority class samples
    df_n = Xy_train[Xy_train.Sentiment == "negative"].reset_index(drop=True)
    # data augmentation loop
    for i in tqdm(np.random.randint(0, len(df_n), samples)):
        text = df_n.iloc[i]["Text"]
        augmented_text = aug_bert.augment(text)
        new_text.append(augmented_text[0])
    # dataframe
    new = pd.DataFrame({"Text": new_text, "Sentiment": "negative"})
    # assign new index values from training set
    new.index = list(range(Xy_train.shape[0] + 1, Xy_train.shape[0] + 1 + samples))
    new_Xy_train = pd.concat([Xy_train, new])  # combine ori training df and new samples
    return new_Xy_train


def generate_oversample(raw_df, oversample_filepath):
    """Runs raw reviews and generates an oversample of the training set"""
    # train-test split
    train, test = train_test_split(raw_df, test_size=0.2, random_state=4263, stratify=raw_df["Sentiment"])
    # need time to generate (~2hrs)
    print("starting oversampling")
    new_Xy_train = bert_aug(train)
    # combine back new train and old test set to proceed for cleaning & FE later on
    os_raw_df = pd.concat([new_Xy_train, test], axis=0)
    os_raw_df.to_csv(oversample_filepath, index=False)  # save the new oversampled reviews
    return "oversampling completed"
