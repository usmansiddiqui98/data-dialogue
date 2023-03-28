# -*- coding: utf-8 -*-
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from sklearn.model_selection import train_test_split

from src.data.feature_engineering import FeatureEngineer
from src.data.preprocess import Preprocessor


def main(input_filepath, train_split_output_filepath, test_split_output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    preprocessor = Preprocessor(input_filepath)
    preprocessor.clean_csv()
    pre_processed_df = preprocessor.clean_df
    feature_engineer = FeatureEngineer(pre_processed_df)
    feature_engineer.add_features()
    feature_engineered_df = feature_engineer.feature_engineered_df
    # Separate target variable (y) and features (X)
    X = feature_engineered_df.drop("sentiment", axis=1)
    y = feature_engineered_df["sentiment"]
    # train test split and write splits to csv
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=4263, stratify=feature_engineered_df["sentiment"]
    )

    # Write splits to csv
    train = X_train.join(y_train)
    test = X_test.join(y_test)
    train.to_csv(train_split_output_filepath)
    test.to_csv(test_split_output_filepath)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
    load_dotenv(find_dotenv())
    # relative dir
    input_file = "../../data/raw/reviews.csv"
    train_output_file = "../../data/processed/train_final_processed_reviews.csv"
    test_output_file = "../../data/processed/test_final_processed_reviews.csv"
    X_train, X_test, y_train, y_test = main(input_file, train_output_file, test_output_file)
