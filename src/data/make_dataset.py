# -*- coding: utf-8 -*-
from pathlib import Path
from sklearn.model_selection import train_test_split
from dotenv import find_dotenv, load_dotenv

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
    # train test split and write splits to csv
    train, test = train_test_split(
        feature_engineered_df, test_size=0.2, random_state=4263, stratify=feature_engineered_df["sentiment"]
    )

    train.to_csv(train_split_output_filepath)
    test.to_csv(test_split_output_filepath)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    # relative dir
    input_file = "../../data/raw/reviews.csv"
    train_output_file = "../../data/processed/train_final_processed_reviews.csv"
    test_output_file = "../../data/processed/test_final_processed_reviews.csv"
    main(input_file, train_output_file, test_output_file)
