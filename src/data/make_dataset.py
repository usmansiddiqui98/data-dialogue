import os
import time
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.feature_engineering import FeatureEngineer
from src.data.feature_engineering_optimised import FeatureEngineerOptimised
from src.data.preprocess import Preprocessor


def main(input_df, train_split_output_filepath=None, test_split_output_filepath=None):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """

    preprocessor = Preprocessor(input_df)
    preprocessor.clean_csv()
    pre_processed_df = preprocessor.clean_df
    print("[PP] Preprocessing complete")
    feature_engineer = FeatureEngineerOptimised(pre_processed_df)
    feature_engineer.add_features()
    feature_engineered_df = feature_engineer.feature_engineered_df
    print("[FE] finished adding features...")
    # Separate target variable (y) and features (X)
    X = feature_engineered_df.drop(["sentiment", "time"], axis=1)
    y = feature_engineered_df["sentiment"]
    # train test split and write splits to csv
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4263, stratify=y)
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)
    if train_split_output_filepath and test_split_output_filepath:
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
    # relative dir
    input_df = pd.read_csv(os.path.join(project_dir, "data/raw/reviews.csv"))
    train_output_file = os.path.join(project_dir, "data/processed/train_final_processed_reviews.csv")
    test_output_file = os.path.join(project_dir, "data/processed/test_final_processed_reviews.csv")
    start = time.time()
    X_train, X_test, y_train, y_test = main(input_df, train_output_file, test_output_file)
    end = time.time()
    total_time = end - start
    print("\n" + "Preprocessing and Feature Engineering finished in " + str(round(total_time)) + "s")
