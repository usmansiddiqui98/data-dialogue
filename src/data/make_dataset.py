# -*- coding: utf-8 -*-
import logging
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

# from src.data.preprocess import Preprocessor


def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    # preprocessor = Preprocessor(input_filepath)
    # pre_processed_df = preprocessor.clean_df
    # feature_extracted_df = FeatureExtractpr.extract(pre_processed_df)
    # train_test_split =


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    # relative dir
    input_file = "../../data/raw/reviews.csv"
    output_file = "../../data/processed/final_processed_reviews.csv"
    main(input_file, output_file)
