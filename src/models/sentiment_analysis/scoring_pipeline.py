import os
from sys import platform

import pandas as pd

from src.data.feature_engineering import FeatureEngineer
from src.data.preprocess import Preprocessor
from src.models.sentiment_analysis.pre_trained.seibert import Seibert

input_filepath = "../../../data/raw/reviews_test.csv"
df = pd.read_csv(input_filepath)
preprocessor = Preprocessor(df)
preprocessor.clean_test_csv()
pre_processed_df = preprocessor.clean_df
feature_engineer = FeatureEngineer(pre_processed_df)
feature_engineer.add_features()
feature_engineered_df = feature_engineer.feature_engineered_df

time = feature_engineered_df.time
X_test = feature_engineered_df.drop(["time"], axis=1)

if platform == "win32":
    models_path = "..\\..\\..\\models\\sentiment_analysis"
else:
    models_path = "../../../models/sentiment_analysis"

best_model = "seibert"

model = Seibert(models_path)
model.load(best_model)
pred = model.predict(X_test)

# The output file should be named "reviews_test_predictions_<your_group_name>.csv ,
# and it should have columns - "Text", Time", "predicted_sentiment_probability", "predicted_sentiment".

output = pd.DataFrame(
    {
        "Text": X_test.text,
        "Time": time,
        "predicted_sentiment_probability": pred["predicted_sentiment_probability"],
        "predicted_sentiment": pred["predicted_sentiment"],
    }
)

output.to_csv("reviews_test_predictions_data-dialogue.csv", index=False)
