from src.data.make_dataset import main as make_dataset
from src.models.sentiment_analysis.bert import BERTModel
from src.models.sentiment_analysis.logistic_regression import LogisticRegression


def run_training_pipeline(input_file, train_output_file, test_output_file):
    X_train, X_test, y_train, y_test = make_dataset(input_file, train_output_file, test_output_file)

    models = [
        LogisticRegression(),
        BERTModel(),
    ]

    for model in models:
        model.train(X_train, y_train)

    best_f1 = 0
    best_model = None
    for model in models:
        y_pred = model.predict(X_test)  # Predict using X_test
        f1, precision, recall = model.evaluate(y_test, y_pred)  # Evaluate model performance
        print(f"{model.__class__.__name__} - F1: {f1}, Precision: {precision}, Recall: {recall}")

        if f1 > best_f1:
            best_f1 = f1
            best_model = model

    print(f"Best model: {best_model.__class__.__name__} with F1 score: {best_f1}")
