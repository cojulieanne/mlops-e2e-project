from load_data import load_data
from preprocessing.preprocess_data import preprocess_data
from feature_engineering.split_data import split_data
from training.training import train_model
from evaluation.evaluation import evaluate_model

def main():
    print("Starting the ML Pipeline...")
    print("Loading Data...")
    raw_data = load_data()
    print("Loading Data Completed.")

    print("Preprocessing data...")
    preprocessed_data = preprocess_data()
    print("Preprocessing completed.")

    print("Splitting Data...")
    train_data, test_data = split_data(preprocessed_data)
    print("Data Splitting Completed.")

    print("Training Model...")
    model = train_model(train_data)
    print("Model Training Completed.")

    print("Evaluating Model...")
    evaluate_model(model, test_data)
    print("Model Evaluation Completed.")

    print("Pipeline completed successfully.")

if __name__ == "__main__":
    main()