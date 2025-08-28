import mlflow
from load_data import load_data
from preprocessing import preprocess_data
from training import training
from evaluation import evaluation
from utils import logger
from drift_detection import detect_drift

logger = logger.get_logger(__name__)


def main():
    logger.info("Starting the ML Pipeline...")

    logger.info("Loading Data...")
    load_data.load()
    logger.info("Loading Data Completed.")

    logger.info("Preprocessing data...")
<<<<<<< HEAD
    # X_train, X_test, y_train, y_test, X_train_drifted, X_test_drifted = 
=======
    # X_train, X_test, y_train, y_test, X_train_drifted, X_test_drifted = preprocess_data.preprocess_data()
>>>>>>> origin
    preprocess_data.preprocess_data()
    logger.info("Preprocessing completed.")

    logger.info("Training Model...")
    # model = training.training(X_train, y_train)
    training.training()
    logger.info("Model Training Completed.")

    logger.info("Evaluating Model...")
    # evaluation.evaluation(X_train, y_train, X_test, y_test)
    evaluation.evaluation()
    logger.info("Model Evaluation Completed.")

    logger.info("Running Drift Detection...")

    test_drift_results = detect_drift(
        "data/silver/preprocessed_ml2_student_performance.csv",
        "data/gold/drifted_test.csv" 
    )
    mlflow.log_param("test_drift_detected", test_drift_results["drift_detected"])
    mlflow.log_param("test_overall_drift_score", test_drift_results["overall_drift_score"])

    if test_drift_results["drift_detected"]:
        raise ValueError("Data drift detected in test set! Model retraining required.")

    train_drift_results = detect_drift(
        "data/silver/preprocessed_ml2_student_performance.csv", 
        "data/gold/drifted_train.csv" 
    )
    mlflow.log_param("train_drift_detected", train_drift_results["drift_detected"])
    mlflow.log_param("train_overall_drift_score", train_drift_results["overall_drift_score"])

    if train_drift_results["drift_detected"]:
        raise ValueError("Data drift detected in training set! Model retraining required.")

    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()