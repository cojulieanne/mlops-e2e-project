from load_data import load_data
from preprocessing import preprocess_data
from feature_engineering import split_data
from training import training
from evaluation import evaluation
from utils import logger
logger = logger.get_logger(__name__)

def main():
    logger.info("Starting the ML Pipeline...")
    logger.info("Loading Data...")
    load_data.load()
    logger.info("Loading Data Completed.")

    logger.info("Preprocessing data...")
    preprocessed_data = preprocess_data.preprocess_data()
    logger.info("Preprocessing completed.")

    logger.info("Splitting Data...")
    train_data, test_data = split_data.split_data(preprocessed_data)
    logger.info("Data Splitting Completed.")

    logger.info("Training Model...")
    model = training.training(train_data)
    logger.info("Model Training Completed.")

    logger.info("Evaluating Model...")
    evaluation.evaluation(train_data, test_data)
    logger.info("Model Evaluation Completed.")

    logger.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()