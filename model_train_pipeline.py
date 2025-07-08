import sys
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    logging.info("Pipeline execution initiated.")

    try:
        # Step 1: Data ingestion
        ingestion = DataIngestion()
        train_path, test_path = ingestion.execute()

        # Step 2: Data transformation
        transformer = DataTransformation()
        train_data, test_data, _ = transformer.transform_data(train_path, test_path)

        # Step 3: Model training
        trainer = ModelTrainer()
        model_score = trainer.train_and_select_model(train_data, test_data)
        print(f"Model R2 Score: {model_score}")

    except Exception as error:
        logging.error("An error occurred during pipeline execution.")
        raise CustomException(error, sys)
