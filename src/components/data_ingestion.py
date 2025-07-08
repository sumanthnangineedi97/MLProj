import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.utils import read_sql_data

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')

class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def execute(self):
        try:
            # Load data from MySQL using utility function
            df = read_sql_data()
            logging.info("Successfully fetched data from MySQL database")

            # Ensure target directory exists
            os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)

            # Save full raw dataset
            df.to_csv(self.config.raw_data_path, index=False)
            logging.info("Raw dataset saved")

            # Split into train and test sets
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
            train_df.to_csv(self.config.train_data_path, index=False)
            test_df.to_csv(self.config.test_data_path, index=False)

            logging.info("Data ingestion and splitting completed successfully")

            return self.config.train_data_path, self.config.test_data_path

        except Exception as e:
            raise CustomException(e, sys)
