import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from src.utils import save_object
from src.logger import logging
from src.exception import CustomException


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def build_preprocessing_pipeline(self):
        """
        Constructs and returns a preprocessing pipeline for numeric and categorical features.
        """
        try:
            # Define feature groups
            numeric_features = ["writing_score", "reading_score"]
            categorical_features = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            # Numeric pipeline: handle missing values and scale
            numeric_pipeline = Pipeline(steps=[
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler())
            ])

            # Categorical pipeline: impute, encode, and scale
            categorical_pipeline = Pipeline(steps=[
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder()),
                ("scale", StandardScaler(with_mean=False))
            ])

            # Combine both pipelines
            preprocessor = ColumnTransformer(transformers=[
                ("num", numeric_pipeline, numeric_features),
                ("cat", categorical_pipeline, categorical_features)
            ])

            logging.info("Preprocessing pipelines created successfully")
            logging.info(f"Numeric features: {numeric_features}")
            logging.info(f"Categorical features: {categorical_features}")

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def transform_data(self, train_path: str, test_path: str):
        """
        Applies preprocessing to training and testing data and saves the preprocessor object.
        """
        try:
            # Load training and testing datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Training and testing datasets loaded")

            # Create preprocessing object
            preprocessor = self.build_preprocessing_pipeline()

            target_column = "math_score"
            feature_columns = ["writing_score", "reading_score"]

            # Split input and target for training data
            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]

            # Split input and target for test data
            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            logging.info("Splitting datasets into features and targets")

            # Apply transformations
            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test)

            # Combine features and targets
            train_data = np.c_[X_train_processed, y_train.values]
            test_data = np.c_[X_test_processed, y_test.values]

            # Save the preprocessing object
            save_object(
                file_path=self.config.preprocessor_obj_file_path,
                obj=preprocessor
            )

            logging.info("Preprocessing object saved successfully")

            return train_data, test_data, self.config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)
