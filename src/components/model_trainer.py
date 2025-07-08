import os
import sys
import mlflow
import mlflow.sklearn
import numpy as np
from dataclasses import dataclass
from urllib.parse import urlparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    @staticmethod
    def calculate_metrics(actual, predicted):
        """Compute RMSE, MAE, and R2 metrics."""
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)
        r2 = r2_score(actual, predicted)
        return rmse, mae, r2

    def train_and_select_model(self, train_array, test_array):
        try:
            logging.info("Preparing training and testing datasets")

            # Extract features and target variables
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # Define candidate models
            model_candidates = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # Define hyperparameter search space
            hyperparameter_grid = {
                "Decision Tree": {'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']},
                "Random Forest": {'n_estimators': [8, 16, 32, 64, 128, 256]},
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "CatBoost Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [0.1, 0.01, 0.5, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }

            # Evaluate models and select the best
            model_scores = evaluate_models(X_train, y_train, X_test, y_test, model_candidates, hyperparameter_grid)
            top_model_name = max(model_scores, key=model_scores.get)
            top_model_score = model_scores[top_model_name]
            top_model = model_candidates[top_model_name]
            selected_params = hyperparameter_grid.get(top_model_name, {})

            logging.info(f"Best model: {top_model_name} | Score: {top_model_score}")
            print(f"Top model selected: {top_model_name}")

            # Start MLflow tracking
            mlflow.set_tracking_uri("http://127.0.0.1:5000")
            tracking_scheme = urlparse(mlflow.get_tracking_uri()).scheme

            with mlflow.start_run():
                predictions = top_model.predict(X_test)
                rmse, mae, r2 = self.calculate_metrics(y_test, predictions)

                if selected_params:
                    mlflow.log_params(selected_params)

                mlflow.log_metric("RMSE", rmse)
                mlflow.log_metric("MAE", mae)
                mlflow.log_metric("R2", r2)

                # Save and log the model manually as an artifact (not registry)
                local_model_path = os.path.join("artifacts", "mlflow_model.pkl")
                save_object(local_model_path, top_model)
                mlflow.log_artifact(local_model_path, artifact_path="model")

            # Ensure performance is acceptable
            if top_model_score < 0.6:
                raise CustomException("No model passed the performance threshold.")

            # Save model locally for use
            save_object(self.config.trained_model_file_path, top_model)
            logging.info("Final model saved successfully.")

            final_predictions = top_model.predict(X_test)
            final_r2 = r2_score(y_test, final_predictions)
            return final_r2

        except Exception as e:
            raise CustomException(e, sys)
