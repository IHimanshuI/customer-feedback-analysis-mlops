import logging
import pandas as pd
from zenml import step
from src.model_dev import LineraRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig

@step
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    config: ModelNameConfig
) -> RegressorMixin:
    """
    Trains the model on the ingested data.
    Args:
        df: the ingested data
    """
    try:
        model = None
        if config.model_name == "LinearRegression":
            model = LineraRegressionModel()
            trained_model = model.train(X_train, y_train)
            return trained_model
        # if config.model_name == "RandomForestRegressor":
        #     model = RandomForestRegressor()
        else:
            raise ValueError(f"Model {config.model_name} is not supported.")
    except Exception as e:
        logging.error(f"Error in training model: {e}")
        raise e