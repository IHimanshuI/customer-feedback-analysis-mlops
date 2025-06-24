import logging 
import pandas as pd
from zenml import step
from src.evaluation import MSE, R2, RMSE
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated
import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(
    model: RegressorMixin,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[
    Annotated[float, "R2 Score"],
    Annotated[float, "RMSE Score"]
]:
    """
    Evaluates the model on the ingested data.
    Args:
        df: the ingested data
    """
    try:
            
        predictions = model.predict(X_test)
        mse_class = MSE()
        mse_score = mse_class.calculate_scores(y_test, predictions)
        mlflow.log_metric("mse", mse_score)
        r2_class = R2()
        r2_score = r2_class.calculate_scores(y_test, predictions)
        mlflow.log_metric("r2_score", r2_score)
        rmse = RMSE()
        rmse_score = rmse.calculate_scores(y_test, predictions)
        mlflow.log_metric("rmse", rmse_score)
        # acc = Accuracy()
        # accuracy_score = acc.calculate_scores(y_test, predictions)
        
        return r2_score , rmse_score
    except Exception as e:
        logging.error(f"Error in evaluating model: {e}")
        raise e