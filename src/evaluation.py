import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

class Evaluation(ABC):
    """
    Abstract class defing strategy for evaluation our model
    """
    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculate evaluation scores.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        Returns:
            None
        """
        pass

class MSE(Evaluation):
    """
    Evaluation strategy for calculating Mean Squared Error (MSE).
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info('Calculating MSE.')
            mse = mean_squared_error(y_true, y_pred)
            logging.info(f"Error in calculation MSE: {mse}")
            return mse
        except Exception as e:
            logging.error(f"Error in calculating MSE: {e}")
            raise e
        
class R2(Evaluation):
    """
    Evaluation strategy for calculating R-squared (R2).
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info('Calculating R2.')
            r2 = r2_score(y_true, y_pred)
            logging.info(f"Error in calculation R2: {r2}")
            return r2
        except Exception as e:
            logging.error(f"Error in calculating R2: {e}")
            raise e
        
class RMSE(Evaluation):
    """
    Evaluation strategy for calculating Root Mean Squared Error (RMSE).
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info('Calculating RMSE.')
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            logging.info(f"Error in calculation RMSE: {rmse}")
            return rmse
        except Exception as e:
            logging.error(f"Error in calculating RMSE: {e}")
            raise e
        
# class Accuracy(Evaluation):
#     """
#     Evaluation strategy for calculating accuracy.
#     """
#     def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
#         try:
#             logging.info('Calculating Accuracy.')
#             accuracy = (np.sum(y_true == y_pred))/ len(y_true)
#             logging.info(f"Accuracy: {accuracy}")
#             return accuracy
#         except Exception as e:
#             logging.error(f"Error in calculating Accuracy: {e}")
#             raise e
        