import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression


class Model(ABC):
    """
    Abstract class for all models.
    """

    @abstractmethod
    def train(self, X_train, y_train):
        """
        Train the model.
        
        Args:
            X_train: Training data
            y_train: Training labels
        Returns:
            None
        """

        pass

class LineraRegressionModel(Model):
    """
    Linear Regression model.
    """

    def train(self, X_train, y_train, **kwargs):
        """
        Train the linear regression model.
        
        Args:
            X_train: Training data
            y_train: Training labels
        Returns:
            None
        """
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logging.info("Model trained successfully.")
        except Exception as e:
            logging.error(f"Error in training model: {e}")
            raise e