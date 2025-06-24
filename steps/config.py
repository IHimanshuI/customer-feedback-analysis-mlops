# from zenml.steps import BaseParameters
from zenml.config.base_settings import BaseSettings


class ModelNameConfig(BaseSettings):
    """
    Configuration for the model name.
    """
    model_name: str = "LinearRegression"