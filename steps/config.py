# from zenml.steps import BaseParameters
from zenml.config.base_settings import BaseSettings


class ModelNameConfig(BaseSettings):
    """Model Configurations"""

    model_name: str = "lightgbm"
    fine_tuning: bool = False
