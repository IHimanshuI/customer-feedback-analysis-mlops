import logging
import pandas as pd
from src.data_cleaning import DataCleaning, DataDivideStrategy, DataPreProcessStrategy
from zenml import step
from typing import Tuple
from typing_extensions import Annotated

@step
def clean_df(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X Train"],
    Annotated[pd.DataFrame, "X Test"],
    Annotated[pd.Series, "y Train"],
    Annotated[pd.Series, "y Test"]
]:
    """
    Cleans the data and divides it into train and test.

    Args:
        df: Raw data
    Returns:
        X_train: Training data
        X_test: Testing data
        y_train: Training labels
        y_test: Testing labels
    """
    try:
        process_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(df, process_strategy)
        processed_data = data_cleaning.handle_data()

        divide_strtegy = DataDivideStrategy()
        data_cleaning = DataCleaning(processed_data, divide_strtegy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logging.info("Data cleaning completed.")
    except Exception as e:
        logging.error(f"Error in cleaning data: {e}")
        raise e