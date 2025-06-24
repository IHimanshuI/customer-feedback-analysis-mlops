import logging
import pandas as pd
from src.data_cleaning import DataCleaning, DataDivideStrategy, DataPreProcessStrategy
from zenml import step
from typing import Tuple
from typing_extensions import Annotated

@step
def clean_df(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
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
        preprocessor = DataCleaning(df, DataPreProcessStrategy())
        processed_data = preprocessor.handle_data()

        logging.info(f"Processed data shape: {processed_data.shape}")

        # Train-test split step
        splitter = DataCleaning(processed_data, DataDivideStrategy())
        X_train, X_test, y_train, y_test = splitter.handle_data()
        logging.info(f"X_train type: {type(X_train)}, y_train type: {type(y_train)}")
        logging.info(f"y_train.columns (if DF): {getattr(y_train, 'columns', 'Series')}")
        if isinstance(y_train, pd.DataFrame):
            y_train = y_train.squeeze()
        if isinstance(y_test, pd.DataFrame):
            y_test = y_test.squeeze()
        

        logging.info("Data cleaning and splitting completed successfully.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error in cleaning data: {e}")
        raise e