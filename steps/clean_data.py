import logging

import pandas as pd
from zenml import step
from src.data_cleaning import DataCleaning , DataDivideStrategy , DataPreprocessStrategy

from typing_extensions import Annotated
from typing import Tuple

@step
def clean_df(data:pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame,"X_train"],
    Annotated[pd.DataFrame,"y_train"],
    Annotated[pd.DataFrame,"X_test"],
    Annotated[pd.DataFrame,"y_test"],    
]:
    
    try:
        preprocess_strategy = DataPreprocessStrategy()
        data_cleaning = DataCleaning(data ,DataPreprocessStrategy)
        preprocessed_data = data_cleaning.handle_data()
        
        divide_strategy = DataDivideStrategy()
        data_cleaning = divide_strategy(preprocessed_data,divide_strategy)
        X_train , X_test, y_train, y_test = data_cleaning.handle_data()
        return X_train , X_test, y_train, y_test 
    except Exception as e:
        logging.error("Error in clean_data.py :{}".format(e))
        raise e
    