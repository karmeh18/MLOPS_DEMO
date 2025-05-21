import os
import pandas as pd
from src.logger import logging
from src.exception import Custom_Exception

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    
    Args:
        file_path (str): The path to the CSV file.
        
    Returns:
        pd.DataFrame: The loaded data as a DataFrame.
    """
    try:
        data = pd.read_csv(file_path)
        data.fillna('', inplace=True)  # Fill NaN values with empty strings
        logging.debug('Data Loaded from %s', file_path)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    