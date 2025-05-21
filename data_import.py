import pandas as pd
from sklearn.model_selection import train_test_split
import os
import logging
import yaml
import sys
from src.logger import logging
from src.exception import Custom_Exception
from helper.load_data import load_data
from helper.load_yml import load_params

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    try:
    # Example preprocessing: drop rows with missing values
        data.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
        data.rename(columns={'v1': 'label', 'v2': 'messages'}, inplace=True)
        logging.debug('Data Preprocessed Completed')
        return data
    except Exception as e:
        logging.error('Failed to preprocess the data %s', e)
        print(f"Error preprocessing data: {e}")

    
def save_data(data: pd.DataFrame, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    """
    Save DataFrame to a CSV file.
    Args:
        data (pd.DataFrame): The DataFrame to save.
        file_path (str): The path where the CSV file will be saved.
    """
    try:
        file_path = r"D:\Vikash_dash_Demo_Spam_MLOps\MLOPS_DEMO\data\raw"
        data_import_path=os.path.join(file_path,"raw_spam_data.csv")
        train_data_path=os.path.join(file_path,"train_data.csv")
        test_data_path=os.path.join(file_path,"test_data.csv")

        os.makedirs(file_path,exist_ok=True)        # Save the data
        data.to_csv(data_import_path, index=False)
        train_data.to_csv(train_data_path, index=False)
        test_data.to_csv(test_data_path, index=False)
        logging.debug('Data Saved to %s',data_import_path)
        logging.debug('Train Data Saved to %s',train_data_path)
        logging.debug('Test Data Saved to %s',test_data_path)
    except Exception as e:
        print(f"Error saving data: {e}")

def train_test_split_data(data: pd.DataFrame,test_size: int) -> tuple:
    """
    Split the data into training and testing sets.

    Args:
        data (pd.DataFrame): The DataFrame to split.

    Returns:
        tuple: A tuple containing the training and testing sets.
    """
    try:
        X=data['messages']
        y=data['label']
        # Splitting the data into training and testing sets
        train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
        logging.debug('Data Split into Train and Test Sets')
        return train_data, test_data
    except Exception as e:
        logging.error('Failed to split the data %s', e)
        print(f"Error splitting data: {e}")



    # Example usage
def main():
    try:
        file_path_param=load_params('D:\Vikash_dash_Demo_Spam_MLOps\MLOPS_DEMO\config.yml')

        file_path=file_path_param["import_path"]["file_path"]
        logging.debug('File Path %s',file_path)

        params=load_params("D:\Vikash_dash_Demo_Spam_MLOps\MLOPS_DEMO\params.yaml")

        new_test_size=params["data_import"]["test_size"]
        logging.debug('Test Size %s',new_test_size)

        data = load_data(file_path)

        pre_process_data=preprocess_data(data)

        train_data, test_data = train_test_split_data(pre_process_data, test_size=new_test_size)

        # Save the data

        save_data(data, train_data, test_data)
        logging.debug('Data Import Completed')
    except Exception as e:
        logging.error('Failed to run the main function %s', e)
        print(f"Error in main function: {e}")

if __name__ == "__main__":
    main()


