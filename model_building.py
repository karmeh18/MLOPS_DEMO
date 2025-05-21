import os
import numpy as np
import pandas as pd
import pickle
from src.logger import logging
from src.exception import Custom_Exception
from helper.load_data import load_data
from helper.load_yml import load_params
import yaml
import sys
from sklearn.ensemble import RandomForestClassifier


def train_model(X_train: pd.DataFrame, y_train: pd.Series, n_estimators:int) -> RandomForestClassifier:
    """
    Train a Random Forest model.
    
    Args:
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training labels.
        n_estimators (int): The number of trees in the forest.
        
    Returns:
        RandomForestClassifier: The trained model.
    """
    try:
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("The number of samples in X_train and y_train must be equal.")
        logging.debug('Training the model with %d estimators', n_estimators)
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
        logging.debug('Model Training Completed')
        return model
    except Exception as e:
        logging.error('Failed to train the model %s', e)
        raise Custom_Exception(e, sys)
    
def save_model(model: RandomForestClassifier, file_path: str) -> None:
    """
    Save the trained model to a file.
    
    Args:
        model (RandomForestClassifier): The trained model.
        file_path (str): The path where the model will be saved.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logging.debug('Model Saved to %s', file_path)
    except Exception as e:
        logging.error('Failed to save the model %s', e)
        raise Custom_Exception(e, sys)
    
def main():
    try:
        n_estimators = load_params("D:\Vikash_Dash_Demo_Spam_MLOps\MLOPS_DEMO\params.yaml")["data_explorer"]["n_estimators"]
        train_data = load_data(os.path.join("./data","tfidf_processed","train_data.csv"))
        file_path='data/models/rfc.pkl'
        logging.debug('Data Loaded Completed')
        X_train = train_data.drop(columns=['label'])
        y_train = train_data['label']

        model= train_model(X_train, y_train, n_estimators)
        save_model(model,file_path)
    except Exception as e:
        logging.error('Error in main function: %s', e)
        raise Custom_Exception(e, sys)
    
if __name__ == "__main__":
    main()