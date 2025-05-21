import os
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score , roc_auc_score, recall_score, f1_score
from helper.load_data import load_data

from src.logger import logging
from src.exception import Custom_Exception
import sys
from dvclive import Live
import json

def load_model(file_path: str) -> object:
    """
    Load a model from a file.
    
    Args:
        file_path (str): The path to the model file.
        
    Returns:
        object: The loaded model.
    """
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logging.debug('Model Loaded from %s', file_path)
        return model
    except Exception as e:
        logging.error('Failed to load the model %s', e)
        raise Custom_Exception(e, sys)
    
def evaluate_model(model: object, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate the model on the test data.
    
    Args:
        model (object): The trained model.
        X_test (pd.DataFrame): The test features.
        y_test (pd.Series): The test labels.
        
    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    try:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        logging.debug('Model Predictions Completed')
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        logging.debug('Model Evaluation Metrics Computed')
        
        logging.debug('Model Evaluation Completed')
        metric_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc
        }
        return metric_dict

    except Exception as e:
        logging.error('Failed to evaluate the model %s', e)
        raise Custom_Exception(e, sys)
    
def save_metrics(metrics: dict, file_path: str) -> None:
    """
    Save the evaluation metrics to a CSV file.
    
    Args:
        metrics (dict): The evaluation metrics.
        file_path (str): The path to save the metrics.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
            json.dump(metrics, file)
        logging.debug('Metrics Saved to %s', file_path)
    except Exception as e:
        logging.error('Failed to save the metrics %s', e)
        raise Custom_Exception(e, sys)
    
def main():
    try:
        # Load the model
        model_path = os.path.join('data', 'models', 'rfc.pkl')
        model = load_model(model_path)

        # Load the test data
        test_data_path = os.path.join('data', 'tfidf_processed', 'test_data.csv')
        test_data = load_data(test_data_path)
        logging.debug('Test Data Loaded from %s', test_data_path)

        # Split the data into features and labels
        X_test = test_data.iloc[:,:-1].values
        y_test = test_data.iloc[:,-1].values
        logging.debug('Test Data Split into Features and Labels')

        # Evaluate the model
        metrics = evaluate_model(model, X_test, y_test)

        # Save the metrics
        metrics_path = os.path.join('data', 'metrics', 'evaluation_metrics.json')
        save_metrics(metrics, metrics_path)

    except Exception as e:
        logging.error('Error in main function: %s', e)
        raise Custom_Exception(e, sys)
    
if __name__ == "__main__":
    main()