import sys
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from src.logger import logging
from src.exception import Custom_Exception
from helper.load_data import load_data
from helper.load_yml import load_params



def apply_tfidf(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features:int) -> tuple:
    """
    Apply TF-IDF vectorization to the text data in the DataFrame. and return tuple of train and test data.
    """
    try:
        vectorizer= TfidfVectorizer(max_features=max_features)
        
        X_train = train_data['messages'].values
        y_train = train_data['label'].values

        X_test = test_data['messages'].values
        y_test = test_data['label'].values
        logging.debug('TF-IDF Vectorization started')
        
        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)

        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['label'] = y_train
        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['label'] = y_test
        logging.debug('TF-IDF Vectorization completed')

        return train_df, test_df
    except Exception as e:
        logging.error('Error in TF-IDF vectorization: %s', e)
        raise Custom_Exception(e, sys)
    
def save_data(df:pd.DataFrame,file_path:str) -> None:
    """
    Save the DataFrame to a CSV file.
    
    Args:
        df (pd.DataFrame): The DataFrame to save.
        file_path (str): The path to save the CSV file.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Create directory if it doesn't exist
        df.to_csv(file_path, index=False)
        logging.debug('Data saved to %s', file_path)
    except Exception as e:
        logging.error('Error saving data to %s: %s', file_path, e)
        raise Custom_Exception(e, sys)
    

def main():
    try:
        train_data = load_data(os.path.join("./data","processed_data","train_processed_data.csv"))
        test_data = load_data(os.path.join("./data","processed_data","test_processed_data.csv"))

        max_features=50
        train_df,test_df = apply_tfidf(train_data, test_data, max_features)

        save_data(train_df, os.path.join('data', 'tfidf_processed', 'train_data.csv'))
        save_data(test_df, os.path.join('data', 'tfidf_processed', 'test_data.csv'))
    except Exception as e:
        logging.error('Error in main function: %s', e)
        raise Custom_Exception(e, sys)
    
if __name__ == "__main__":
    main()
