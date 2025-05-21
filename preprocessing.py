import os
from src.logger import logging
from src.exception import Custom_Exception
import sys
import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.preprocessing import LabelEncoder
nltk.download('stopwords')
nltk.download('punkt_tab')

def transform_text(text: str) -> str:
    """
    Transform the input text by removing punctuation, converting to lowercase,
    removing stopwords, and stemming.

    Args:
        text (str): The input text to transform.

    Returns:
        str: The transformed text.
    """
    try:
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Convert to lowercase
        text = text.lower()

        # Tokenize the text
        tokens = nltk.word_tokenize(text)

        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]

        # Stemming
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]

        return ' '.join(tokens)

    except Exception as e:
        logging.error('Error in transforming text: %s', e)
        raise Custom_Exception(e, sys)
    
def preprocess_df(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the input DataFrame by transforming the text and encoding labels.

    Args:
        data (pd.DataFrame): The input DataFrame to preprocess.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    try:
        # Transform the text in the 'messages' column
        logging.debug('Starting data preprocessing')
        data['messages'] = data['messages'].apply(transform_text)

        # Encode the labels in the 'label' column
        logging.debug('Encoding labels started')
        label_encoder = LabelEncoder()
        data['label'] = label_encoder.fit_transform(data['label'])

        #Removing duplcate rows
        logging.debug('Removing duplicate rows')
        data.drop_duplicates(keep="first",inplace=True)

        logging.debug('Data preprocessing completed')
        return data
    except Exception as e:
        logging.error('Error in preprocessing DataFrame: %s', e)
        raise Custom_Exception(e, sys)
    
def main(independent_variable='messages'):
    """
    Main function to run the preprocessing steps.

    Args:
        data (pd.DataFrame): The input DataFrame to preprocess.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    try:
        train_data=pd.read_csv(r'D:\Vikash_Dash_Demo_Spam_MLOps\MLOPS_DEMO\data\raw\train_data.csv')
        test_data=pd.read_csv(r'D:\Vikash_Dash_Demo_Spam_MLOps\MLOPS_DEMO\data\raw\test_data.csv')
        logging.debug('Text Loaded completed')

        # Transforming the DataFrame
        train_data[independent_variable] = train_data[independent_variable].apply(transform_text)
        test_data[independent_variable] = test_data[independent_variable].apply(transform_text)


        # Preprocessing the DataFrame
        train_processed_data= preprocess_df(train_data)
        test_preprocessed_data= preprocess_df(test_data)

        #Storing the preprocessed data
        data_path=os.path.join('.\data','processed_data')
        os.makedirs(data_path, exist_ok=True)
        

        train_processed_data.to_csv(os.path.join(data_path,'train_processed_data.csv'), index=False)
        test_preprocessed_data.to_csv(os.path.join(data_path,'test_processed_data.csv'), index=False)
        logging.debug(f'Preprocessed data saved successfully {data_path}')
    except Exception as e:
        logging.error('Error in main function: %s', e)
        raise Custom_Exception(e, sys)
    

if __name__ == "__main__":
    main()