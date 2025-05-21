import pandas as pd
import os
from helper.load_yml import load_params
#train_data=pd.read_csv(r"D:\Vikash_Dash_Demo_Spam_MLOps\data\train_data.csv")

#print(train_data.shape)

params=load_params("D:\Vikash_dash_Demo_Spam_MLOps\MLOPS_spam_Demo\params.yml")
new_test_size=params['data_import']['test_size']
print(new_test_size)
#print(os.path.dirname(file_path))
#print(file_path)
print(os.getcwd())
#print(os.path.dirname(__file__))