"""
Script: prepare_data.py
Desc: Applying 4_PCA to a dataset.
Using loaded 4_PCA output transforming dataset, applying MinMax scaler and saving it to a new DB.
"""

root_directory = '/Users/vojtechremis/Desktop/bachelorproject/'

import sys
sys.path += [root_directory+'Inc']
from Inc import pca
import log
import database as db
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
Log = log.log()

# Loading dataset
path_to_database = '.../0_Preprocessing/_output/dataset_clean_B10.db'
query = 'SELECT * FROM table_name'

Log.info('Loading database...')
connectionToSourceDB = db.Connection(path_to_database)
researchwizzDB = connectionToSourceDB.tableToDF(query)
if researchwizzDB is None:
    sys.exit(0)

columns_all = researchwizzDB.columns.tolist()
nonNumeric_columns = ['Company', 'Ticker', 'NERD', 'export_date', 'relative_profit']

db_sum = researchwizzDB.sum()
const_zero_columns = db_sum[ db_sum == 0].index.tolist() # Columns with all values equal to zero

for column in nonNumeric_columns+const_zero_columns:
    columns_all.remove(column)

columns_PCA = columns_all
target_variable = 'relative_profit'

# Getting X, y
Y = researchwizzDB[target_variable]
X = np.array(researchwizzDB[columns_PCA])

Log.info(f'Columns to be transformed: \n {columns_PCA}.')

# Transformation to the 4_PCA space
path_to_PCA = root_directory+'0_Preprocessing/4_PCA/dataset_clean_B10/Output'

Retrieved_PCA = pca.PCA(path_to_PCA)
X_PCA_space = Retrieved_PCA.transform(X)

# Minmax scaling
path_to_scalerData = '.../0_Preprocessing/_output/data_B10_MinMaxScaler.pkl'
MinMaxScaling = True
if MinMaxScaling:
    scaler = MinMaxScaler()
    X_PCA_space_real = MinMaxScaler().fit_transform(X_PCA_space.real)
    joblib.dump(scaler, path_to_scalerData)
else:
    X_PCA_space_real = X_PCA_space.real

# Save Data in (PCA1 x PCA2) space
Log.info('Saving data in (PCA1 x PCA2) space.')
path_to_new_database = '.../0_Preprocessing/_output/data_B10_pcaSpace_minmax.db'

X_PCA_space_target = np.column_stack((X_PCA_space_real, Y))
new_column_names = ['PC'+str(i+1) for i in range(X_PCA_space.shape[1])] + ['relative_profit']

db.createDatabase(path_to_new_database)
db_connection_cln = db.Connection(path_to_new_database)
db_connection_cln.dataframeToTable('pca_space', pd.DataFrame(data=X_PCA_space_target, columns=new_column_names))

Log.info(f'Database in (PCA1 x PCA2) space saved to [{path_to_new_database}].')