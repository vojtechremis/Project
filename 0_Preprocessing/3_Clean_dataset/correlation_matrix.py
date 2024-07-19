"""
Script: correlation_matrix.py
Desc: Calculating correlation matrix for a table loaded from DB
"""

root_directory = '/Users/vojtechremis/Desktop/bachelorproject/'

import pandas as pd
from matplotlib import pyplot as plt
import sys
sys.path += [root_directory+'Inc']
import database as db
import log
Log = log.log()

# Using custom style
plt.style.use(root_directory+'/Inc/MatPlotLib_styles/classicChart.mplstyle')

# Load data
path_to_database = f'.../0_Preprocessing/_output/dataset_clean_B10.db'
query = 'SELECT * FROM table_name'

connectionToSourceDB = db.Connection(path_to_database)
FundamentalData = connectionToSourceDB.tableToDF(query)

if FundamentalData:
    correlation_matrix = FundamentalData.corr()

    pd.DataFrame(correlation_matrix).to_csv(root_directory+'0_Preprocessing/_output/correlation_matrix.csv')

    print('Correlation matrix shape:', correlation_matrix.shape)