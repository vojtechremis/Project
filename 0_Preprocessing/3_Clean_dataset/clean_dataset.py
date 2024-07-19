"""
Script: clean_dataset.py
Desc: Applies outputs of cleaning (outliers and missing values) to a loaded dataset and saves it to new DB.

Removes a list of indicators with frequent missing values from a dataset.
Filters a dataset based on a dictionary of indicators and their boundary values, excluding outliers.
"""

root_directory = '/Users/vojtechremis/Desktop/bachelorproject/'

import sys
sys.path += [root_directory+'Inc']
import json
import pandas as pd
import log
import database as db
Log = log.log()

path_to_blacklist = '.../0_Preprocessing/_output/blacklist_tickers.json' # Not used in thesis
path_to_outliers_conditions = root_directory+'0_Preprocessing/2_Outliers/_output/outliers_conditions.json'
path_to_blank =  root_directory+'0_Preprocessing/2_Outliers/_output/blank_indicators_B10.csv'


# Loading dataset
path_to_database = '.../0_Preprocessing/_output/researchwizz_rp.db'
path_to_clean_database = '.../0_Preprocessing/_output/dataset_clean_B10.db'
query = 'SELECT * FROM table_name'

Log.info('Loading database...')
connectionToSourceDB = db.Connection(path_to_database)

researchwizzDB = connectionToSourceDB.tableToDF(query)
if researchwizzDB is None:
    sys.exit(0)

noNumberColumns = ['Company', 'Ticker', 'NERD', 'export_date']
numberColumns = set(researchwizzDB.columns.tolist()) - set(noNumberColumns)
numberColumns = list(numberColumns)

Log.info(f'ResearchwizzDB initial size: {len(researchwizzDB)} rows.')

# Remove blank tickers
blanklist = pd.read_csv(path_to_blank)
blanklist_indicators = blanklist['Indicator'].values.tolist()
remaining_indicators = [item for item in researchwizzDB.columns.tolist() if item not in blanklist_indicators]
researchwizzDB = researchwizzDB[ remaining_indicators ]

Log.info(f'ResearchwizzDB after removing blank tickers size: {len(researchwizzDB)} rows.')


"""
# Remove blacklist
with open(path_to_blacklist, 'r') as file:
    blacklist = json.load(file)

blacklist_tickers = blacklist.keys()
researchwizzDB = researchwizzDB[ ~researchwizzDB['Ticker'].isin(blacklist_tickers) ]

Log.info(f'ResearchwizzDB after removing blacklist size: {len(researchwizzDB)} rows.')
"""

# Log size without NaN
researchwizzDB_noBlank = researchwizzDB.dropna(how='any')

researchwizzDB_noBlank_Len = len(researchwizzDB_noBlank)
Log.info(f'ResearchwizzDB after removing all NaN (before dealing with outliers) rows size: {researchwizzDB_noBlank_Len} rows.')


# Remove outliers
with open(path_to_outliers_conditions, 'r') as file:
    outliers_conditions = json.load(file)

filter = pd.Series(True, index=researchwizzDB.index)

dataset_reduction = []
for indicator, limits in outliers_conditions.items():
    if indicator in remaining_indicators:
        # if indicator in no_slice_indicators:
        #     print(f'Not slicing indicator [{indicator}].')
        #     continue
        min_ = limits[0]
        max_ = limits[1]

        filter &= ((min_ <= researchwizzDB[indicator]) & (researchwizzDB[indicator] <= max_))

        researchwizzDB = researchwizzDB.loc[ filter ]

        dataset_reduction.append({
            'inicator': indicator,
            'dataset_size': len(researchwizzDB)
        })
        Log.info(f'ResearchwizzDB after removing indicator={indicator} outliers size: {len(researchwizzDB)} rows.')

# Saving dataset reduction summary
# Data reduction summary gives an information how much rows has dataset lost after removing outliers of each indicator.
pd.DataFrame(dataset_reduction).to_csv('.../0_Preprocessing/3_Clean_dataset/dataset_reduction_summary.csv')

# Additional size info log
researchwizzDB.dropna(how='any', inplace=True)
noNan_datasetLen = len(researchwizzDB)
Log.info(f'ResearchwizzDB after removing all NaN rows size: {noNan_datasetLen} rows.')


# Save database with relative profit
Log.info('Saving clean database.')

db.createDatabase(path_to_clean_database)
db_connection_cln = db.Connection(path_to_clean_database)
db_connection_cln.dataframeToTable('new_table_name', researchwizzDB)

Log.info(f'Cleaned database saved to [{path_to_clean_database}].')