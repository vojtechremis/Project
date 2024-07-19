"""
Script: calculate_profit.py
Desc: Calculating relative profit on a table loaded from DB.
"""

root_directory = '/Users/vojtechremis/Desktop/bachelorproject/'

import numpy as np
import pandas as pd
import json

import sys
sys.path += [root_directory+'Inc']

import database as db

import log
Log = log.log("log_calculateProfit.txt")

def calculate_profit(query, path_to_database, path_to_database_rp, blacklist_dir):
    """
    Loads table from DB, calculates relative profit and saves it to new DB.

    Parameters:
    query (str): Select query.
    path_to_database (str): Absolute path (name included) to source DB.
    path_to_database_rp (str): Absolute path (name included) to new DB.
    blacklist_dir (str): Absolute path to directory where blacklist file is located.

    Returns:
    int or float: The sum of the two numbers.
    """
    # Loading data
    Log.info('Loading Database...')
    db_connection = db.Connection(path_to_database)
    if db_connection is None:
        return None

    researchwizzDB = db_connection.tableToDF(query)
    if researchwizzDB:
        Log.info('Database has been loaded.')
    else:
        return

    researchwizzDB.replace('N/A', np.nan, inplace=True) 
    researchwizzDB.replace('--------', np.nan, inplace=True)

    # Transform numerical columns to FLOAT
    for column in researchwizzDB.columns:
        try:
            researchwizzDB[column] = researchwizzDB[column].astype(float)
        except Exception as e:
            Log.warning(f'Column [{column}] could not be transformed to float.')

    researchwizzDB.sort_values(by='export_date', inplace=True)
    researchwizzDB.reset_index(drop=True, inplace=True)


    # Perform calculation

    # If ticker's price is above treshold value, add ticker to blacklist
    priceNonsense_trshld = 10e5
    blacklist = {}

    # Prepare list of tickers with ticker price = np.nan
    tickers = researchwizzDB['Ticker'].copy().drop_duplicates()
    TickersList_template = pd.Series(data=np.nan, index=tickers, dtype=int)

    # Initialize 'relative_profit' columns
    researchwizzDB['relative_profit'] = np.nan

    week_prev = None
    TickersList_prev = pd.Series
    TickersList_i = TickersList_template.copy()

    for i, row_i in researchwizzDB.iterrows():
        ticker_i = row_i['Ticker']
        price_i = row_i['CP']
        date_i = row_i['export_date']

        if date_i != week_prev:
            Log.info(f'\n\nNew week: {date_i}')

            TickersList_prev = TickersList_i
            TickersList_i = TickersList_template.copy()
            week_prev = date_i

        if np.isnan(TickersList_prev[ticker_i]):
            Log.info(f'Ticker \'{ticker_i}\' not listed in previous week ({date_i}).')
        else:
            try:
                index_prev = int(TickersList_prev[ ticker_i ])
                price_prev = researchwizzDB.iloc[ index_prev ]['CP']
            
                if np.isnan(price_i) or np.isnan(price_prev):
                    Log.info(f'Ticker \'{ticker_i}\' price was NAN in current week ({date_i}).')

                elif price_prev < 10e-9: # Price is probably 0, leave as nan
                    # We leave as nan
                    continue

                elif price_i > priceNonsense_trshld: # 
                    Log.info(f'Ticker \'{ticker_i}\' price was nonsense in current week ({date_i}). CP = {price_i}.')
                    blacklist[ ticker_i ] = row_i['Company']

                else:
                    relativeProfit = (price_i-price_prev) / price_prev
                    researchwizzDB.at[ index_prev, 'relative_profit' ] = relativeProfit
            except Exception as e:
                Log.info(f'Problem with ticker \'{ticker_i}\' in week {date_i}. i = {i}. Error: {e}')

        TickersList_i[ ticker_i ] = i


    # Print and save blacklist
    print('Tickers on blacklist: ', blacklist)
    with open(f'{blacklist_dir}/blacklist_tickers.json', 'w') as blacklist_f:
        json.dump(blacklist, blacklist_f)

    # Save database with relative profit

    Log.info('Saving database with relative profit.')

    db.createDatabase(path_to_database_rp)
    db_connection_rp = db.Connection(path_to_database_rp)
    db_connection_rp.dataframeToTable('vw_stock_data_vs2', researchwizzDB)

    Log.info(f'Database with relative profit saved to [{path_to_database_rp}].')


if __name__ == '__main__':
    calculate_profit(
        query = 'SELECT * FROM table_name',
        path_to_database = '.../data/researchwizz.db',
        path_to_database_rp = '.../0_Preprocessing/_output/researchwizz_rp.db',
        blacklist_dir = '.../0_Preprocessing/_output'
    )