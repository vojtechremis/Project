"""
Script: outliers.py
Desc: Performs Z-Score analysis and saves non-outliers boundaries for each indicator to a .json file. Saves also plots.
Performs missing data analysis and saves matrix Weeks x Indicator (more is discussed in Thesis).
"""

root_directory = '/Users/vojtechremis/Desktop/VŠ/BP/bachelorproject_git/bachelorproject/'

import numpy as np
import pandas as pd
import json
import sys
from matplotlib import pyplot as plt
sys.path += [root_directory+'Inc']
import log
Log = log.log('log_outliers.txt')

import database as db
from FolderManagement import createDirectory

# Using custom style
plt.style.use(root_directory+'Inc/MatPlotLib_styles/classicChart.mplstyle')


# Z-Score analysis
def perform_zscore(query, path_to_database, output_dir, z_score_thrshld = 3):
    Log.info('Loading database...')
    connectionToSourceDB = db.Connection(path_to_database)
    researchwizzDB = connectionToSourceDB.tableToDF(query)
    if researchwizzDB is None:
        return
    
    numberColumns = []
    for column in researchwizzDB.columns:
        try:
            researchwizzDB[column] = researchwizzDB[column].astype(float)
            numberColumns.append(column)
        except Exception as e:
            Log.warning(f'Column [{column}] could not be transformed to float.')

    Log.info('Database has been loaded.')

    # Keep only columns including number
    researchwizzDB = researchwizzDB[numberColumns]

    # Computing Z-Score
    Mean = np.mean(researchwizzDB, axis=0)
    STD = np.std(researchwizzDB, axis=0)

    # researchwizzDB_zScore = (researchwizzDB - Mean) / STD
    outliers_lowerBounds = Mean - z_score_thrshld * STD
    outliers_upperBounds = Mean + z_score_thrshld * STD

    # Create plot dir
    plotDirectory = output_dir+'/plot'
    createDirectory(plotDirectory)

    # Save filter conditions for each indicator
    outliers_conditions = {}
    for indicator in researchwizzDB.columns:
        outliers_lowerBounds_ind = outliers_lowerBounds[indicator]
        outliers_upperBounds_ind = outliers_upperBounds[indicator]
        outliers_conditions[indicator] = [outliers_lowerBounds_ind, outliers_upperBounds_ind]

        if researchwizzDB[indicator].notna().any():
        
            #Save plot visuals
            min_value = np.nanmin(researchwizzDB[indicator])
            max_value = np.nanmax(researchwizzDB[indicator])

            hist, bins = np.histogram(researchwizzDB[indicator], bins=10000, range=(min_value, max_value))
            cumsum = np.cumsum(hist)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            plt.plot(bin_centers, cumsum, label=f'CDF {indicator}', linewidth=2, color='skyblue')
            
            # Plot the clippers on value axis
            plt.axvline(x=outliers_lowerBounds_ind, color='red', linestyle='dashed', label='ZScore - dolní', linewidth=1)
            plt.axvline(x=outliers_upperBounds_ind, color='green', linestyle='dashed', label='ZScore - horní', linewidth=1)

            # Plot clippers on histogram axis
            lower_hist_clipper = (researchwizzDB[indicator] <= outliers_lowerBounds_ind).sum()
            upper_hist_clipper = (researchwizzDB[indicator] <= outliers_upperBounds_ind).sum()
            
            # Plot the bounds on the y-axis
            plt.axhline(lower_hist_clipper, color='red', linestyle='dashed', linewidth=1)
            plt.axhline(upper_hist_clipper, color='green', linestyle='dashed', linewidth=1)
    


            
            plt.title(f'Komulativní histogram hodnot indikátoru {indicator}')
            plt.xlabel(f'Hodnota indikátoru {indicator}')
            plt.ylabel('Komulativní histogram')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.savefig(f'{plotDirectory}/CH_ZSCORE_indicator_{indicator}.svg')
            plt.clf()



            # Save plot clipped visual
            indicator_clipped = researchwizzDB.loc[((researchwizzDB[indicator] >= outliers_lowerBounds_ind) & (researchwizzDB[indicator] <= outliers_upperBounds_ind)), indicator]
            min_value_ = np.nanmin(indicator_clipped)
            max_value_ = np.nanmax(indicator_clipped)

            hist_, bins_ = np.histogram(indicator_clipped, bins=10000, range=(min_value_, max_value_))
            cumsum_ = np.cumsum(hist_)
            bin_centers_ = (bins_[:-1] + bins_[1:]) / 2
            
            plt.plot(bin_centers_, cumsum_, label=f'CDF {indicator}', linewidth=2, color='royalblue')
            plt.title(f'Komulativní histogram hodnot indikátoru {indicator} (po odstranění outlierů)')
            plt.xlabel(f'Hodnota indikátoru {indicator}')
            plt.ylabel('Komulativní histogram')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.savefig(f'{plotDirectory}/CH_ZSCORE_indicator_{indicator}-nooutlier.svg')
            plt.clf()


        else:
            Log.warning(f'Indicator \'{indicator}\' has only nan values.')

    # Save outlier conditions
    with open(f'{output_dir}/outliers_conditions.json', 'w') as outliers_f:
        json.dump(outliers_conditions, outliers_f)


# Data gap analysis
def missing_analysis(query, path_to_database, output_dir):
    Log.info('Loading database...')
    connectionToSourceDB = db.Connection(path_to_database)
    researchwizzDB = connectionToSourceDB.tableToDF(query)

    if researchwizzDB is None:
        return
    
    numberColumns = []
    for column in researchwizzDB.columns:
        try:
            researchwizzDB[column] = researchwizzDB[column].astype(float)
            numberColumns.append(column)
        except Exception as e:
            Log.warning(f'Column [{column}] could not be transformed to float.')


    columns = numberColumns + ['Ticker', 'export_date']

    researchwizzDB.sort_values(by='export_date', inplace=True)

    Log.info('Database has been loaded.')


    
    # Keep only columns including number + ticker
    researchwizzDB = researchwizzDB[columns]


    # Missing values
    IndWeeksMatrix_countNAN = pd.DataFrame(index = researchwizzDB['export_date'].drop_duplicates(), columns=numberColumns)
    IndWeeksMatrix_countNAN[:] = 0

    IndWeeksMatrix_countValues = pd.DataFrame(index = researchwizzDB['export_date'].drop_duplicates(), columns=numberColumns)
    IndWeeksMatrix_countValues[:] = 0

    print('IndWeeksMatrix_countNAN.shape', IndWeeksMatrix_countNAN.shape)
    print('IndWeeksMatrix_countValues.shape', IndWeeksMatrix_countValues.shape)

    Ind_msngOrNot = pd.DataFrame(index = numberColumns)
    Ind_msngOrNot['coutNAN'] = 0
    Ind_msngOrNot['countValues'] = 0

    week_prev = None
    for index, row in researchwizzDB.iterrows():
        export_date = row['export_date']

        # If we enter different week, save previous
        if week_prev != export_date:
                
            # Save values to matrices
            if week_prev is not None:
                for indicator in Ind_msngOrNot.index:
                    IndWeeksMatrix_countNAN.at[week_prev, indicator] = Ind_msngOrNot.at[indicator, 'coutNAN']
                    IndWeeksMatrix_countValues.at[week_prev, indicator] = Ind_msngOrNot.at[indicator, 'countValues']
            
            # Reset temporary lists
            Ind_msngOrNot['coutNAN'] = 0
            Ind_msngOrNot['countValues'] = 0
            week_prev = export_date
        else:
            hot_vector_nan = row[ numberColumns ].apply(lambda x: int(np.isnan(x)))
            hot_vector_val = row[ numberColumns ].apply(lambda x: 1 - int(np.isnan(x)))

            Ind_msngOrNot['coutNAN'] += hot_vector_nan
            Ind_msngOrNot['countValues'] += hot_vector_val
    

    

    IndWeeksMatrix_countAll = IndWeeksMatrix_countValues+IndWeeksMatrix_countNAN
    IndWeeksMatrix_rel = np.divide(IndWeeksMatrix_countNAN, IndWeeksMatrix_countAll, where=IndWeeksMatrix_countAll != 0)

    # Create plot dir
    plotDirectory = output_dir+'/plot'
    createDirectory(plotDirectory)

    IndWeeksMatrix_rel.to_csv(plotDirectory+'/IndWeeksMatrix_rel.csv')

    IndWeeksMatrix_countValues.to_csv(plotDirectory+'/IndWeeksMatrix_countValues.csv')
    IndWeeksMatrix_countNAN.to_csv(plotDirectory+'/IndWeeksMatrix_countNAN.csv')


perform_zscore(
    query = 'SELECT * FROM table_name',
    path_to_database = '.../0_Preprocessing/_output/researchwizz_rp.db', # Database must contain relative_profit column
    output_dir = root_directory+'0_Preprocessing/2_Outliers/_output'
)

missing_analysis(
    query = 'SELECT * FROM table_name',
    path_to_database = '.../0_Preprocessing/_output/researchwizz_rp.db', # Database must contain relative_profit column
    output_dir = root_directory+'0_Preprocessing/2_Outliers/_output'
)