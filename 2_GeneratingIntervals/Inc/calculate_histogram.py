"""
Script: calculate_histogram.py
Desc: Calculating histogram and CDF of data provided by generate_function module.
"""

root_directory = '/Users/vojtechremis/Desktop/bachelorproject/'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#Importing shared modules from /Inc
import sys
sys.path.append(root_directory+'Inc')
sys.path.append(root_directory+'1_GenerateFunction')

import generate_function
import log
Log = log.log()

# Použití vlastního stylu
plt.style.use(root_directory+'Inc/MatPlotLib_styles/classicChart.mplstyle')

# Class
class Distribution:
    def __init__(self, FundamentalDataset=None, fileSaveDir='DefaultExportFile') -> None:
        self.FundamentalDataset = FundamentalDataset
        self.fileSaveDir = fileSaveDir
        
        if fileSaveDir is not None:
            self.createDirectory(fileSaveDir)

    
    def createDirectory(self, pathToDirectory):
        try:
            if not os.path.exists(pathToDirectory):
                os.makedirs(pathToDirectory)
                Log.info(f'Folder has been created [{pathToDirectory}].')
        except Exception as e:
            Log.error(f'Error when creating folders. Error: {e}')



    def Histogram(self, indicatorName=None, indicatorColumn=None, saveFile=False, subFolderPath='HISTOGRAMS', filePrefix='Histogram_') -> pd.DataFrame:
        
        self.createDirectory(self.fileSaveDir+'/'+subFolderPath)
        
        if indicatorName:
            if self.FundamentalDataset is not None:
                indicatorToProcces = self.FundamentalDataset[indicatorName].tolist()
            elif indicatorColumn is not None:
                indicatorToProcces = indicatorColumn
            else:
                Log.error('Neither FundamentalDataset or indicatorColumn not provided.')
                return None
        else:
            Log.error('You must provide indicator name.')
        
        indicatorToProcces_sorted = self.Sort(indicatorToProcces, indicatorName)
        indicatorToProcces_len = len(indicatorToProcces_sorted)

        Values = []
        Frequencies = []

        partialCount = 1
        previousValue = None

        # Writing into Dataframe is delayed by 1 (if first value [0] is unique, it will be written on i=1)
        index=0
        for currentValue in indicatorToProcces_sorted:

            # Log every 100k-th row
            if index%100000 == 0:
                Log.info('Processing row '+str(index)+' out of '+str(indicatorToProcces_len))

            # If index = 0
            if index == 0:
                previousValue = currentValue
                index+=1
                continue
            
            # If previous value was different, write it into list
            if (currentValue != previousValue) and not (np.isnan(currentValue) and np.isnan(previousValue)):

                Values.append(previousValue)
                Frequencies.append(partialCount)

                # Clean for new value
                partialCount = 1
                previousValue = currentValue
            else:
                partialCount += 1
            
            index+=1
        

        Values.append(previousValue)
        Frequencies.append(partialCount)

        Histogram = pd.DataFrame(data={'Value': Values, 'Frequency': Frequencies}, columns=['Value', 'Frequency'])

        if saveFile:
            Histogram.to_csv(self.fileSaveDir+'/'+subFolderPath+'/'+filePrefix+indicatorName+'.csv', sep=';', index=False)

        return Histogram

    def Sort(self, indicatorColumn, indicatorName):
        Log.info('Sorting ' + indicatorName)
        return sorted(indicatorColumn)

    def CDF(self, indicatorName=None, indicatorHistogram=None, bins='ini', saveFile=False, subFolderPath='CDF', filePrefix='CDF_') -> pd.DataFrame:
        
        self.createDirectory(self.fileSaveDir+'/'+subFolderPath)

        if bins == 'ini':
            self.createDirectory(self.fileSaveDir+'/'+subFolderPath+'/ini')

        Log.info('Calculating CDF for '+indicatorName)

        # Calculate histogram
        cumsums_ini = []
        intervals_ini = []
        cumsum = 0

        for index, row in indicatorHistogram.iterrows():
            if np.isnan(row['Value']):
                continue

            intervals_ini.append(row['Value'])
            
            cumsum += row['Frequency']
            cumsums_ini.append(cumsum)
    
        # Plot Initial CDF
        plt.title('Empirická CDF pro '+indicatorName)
        plt.ylabel('$ P(X \leq x) $')
        plt.xlabel('x')
        plt.plot(intervals_ini, cumsums_ini)
        plt.show()
        plt.clf()

        CDF_ini = pd.DataFrame(data={'Value': intervals_ini, 'CDF': cumsums_ini}, columns=['Value', 'CDF'])


        if bins == 'ini':
            if saveFile:
                CDF_ini.to_csv(self.fileSaveDir+'/'+subFolderPath+'/ini/'+filePrefix+indicatorName+'.csv', sep=';', index=False)
            return CDF_ini
        else:

            # Calculate reduced CDF
            min = CDF_ini['CDF'].min()
            max = CDF_ini['CDF'].max()
            Range = max-min

            step = Range/bins

            cumsums = [] # rescaled cumsums
            intervals = [] # rescaled intervals
            cumsum = 0
            boundary = min

            for i in range(bins+1):

                CDFsInRange = CDF_ini.loc[ (CDF_ini['CDF'] >= boundary ) & (CDF_ini['CDF'] < boundary+step ), 'CDF']
                
                if len(CDFsInRange) != 0: # If selection isn't empty
                    sum = CDFsInRange.iloc[len(CDFsInRange)-1] # Sum is the last CDF value (y-axis)
                
                    # Getting most near left Value
                    rightInDataBoundary = CDFsInRange.iloc[len(CDFsInRange)-1]
                    values = CDF_ini.loc[CDF_ini['CDF'] == rightInDataBoundary, 'Value']
                    rightIntervalBoundary = values.iloc[len(values)-1]

                    cumsum = sum
                    cumsums.append(cumsum)
                    intervals.append(rightIntervalBoundary)
                    
                boundary += step # Update boundary in both cases

            # Plot discrete CDF
            plt.title('Empirická CDF pro '+indicatorName)
            plt.ylabel('\( P(X \leq x) \)')
            plt.xlabel('x')
            plt.plot(intervals, cumsums)
            plt.show()
            plt.clf()

            CDF = pd.DataFrame(data={'Value': intervals, 'CDF': cumsums}, columns=['Value', 'CDF'])
            
            if saveFile:
                CDF.to_csv(self.fileSaveDir+'/'+subFolderPath+'/'+filePrefix+indicatorName+'.csv', sep=';', index=False)

            return CDF


    def plot(self, savePlot=False):
        pass

if __name__ == '__main__':
    # Load data
    SampledFunctionPath = root_directory+'1_GenerateFunction/generated_data'
    SampledFunction = generate_function.SyntheticFunction(
        dimension=2,
        rootToLoad=SampledFunctionPath+'/rootSamples_thesis.npz',
        samplesToLoad=SampledFunctionPath+'/subsetSamples_thesis.npz'
    )

    FundamentalData = pd.DataFrame(
        {'synthetic_indicator_one': SampledFunction.subset_axes[0],
        'synthetic_indicator_nd': SampledFunction.subset_axes[1],
        'relative_profit': SampledFunction.subset_values}
    )

    D = Distribution(FundamentalData, root_directory+'2_GeneratingIntervals/FirstSampling')
    # First indicator
    Hist = D.Histogram(indicatorName='synthetic_indicator_one', indicatorColumn=None, saveFile=True)
    CDF_ini = D.CDF(indicatorName='synthetic_indicator_one', indicatorHistogram=Hist, bins='ini', saveFile=True)
    CDF = D.CDF(indicatorName='synthetic_indicator_one', indicatorHistogram=Hist, bins=50, saveFile=True)

    # Second indicator
    Hist = D.Histogram(indicatorName='synthetic_indicator_nd', indicatorColumn=None, saveFile=True)
    CDF_ini = D.CDF(indicatorName='synthetic_indicator_nd', indicatorHistogram=Hist, bins='ini', saveFile=True)
    CDF = D.CDF(indicatorName='synthetic_indicator_nd', indicatorHistogram=Hist, bins=50, saveFile=True)