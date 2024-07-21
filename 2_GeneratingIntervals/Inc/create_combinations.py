"""
Script: create_combinations.py
Desc: Generating training data based on multiple parameters.
"""

root_directory = '/Users/vojtechremis/Desktop/VŠ/BP/bachelorproject_git/bachelorproject/'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import csv
import random


#Importing shared modules from /Inc
import sys
sys.path.append(root_directory+'Inc')
import log
Log = log.log("Combinations_cropp_lessThenPercent_log")

# Použití vlastního stylu
plt.style.use(root_directory+'Inc/MatPlotLib_styles/classicChart.mplstyle')

class CreateCombinations:
    """
    Class which transforms Dataframe (which consists from Indicators and Relative profit) to randomly generated intervals
    """

    CDF = {}
    
    """
    Initialize CreateCombination Class

    Args:
        Dataset (pd.DataFrame): indicators and Relative Profit
        continuousIndicator_names (list): names of indicators with continuous format
        categoricalndicator_names (list): names of indicators with categorical format
        CDFFolderPath (str): path to a folder, where CDFs are saved
        intervalStretching (dict): Stretching interval ('default' is applied to every indicator, which isn't contained in intervalStretching dict),
        IntervalLenLimits (tupple): minimal and maximal relative size of an generated interval
        exportFolderPath (str): path to the folder, where result files have to be saved (if doesn't exist, it will be created)
        exportFileName (str): name of output file
        detailFileName (str): name of detail file
        tresholdFileFull: True - number of valid samples must be equal to numberOfSamples, False - number of any sample must be equal to numberOfSamples
    """
    def __init__(self, Dataset, numberOfSamples, continuousIndicator_names, categoricalndicator_names, samplesTreshold=5, CDFFolderPath='CDF', intervalStretching=None, IntervalLenLimits=(0.3, 0.9), exportFolderPath='CombinationsExport', exportFileName='Combinations_Dataset', detailFileName='Combinations_Detail', tresholdFileFull=False):
        
        # Process parameters
        self.continuousIndicator_names = continuousIndicator_names
        self.categoricalndicator_names = categoricalndicator_names
        self.indicatorNames = continuousIndicator_names+categoricalndicator_names 
        self.dataframe = Dataset
        self.intervalStretching = intervalStretching
        self.minRatio = IntervalLenLimits[0]
        self.maxRatio = IntervalLenLimits[1]
        self.tresholdFileFull = tresholdFileFull

        # Info parameters
        self.emptyCount = 0
        self.combinationCount = 0
        self.dataframe_length = len(Dataset)

        # Paths, folder and file names

        self.exportFolderPath = exportFolderPath
        self.exportFileName = exportFileName
        self.detailFileName = detailFileName

        # Classes
        self.loadCDFs(self.indicatorNames, CDFFolderPath)

        # Results
        self.intervalRange = {} # Range boundaries
        self.intervalPerc = {} # Percentage length of intervals on y-axis (cumsum axis)

        self.start(numberOfSamples=numberOfSamples, samplesTreshold=samplesTreshold)


    # Loading and manipulating with CDFs

    def loadCDFs(self, indicators, CDFFolderPath):
        for indicator in indicators:
            try:
                indicatorCDF = pd.read_csv(f'{CDFFolderPath}/CDF_{indicator}.csv', sep=';')

                self.CDF[indicator] = indicatorCDF
                Log.info(f'Indicator {indicator} CDF loaded with size {len(indicatorCDF)}')
            except Exception as e:
                Log.warning(f'Error when loading CDFs accured: {e}')

    def getCDFX_byCumsum(self, indicator, cumsum):
        cdf = self.CDF[indicator]

        return cdf.loc[cdf['Cumsum'] == cumsum, 'Value'].tolist()[0]
    
    def getCDFCumSum_byIndex(self, indicator, index):
        cdf = self.CDF[indicator]
        try:
            cumsum = cdf.iloc[index].tolist()[0]
        except Exception as e:
            Log.error(f'Function getCDFCumSum_byIndex() : {indicator} index={index} with Error: {e}')
            cumsum = 0

        return cumsum
    

    def getIndicatorRndOrdered(self):
        randomPermutation = np.random.permutation(18)
        indicators = self.indicatorNames.copy()
        return [indicators[i] for i in randomPermutation]

    def returnIndexInterval(self, bins):
        """
        Returns interval boundaries [ind_min, ind_max], but with conditions: interval must be relatively larger than minRatio and smaller than maxRatio

        Steps:
        1. Convert minRatio, maxRation to terms of bins (bins is number of rows in discrete empirical CDF).
        2. Randomly choose first bin.
        3. Choose scenario:
            a) There is space on left and right side of the bin, so you can randomly place there another point and you will fulfill conditions
            b) The bin is near the left boundary of the indicator axis, so you must randomly place another point only on the right side of the interval.
            c) The bin is near the right boundary of the indicator axis, so you must randomly place another point only on the left side of the interval.
            
            Based on scenarios above, you get index of the second bin (assumption is made that the CDF is uniformly sampled, so indexes of bins relatively correspond to their absolute value).
        4. Sort interval boundaries.
        5. Return [bins].
        """

        # Minimum and maximum of bins, which can selection contain
        minAbsolute = math.floor(self.minRatio*bins-1)
        maxAbsolute =  math.ceil(self.maxRatio*bins-1)

        # Subinterval length
        subinterval_length = random.randint(minAbsolute, maxAbsolute)

        # Possible starting values
        start_min = 0
        start_max = bins - 1 - subinterval_length

        start = random.randint(start_min, start_max)
        end = start + subinterval_length
        
        return (start, end)

    def getCombination(self):
        """
        Return [indicator_min, indicator_max] (absolute indicator boundaries) based on [ind_min, ind_max] (index indicator boundaries).

        From method returnIndexInterval() you get boundaries of an interval [ind_min, ind_max].
        Then apply stretching (randomly stretch interval if selected). This can be valuable when another form of error required.

        Note: When dealing with categorical indicators, you need to provide special interval selection function
        """

        for indicator in self.indicatorNames:

            # If indicator has continuous format
            if indicator in self.continuousIndicator_names:

                # Each indicator may have different bin size due to empty selections
                minIndex, maxIndex = self.returnIndexInterval(
                    len(self.CDF[indicator])
                )

                # Pair minIndex and maxIndex with corresponding value
                leftBoundary = self.getCDFCumSum_byIndex( indicator, minIndex )
                rightBoundary = self.getCDFCumSum_byIndex( indicator, maxIndex )

                stretch = []
                if self.intervalStretching:
                    # If indicator has custom interval stretching setting
                    if indicator in self.intervalStretching.keys():
                        stretch = self.intervalStretching[indicator]
                    else:
                        stretch = self.intervalStretching['default']
                    
                    intervalLength = rightBoundary-leftBoundary
                    leftBoundary -= intervalLength*stretch[0]
                    rightBoundary += intervalLength*stretch[0]
                    # leftBoundary -= intervalLength*( np.random.uniform(low=0, high=stretch[0]) )
                    # rightBoundary += intervalLength*( np.random.uniform(low=0, high=stretch[0]) )

                # How big is selected interval
                self.intervalPerc[indicator] = round((maxIndex-minIndex)/len(self.CDF[indicator]),2)


            # If indicator has categorical format
            if indicator in self.categoricalndicator_names:
                if indicator == 'ZR':
                    ZR_intervals = [[1, 2], [2, 4], [4, 5]]
                    leftBoundary, rightBoundary = ZR_intervals[np.random.randint(low=0, high=2)]
                else:
                    Log.error(f'You need to provide indicator selection function for categorical indicator [{indicator}].')

            
            self.intervalRange[indicator] = [leftBoundary, rightBoundary]

    def createSelectionCombination(self) -> None:
        """
        Firstly this method gets interval boundaries using method getCombination()
        Afterwards it filters dataframe using these intervals.
        Finaly, it saves test dataset used in Neural Network training
        """

        # getCombination() saves interval boundaries pairs for each indicator into self.intervalRange
        self.getCombination()

        dataframeFilter = pd.Series( data=np.full(shape=self.dataframe_length, fill_value=True) )

        # For each indicator, filter dataframe
        for indicator in self.intervalRange.keys():
            dataframeFilter = dataframeFilter & (self.dataframe[indicator] >= self.intervalRange[indicator][0]) & (self.dataframe[indicator] <= self.intervalRange[indicator][1])

        dataframe_filtered = self.dataframe.loc[dataframeFilter]


        # Writing detail info

        entryDetail = {}
        for indicator in self.indicatorNames:
            entryDetail[indicator+'_min'] = self.intervalRange[indicator][0]
            entryDetail[indicator+'_max'] = self.intervalRange[indicator][1]
            entryDetail[indicator+'_relLen'] = self.intervalPerc[indicator]
        
        entryDetail['number_of_datapoints'] = len(dataframe_filtered)
        detail_isEmpty = False
        relative_profit = None

        

        # If generated intervals does not correspond to any datapoint
        if dataframe_filtered.empty:
            self.emptyCount += 1
            Log.info(f'Total number of empty selections = {self.emptyCount}')
        
        # If generated intervals does correspond to a datapoint
        else:
            count_filtered = dataframe_filtered['relative_profit'].count()
            relative_profit = dataframe_filtered['relative_profit'].mean()

            self.combinationCount += 1

            result = {}
            for indicator_name, indicator_min_max in self.intervalRange.items():
                result[indicator_name+'_min'] = indicator_min_max[0]
                result[indicator_name+'_max'] = indicator_min_max[1]

            result['relative_profit'] = relative_profit

            if(count_filtered >= self.samplesTreshold):
                # Write result
                self.aboveTreshold_count +=1
                self.appendRowToCSVFile(self.tresholdOutpuFilePrimaryPath, result, self.columnsExport)

            else:
                # Write result
                self.appendRowToCSVFile(self.tresholdOutputFileSeconaryPath, result, self.columnsExport)

        
        entryDetail['relative_profit'] = relative_profit
        entryDetail['is_empty'] = detail_isEmpty

        # Write detail Combination information
        self.appendRowToCSVFile(self.detailFilePath, entryDetail, self.columnsDetail)

    def createCSVFile(self, pathToFile, header):
        try:
            if not os.path.exists(pathToFile):
                with open(pathToFile, 'w', newline='') as detailFileCSV:
                    writer = csv.DictWriter(detailFileCSV, fieldnames=header)
                    writer.writeheader()
            else:
                Log.warning(f"File [{pathToFile}] already exists.")
        except Exception as e:
            Log.error(f'Error when creating csv file. Error: {e}')

    def appendRowToCSVFile(self, pathToFile, row, header):
        try:
            with open(pathToFile, 'a', newline='') as detailFileCSV:
                writer = csv.DictWriter(detailFileCSV, fieldnames=header)
                writer.writerow(row)
        except Exception as e:
            Log.error(f'When saving to CSV [Combination_count = {self.combinationCount}] with error: {e}')



    def start(self, numberOfSamples, samplesTreshold):
        
        # When saving Combinations, you can separate interval selections by number of samples filtered
        self.samplesTreshold = samplesTreshold

        self.tresholdOutpuFilePrimaryPath = f'{self.exportFolderPath}/{self.exportFileName}__trhd_>=_{samplesTreshold}.csv'
        self.tresholdOutputFileSeconaryPath = f'{self.exportFolderPath}/{self.exportFileName}__trhd_<_{samplesTreshold}.csv'
        self.detailFilePath = f'{self.exportFolderPath}/{self.detailFileName}.csv'

        # Create file for Combinations in detail
        columns = []
        columnsExport = []
        for indicator in self.indicatorNames:
            columns += [indicator+'_min', indicator+'_max', indicator+'_relLen']
            columnsExport += [indicator+'_min', indicator+'_max']

        columns += ['relative_profit', 'number_of_datapoints', 'is_empty']
        columnsExport += ['relative_profit']

        self.columnsDetail = columns
        self.columnsExport = columnsExport


        # Create output files
        self.createCSVFile(self.tresholdOutpuFilePrimaryPath, header=self.columnsExport)
        self.createCSVFile(self.tresholdOutputFileSeconaryPath, header=self.columnsExport)
        self.createCSVFile(self.detailFilePath, header=self.columnsDetail)

        # Run loop
        if self.tresholdFileFull:
            self.aboveTreshold_count = 0
            index = 0

            while self.aboveTreshold_count < numberOfSamples:
                self.createSelectionCombination()
                if index%1000==0: # 
                    Log.info(f'Entry [{index}] finished.')
                index += 1
        else:
            for index in range(numberOfSamples):
                self.createSelectionCombination()
                if index%1000==0: # 
                    Log.info(f'Entry [{index}] finished.')


if __name__ == '__main__':
    intervalStretching = {
        'default': [0.05, 0.05]
    }

    syntheticFunction = np.load(root_directory+'/1_GenerateFunction/generated_data/subsetSamples_test.npz', allow_pickle=True)
    df = pd.DataFrame({'synthetic_indicator_one': syntheticFunction[0][0], 'synthetic_indicator_nd': syntheticFunction[0][1], 'relative_profit': syntheticFunction[1]})


    cm = CreateCombinations(
        Dataset=df,
        continuousIndicator_names=['synthetic_indicator_one', 'synthetic_indicator_nd'],
        categoricalndicator_names=[],
        CDFFolderPath=root_directory+'2_GeneratingIntervals/FirstSampling/CDF',
        intervalStretching=intervalStretching,
        IntervalLenLimits=(0.3, 0.9),
        exportFolderPath='CombinationExport_',
        exportFileName='Combinations_Dataset',
        detailFileName='Combinations_Detail'
    )