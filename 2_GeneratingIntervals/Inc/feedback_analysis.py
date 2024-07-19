"""
Script: calculate_histogram.py
Desc: It basically makes some additional visual analysis to allready created combinations (plots are being saved to corresponding folders).
"""

root_directory = '/Users/vojtechremis/Desktop/bachelorproject/'

import pandas as pd
from matplotlib import pyplot as plt

import sys
sys.path.append(root_directory+'Inc')
from FolderManagement import createDirectory
import log

# Custom matplotlib style
plt.style.use(root_directory+'Inc/MatPlotLib_styles/classicChart.mplstyle')

class feedBackAnalysis:
    def __init__(self, sampledFunction, combinationsDetail, logFilePath, Log_instance) -> None:
        self.first_indicator = sampledFunction['axes'][0]
        self.second_indicator = sampledFunction['axes'][1]
        self.rp = sampledFunction['values']
        self.combinationsDetail = combinationsDetail

        # Create backAnalysis output folder
        self.outputFolder = logFilePath+'/feedBackAnalysis'
        Log_instance.info(createDirectory(self.outputFolder))

        self.Log_instance = log.log(self.outputFolder+'/Log')
        self.Log_instance.info(f'\nAnalysing experiment: [{logFilePath}].')
    
    def samplesInIntervals(self):
        plt.hist(self.combinationsDetail['number_of_datapoints'], bins=50, color='lightgreen', edgecolor='black')
        plt.title('Počet vzorků ve vybraném intervalu (histogram)')
        plt.xlabel('Počet vzorků ve vybraném intervalu')
        plt.ylabel('Počet intervalů')
        plt.savefig(f'{self.outputFolder}/samplesInIntervals_hist.pdf')
        plt.show()

    def checkCombinations(self):
        dataframe_forCheck = pd.DataFrame({'x': self.first_indicator, 'y': self.second_indicator, 'z': self.rp})

        # Now, combinations_samples contains 100 randomly sampled rows from your DataFrame
        combinations_samples = self.combinationsDetail.sample(n=100, random_state=42)

        for index, row in combinations_samples.iterrows():
            row['synthetic_indicator_one_min']
            row['synthetic_indicator_one_max']
            row['synthetic_indicator_nd_min']
            row['synthetic_indicator_nd_max']
            relative_profit = dataframe_forCheck[ (dataframe_forCheck['x'] >= row['synthetic_indicator_one_min']) & (dataframe_forCheck['x'] < row['synthetic_indicator_one_max']) & (dataframe_forCheck['y'] >= row['synthetic_indicator_nd_min']) & (dataframe_forCheck['y'] < row['synthetic_indicator_nd_max'])]['z'].mean()
            
            if row['relative_profit'] != relative_profit:
                diff = row['relative_profit'] - relative_profit
                if diff > 1e-15:
                    self.Log_instance.warning(f'[{diff}] difference!')

    def emptyIntervals(self):
        numberOfEmptySelections = self.combinationsDetail['is_empty'].sum()
        self.Log_instance.info(f'Number of empty selections: {numberOfEmptySelections}')

    def relativeIntervalLength(self):
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

        fig.suptitle('Relativní délka intervalu (histogram)', y=0.95)

        axs[0].hist(self.combinationsDetail['synthetic_indicator_one_relLen'], bins=50, color='skyblue', edgecolor='black')
        axs[0].set_title('Indikátor synthetic_indicator_one')
        axs[0].set_xlabel('Relativní délka intervalu')
        axs[0].set_ylabel('Počet intervalů')

        axs[1].hist(self.combinationsDetail['synthetic_indicator_nd_relLen'], bins=50, color='red', edgecolor='black')
        axs[1].set_title('Indikátor synthetic_indicator_nd')
        axs[1].set_xlabel('Relativní délka intervalu')
        axs[1].set_ylabel('Počet intervalů')

        # Adjust layout to prevent overlapping of labels
        plt.tight_layout()

        # Save the plot
        plt.savefig(f'{self.outputFolder}/relativeIntervalLength_hist.pdf')

        # Show the plot
        plt.show()

    def relativeProfitHist(self):
        plt.hist(self.combinationsDetail['relative_profit'], bins=50, color='green', edgecolor='black')
        plt.title('Relativní profit (histogram)')
        plt.xlabel('Hodnota relativního profitu')
        plt.ylabel('Počet intervalů')
        
        # Save the plot
        plt.savefig(f'{self.outputFolder}/relativeProfit_hist.pdf')

        plt.show()