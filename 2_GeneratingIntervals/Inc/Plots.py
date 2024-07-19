"""
Script: Plots.py
Desc: Class providing combination plot, i.e. rectangles visible on background function image.
"""

root_directory = '/Users/vojtechremis/Desktop/bachelorproject/'

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random
import pandas as pd
import numpy as np

# Use custom style
plt.style.use(root_directory+'Inc/MatPlotLib_styles/classicChart.mplstyle')

#Importing shared modules from /VojtaWork/inc
import sys
sys.path.append(root_directory+'Inc')
import log
Log = log.log()

def plotCombinationGrid2D(indicator_names, indicator_names_plot, generatedDataset_initial, generatedDataset_sampled, combinationsDataset, probabilityToPrint, savePlotPath=None, generatedDataset_scatter=None):  # [[indicator_min, indicator_max]..., relative_profit]
    """
    Only two indicators required.
    """

    if len(indicator_names) == 2:
        x_name = indicator_names[0]
        y_name = indicator_names[1]

        plt.figure(figsize=(10, 6))
        fig, ax = plt.subplots()

        # If full generated grid dataset is passed, plot contour plot
        if generatedDataset_initial is not None:
            axes = generatedDataset_initial['axes']
            z = generatedDataset_initial['values']
        
            if len(axes) == 2:
                contour_plot = ax.contourf(*axes, z, cmap='plasma', levels=10)
                plt.colorbar(contour_plot, ax=ax, label='Funkce')
            else:
                Log.warning('Full generated grid dataset dimension is not 2D.')

        if generatedDataset_scatter is not None:
            x_sampled = generatedDataset_scatter[x_name]
            y_sampled = generatedDataset_scatter[y_name]
            scatter_ = ax.scatter(x_sampled, y_sampled, c=generatedDataset_scatter['relative_profit'], cmap='plasma', s=1)
            plt.colorbar(scatter_, label='Relativní profit')

        # If also sampled dataset is passed, scatter samples
        if generatedDataset_sampled is not None:
            x_sampled = generatedDataset_sampled[x_name]
            y_sampled = generatedDataset_sampled[y_name]
            ax.scatter(x_sampled, y_sampled, color='black', s=5, label='Vzorkování funkce')


        x_intervals = combinationsDataset[x_name]
        y_intervals = combinationsDataset[y_name]
        relative_profits = combinationsDataset['relative_profit']

        # Set limits by passed datasets (full grid generated dataset has top priority)
        if generatedDataset_sampled is not None:
            x_min, x_max = x_sampled.min(), x_sampled.max()
            y_min, y_max = y_sampled.min(), y_sampled.max()
        else:
            x_min, x_max = x_intervals[:, 0].min(), x_intervals[:, 1].max()
            y_min, y_max = y_sampled.min(), y_sampled.max()

        # Set plot limits
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        # Print rectangles
        for i in range(len(x_intervals)):

            if random.random() < probabilityToPrint: # Don't display all rectangles if requested
                interval_x = x_intervals[i]
                interval_y = y_intervals[i]
                relative_profit = relative_profits[i]

                width = interval_x[1] - interval_x[0]
                height = interval_y[1] - interval_y[0]
                x = interval_x[0]
                y = interval_y[0]

                # Create rectangle
                rect = Rectangle((x, y), width, height, linewidth=1, edgecolor='black', facecolor='none')
                ax.add_patch(rect)

        # Set labels and title
        ax.set_xlabel(indicator_names_plot[0])
        ax.set_ylabel(indicator_names_plot[1])
        ax.set_title('Vygenerované kombinace intervalů pro dva indikátory')
        # Show plot
        plt.show()

        if savePlotPath:
            plt.savefig(f'{savePlotPath}/combinationVisualPlot.pdf')

    else:
        Log.warning('Plotting combinations requires just 2 indicators (2D function).')

if __name__ == '__main__':
    # Plotting
    print('Jobs done!')