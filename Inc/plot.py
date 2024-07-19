"""
Script: calculate_profit.py
Desc: Calculating relative profit on a table loaded from DB.
"""

root_directory = '/Users/vojtechremis/Desktop/bachelorproject/'

import ast
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np
import time

# Custom class
import sys
sys.path.append(root_directory+'Inc')
import log
Log = log.log()

# Custom style
plt.style.use(root_directory+'Inc/MatPlotLib_styles/classicChart.mplstyle')

def approximationVizualization(xlim, ylim, predict_func=None, indicatorNames=['Indicator1', 'Indicator2'], samples=30, manualSamples=None, showGrid=True, saveFig_path=None, saveFig_title=None, normMinMaxColor=(-1, 1), Title=None):
    """ 
    For visualization of approximation, two approaches are possible:
        Manual: Import a list of samples (rectangles [x_min, x_max, y_min, y_max]) and plot them onto the grid.
        Otherwise: Create a grid based on the number of samples and plot predictions onto it.

    I case of Manual: filePath / dataset ([axes, values]) and normMinMaxColor is required.
        - normMinMaxColor is minimum and maximum of values used when plotting SpecialVisualization of root function. The similarity of these numbers is needed for a correct comparison.
    """
    start_time = time.time()
    allowedDimension = 4
    title = None

    if samples == 'manual':
        if manualSamples is not None:
            if next(iter(manualSamples.keys())) == 'file':
                with open(manualSamples['file'], 'r') as file:
                    lines = file.readlines()

                rectanglesList = np.array([ast.literal_eval(line) for line in lines])
                
                rectanglesDimension = rectanglesList.shape[1]
                if rectanglesDimension == allowedDimension:
                    valuesToPlot = None
                elif rectanglesDimension == allowedDimension + 1:
                    valuesToPlot = rectanglesList[:, allowedDimension]
                    rectanglesList = rectanglesList[:, :allowedDimension]
                else:
                    log.error(f'Manually imported data have wrong dimension. Dimension required: [{allowedDimension}]. Dimension provided: [{rectanglesDimension}].')
                    return None
            

            elif next(iter(manualSamples.keys())) == 'dataset':
                rectanglesList, valuesToPlot =  manualSamples['dataset']

            plt.figure(figsize=(8, 6))

            # Show grid
            if showGrid:
                edgecolor = 'black'
            else:
                edgecolor = 'none'

            # Get norm for color normalization
            # Not necessarily in [min, max] since the normalization function is taken from the root function samples.

            valuesNorm = Normalize(vmin=normMinMaxColor[0], vmax=normMinMaxColor[1])

            index = 0
            for rectangleSample in rectanglesList:
                x_min, x_max, y_min, y_max = rectangleSample
                width = x_max - x_min
                height = y_max - y_min
                rectangleObject = Rectangle((x_min, y_min), width, height, edgecolor=edgecolor, facecolor='none', alpha=0.5)
                
                # If also Values are provided
                if predict_func is not None:
                    predictedValue = predict_func([x_min, x_max, y_min, y_max])[0]
                elif valuesToPlot is not None:
                    predictedValue = valuesToPlot[index]
                else:
                    log.error('No predict function / manual golden data provided!')

                rectangleObject.set_facecolor(plt.cm.plasma(valuesNorm(predictedValue)))
                plt.gca().add_patch(rectangleObject)

                index += 1

            # Add color bar
            sm = plt.cm.ScalarMappable(cmap='plasma', norm=valuesNorm)
            cbar = plt.colorbar(sm, label='Průměrná hodnota')

            # Plot attributes
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.suptitle('Vizualizace aproximace')
            
            if Title is None:
                title = f'Pomocí manuálně importovaných obdélnků.' # Default value
            else:
                title = Title
            plt.title(title)

            plt.xlabel(indicatorNames[0])
            plt.ylabel(indicatorNames[1])

            # Save fig
            if saveFig_path is not None:
                grid_str = '_grid' if showGrid else ''
                plt.savefig(f'{saveFig_path}/approximationTest_{saveFig_title}_{samples}_{grid_str}.pdf')

            plt.show()

        else:
            Log.warning('Manual samples were not provided!')

    else:
        x_size = ((xlim[1]-xlim[0]) / samples)
        y_size = ((ylim[1]-ylim[0]) / samples)

        X = np.linspace(xlim[0] + (x_size / 2), xlim[1] - (x_size / 2), num=samples)
        Y = np.linspace(ylim[0] + (y_size / 2), ylim[1] - (y_size / 2), num=samples)

        # Create meshgrid
        xx, yy = np.meshgrid(X, Y)

        plt.figure(figsize=(8, 6))

        plt.scatter(xx, yy, s=0.3)

        zz = np.zeros_like(xx)

        # Calculating predicted values
        for i in range(len(xx)):
            for j in range(len(yy)):
                x = xx[j, i]
                y = yy[j, i]
                x_min = x - x_size/2
                x_max = x + x_size/2
                y_min = y - y_size/2
                y_max = y + y_size/2
            
                zz[i, j] = predict_func([x_min, x_max, y_min, y_max])[0]

        # Plot
        # contour_plot = plt.contourf(xx, yy, zz, cmap='plasma')
        # plt.colorbar(contour_plot, label='Funkční hodnoty')

        # Not necessarily in [min, max] since the normalization function is taken from the root function samples.
        valuesNorm = Normalize(vmin=normMinMaxColor[0], vmax=normMinMaxColor[1])

        contour_plot = plt.contourf(xx, yy, valuesNorm(zz), cmap='plasma', levels=10)
        sm = plt.cm.ScalarMappable(cmap='plasma', norm=valuesNorm)
        cbar = plt.colorbar(sm, label='Funkční hodnoty')

        # Show grid lines
        if showGrid:
            for i in range(len(X)):
                plt.plot([X[i], X[i]], [Y[0], Y[-1]], color='black', linewidth=0.5)
            for j in range(len(Y)):
                plt.plot([X[0], X[-1]], [Y[j], Y[j]], color='black', linewidth=0.5)

        # Plot attributes
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.suptitle('Vizualizace aproximace')

        if Title is None:
            title = f'Pomocí čtvercové sítě o velikosti [{round(x_size, 3)} × {round(y_size, 3)}]' # Default value
        else:
            title = Title
        plt.title(title)

        plt.xlabel(indicatorNames[0])
        plt.ylabel(indicatorNames[1])

        # Save fig
        if saveFig_path is not None:
            grid_str = '_grid' if showGrid else ''
            plt.savefig(f'{saveFig_path}/approximationTest_{saveFig_title}_{samples}_{grid_str}.pdf')

        plt.show()
    Log.info(f"Execution time of [{title}]: {str(time.time() - start_time)} seconds")