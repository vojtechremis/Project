"""
Script: app.py
Desc: This mini-app allows user to pick rectangles manually annd export them to a .txt file.
"""

root_directory = '/Users/vojtechremis/Desktop/bachelorproject/'

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
import numpy as np


class Rectangles:
    def __init__(self, xlim, ylim, outputFilePath='rectanglesOutput') -> None:
        # Create a plot with empty grid
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_aspect('equal')
        self.ax.grid(True)

        # Initialize variables for rectangle creation
        self.rectangle = None
        self.rectangles = []
        self.start_point = None

        # Define output file path
        self.outputFilePath = outputFilePath

        # Number of rectangles
        self.rectangles_counter = 0

        # Making space for buttons
        self.fig.set_size_inches(6, 8)

    def update_rectangles_counter(self):
        self.rectangles_counter = len(self.rectangles)
        self.rectangles_text.set_text(f'Number of rectangles: {self.rectangles_counter}')
        plt.draw()

    def on_press(self, event):
        """
        Save starting point on press of the left mouse button.
        """
        if event.button == 1:
            self.start_point = (event.xdata, event.ydata)

    def on_release(self, event):
        """
        Save whole rectangle on release of the left mouse button.
        """
        if event.button == 1: 
            if self.rectangle is not None:

                x_min, y_min, width, height = self.rectangle.get_bbox().bounds
                x_max = x_min + width
                y_max = y_min + height

                try:
                    value = self.intervalAvgValue(x_min, x_max, y_min, y_max)
                except:
                    value = np.nan

                self.rectangles.append([x_min, x_max, y_min, y_max, value])
                self.rectangle = None
                plt.draw()

                self.update_rectangles_counter()

    def intervalAvgValue(self, xmin, xmax, ymin, ymax) -> float:
        """ 
        Return average value based on intervals.
        Only for 2D.
        """

        filtered_indices = np.where((self.subset_axes[0] >= xmin) & (self.subset_axes[0] <= xmax) & (self.subset_axes[1] >= ymin) & (self.subset_axes[1] <= ymax))
        filtered_values = self.subset_values[filtered_indices]
        print('len: ', len(filtered_values))

        average_value = np.mean(filtered_values)

        return average_value

    def on_motion(self, event):
        """
        Show rectangle when moving mouse, starting point is selected and left mouse button is being pressed.
        """        
        if self.start_point is not None and event.button == 1:
            if self.rectangle is not None:
                self.rectangle.remove()
            end_point = (event.xdata, event.ydata)
            x_min = min(self.start_point[0], end_point[0])
            x_max = max(self.start_point[0], end_point[0])
            y_min = min(self.start_point[1], end_point[1])
            y_max = max(self.start_point[1], end_point[1])
            width = abs(self.start_point[0] - end_point[0])
            height = abs(self.start_point[1] - end_point[1])
            self.rectangle = Rectangle((x_min, y_min), width, height, edgecolor='black', facecolor='none')

            # SET RECTANGLE COLOR
            try:
                valuesNorm = Normalize(vmin=self.subset_values.min(), vmax=self.subset_values.max()) # Get norm for color normalization
                
                averageValue = self.intervalAvgValue(x_min, x_max, y_min, y_max) # Calculate average of samples in selected rectangle
                self.rectangle.set_facecolor(plt.cm.plasma(valuesNorm(averageValue)))
            except:
                self.rectangle.set_facecolor(plt.cm.plasma(1))

            self.ax.add_patch(self.rectangle)
            plt.draw()

    def save_rectangles(self, event):
        """
        Save rectangles to an export file.
        """
        if event.button == 1:
            with open(f'{self.outputFilePath}.txt', 'w') as outputFile:
                for rect in self.rectangles:
                    x_min, x_max, y_min, y_max, value = rect
                    outputFile.write(f"[{x_min}, {x_max}, {y_min}, {y_max}, {value}]\n")

    def go_back(self, event):
        """
        Remove last added rectangle.
        """
        if event.button == 1:
            if self.rectangles:
                self.rectangles.pop()
                # last_rectangle = self.rectangles.pop()
                last_patch = self.ax.patches[-1]
                last_patch.remove()
                plt.draw()

                self.update_rectangles_counter()

    def plotBackgroundFunc(self, plotDots = False):
        if hasattr(self, 'axes') and hasattr(self, 'values') and hasattr(self, 'subset_axes'):
        # if (x is not None) and (y is not None) and (z is not None):
            x = self.axes[0]
            y = self.axes[1]
            z = self.values

            x_subset = self.subset_axes[0]
            y_subset = self.subset_axes[1]

            # Define the limits
            xlim = [x.min(), x.max()]
            ylim = [y.min(), y.max()]

            # Reset limits
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)

            # Display Sample Plot
            contour_plot = self.ax.contourf(x, y, z, cmap='plasma', levels=35)
            plt.colorbar(contour_plot, ax=self.ax, label='Syntetická funkce')

            if plotDots:
                self.ax.scatter(x_subset, y_subset, color='black', s=2, label='Vzorkování funkce')

        else:
            print('Error while plotting function due to sampled arrays.')

    def setupBackgroundFunc(self, gridFunctionFilePath, subsetFunctionFilePath):
        # Setup plot
        try:
            gridFunction = np.load(gridFunctionFilePath)
            self.axes = gridFunction['axes']
            self.values = gridFunction['values']

            subsetFunction = np.load(subsetFunctionFilePath)
            self.subset_axes = subsetFunction['axes']
            self.subset_values = subsetFunction['values']
        except FileNotFoundError:
            print('At least one of the provided files does not exist.')

    def run(self):
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

        # Adding buttons
        # 
        # Save rectangles to output file
        ax_save = plt.axes([0.8, 0.9, 0.1, 0.05])
        button_save = plt.Button(ax_save, 'Save to file')
        button_save.on_clicked(self.save_rectangles)

        # Go back
        ax_back = plt.axes([0.1, 0.9, 0.1, 0.05])
        button_back = plt.Button(ax_back, 'Remove last')
        button_back.on_clicked(self.go_back)

         # Rectangles counter
        self.rectangles_text = plt.text(0.45, 0.95, f'Rectangles: {self.rectangles_counter}', transform=self.fig.transFigure)

        plt.gcf().canvas.manager.set_window_title('Manual sampling selection')

        self.setupBackgroundFunc(gridFunctionFilePath=root_directory+'1_GenerateFunction/generated_data/rootSamples_thesis.npz', subsetFunctionFilePath=root_directory+'1_GenerateFunction/generated_data/subsetSamples_thesis.npz')

        self.plotBackgroundFunc(plotDots=True)

        plt.show()

if __name__ == "__main__":
    Rec = Rectangles(xlim=[-8, 8], ylim=[-8, 8])
    Rec.run()