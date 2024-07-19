"""
Script: generate_function.py
Desc: Generating synthetic data based with a specific function dependency.
"""

root_directory = '/Users/vojtechremis/Desktop/bachelorproject/'

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(root_directory+'Inc')
from plot import approximationVizualization
import log
Log = log.log('log_generateFunction.txt')

# Using custom style
plt.style.use(root_directory+'Inc/MatPlotLib_styles/classicChart.mplstyle')

# Generate synthetic function
class SyntheticFunction: # Is able to handle general function: R^N -> R
    phi = None
    noise = None
    sampleDistribution = None
    numberOfRootSamples = None
    axes = np.empty(0)
    values = np.empty(0)

    subset_axes = np.empty(0)
    subset_values = np.empty(0)

    def __init__(self, dimension, rootToLoad = None, samplesToLoad=None) -> None:
        self.dim = dimension

        if samplesToLoad is not None and rootToLoad is not None:
            try:
                loadedRoot = self.loadFromFile(rootToLoad)
                loadedSamples = self.loadFromFile(samplesToLoad)
                if len(loadedRoot) == self.dim:
                    self.axes = loadedRoot[0]
                    self.values = loadedRoot[1]
                
                    self.subset_axes = loadedSamples[0]
                    self.subset_values = loadedSamples[1]
    
                    Log.info('Samples file has been loaded.')
                else:
                    Log.error('Loaded function and set dimension does not correspond!')
            except ValueError as e:
                Log.error('Error while loading samples. [Error] = ', e)


    def setFunction(self, phi) -> None:
        if phi.__code__.co_argcount == self.dim:
            self.phi = phi
        else:
            Log.error('Provided function formula does not correspond to dimension.')

    def setNoise(self, noise) -> None:
        if self.dim > 1:
            self.noise = noise
        else:
            Log.error('Problem when setting Noise function.')

    def setSampleDistribution(self, distribution) -> None:
        self.sampleDistribution = distribution

    def generate(self, numberOfRootSamples, dimension, interval=None, noise=False): # interval = ([-I, I], [-I, I], ...)
        self.dim = dimension
        if len(interval) == self.dim:
            sapmled_axes = []
            for single_interval in interval:
                sapmled_axes.append(np.linspace(single_interval[0], single_interval[1], numberOfRootSamples))

            axes = np.meshgrid(*sapmled_axes)
            self.axes = axes
            self.numberOfRootSamples = numberOfRootSamples

            sampledValues = self.phi(*axes)
            
            if noise:
                sampledValues += self.noise(*axes)

            self.values = np.array(sampledValues)
            self.plot()
        else:
            Log.error('Provided intervals does not correspond to dimension.')

    def plot(self, X=np.empty(0), Y=np.empty(0), Z=np.empty(0), plotDots = False, saveFig_fileName=None):
        if (X.size + Y.size + Z.size) == 0:
            X, Y = self.axes
            Z = self.values
        
        if self.dim == 2:
            if (X.size + Y.size + Z.size) != 0:

                # Display 3D plot
                fig = plt.figure(figsize=(15, 6))
                ax1 = fig.add_subplot(121, projection='3d')
                surf = ax1.plot_surface(*self.axes, self.values, cmap='plasma')
                cset = ax1.contour(*self.axes, self.values, zdir='z', offset=np.min(Z), cmap='coolwarm')
                if plotDots:
                    ax1.scatter(X, Y, Z, marker='.', s=10, c="black", alpha=0.5)


                ax1.set_xlabel('x')
                ax1.set_ylabel('y')
                ax1.set_zlabel('z')

                ax1.set_title('Graf funkce v trojrozměrném prostoru')
                fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)

                # Display Sample Plot
                ax2 = fig.add_subplot(122)
                contour_plot = ax2.contourf(*self.axes, self.values, cmap='plasma', levels=40)
                plt.colorbar(contour_plot, ax=ax2)

                ax2.scatter(X, Y, color='black', s=3, label='Vzorkování funkce')

                ax2.set_xlabel('x')
                ax2.set_ylabel('y')
                ax2.set_title('Vzorkování funkce')
                ax2.legend()

                fig.tight_layout()

                if saveFig_fileName is not None:
                    plt.savefig(saveFig_fileName+'.pdf')
                
                plt.show()

            else:
                Log.error('Error while plotting function due to sampled arrays.')
        else:
            Log.warning('Other than 2D plot not implemented or even impossible.')

    def subset(self, numberOfSamples, distribution='uniform'):
        numberOfRootPoints = self.axes[0].size
        rootSampleShape = self.axes[0].shape

        if distribution == 'uniform':
            random_indices = np.random.choice(np.arange(numberOfRootPoints), size=numberOfSamples)
        elif distribution == 'custom':
            if self.sampleDistribution:
                normalizedSamples = self.sampleDistribution(numberOfSamples) 
                random_indices = np.floor(normalizedSamples*numberOfRootPoints).astype(int)
            else:
                Log.error("Custom distribution function hasn't been set!")

        else:
            Log.error('Sample distribution function ['+distribution+'] is not recognized.')
            return None

        random_indices_2d = np.unravel_index(random_indices, rootSampleShape)

        selectedAxes = []
        
        for selectedAxis in self.axes:
            selectedAxes.append(selectedAxis[random_indices_2d])

        selectedValues = self.values[random_indices_2d]

        self.subset_axes = selectedAxes
        self.subset_values = selectedValues

        X, Y = selectedAxes
        Z = selectedValues
        self.plot(X, Y, Z, plotDots=True)
    
        return selectedAxes, selectedValues
    

    def specialVizualization(self, samples=30, manualSamples=None, showGrid=True):
        """
        Reusing ApproximationVisualization method in /inc.
        This method only initialize input parameters as follows:
        xlim, ylim due to root sampled function attribute
        """

        xlim = [self.axes[0].min(), self.axes[0].max()]
        ylim = [self.axes[1].min(), self.axes[1].max()]
        normMinMaxColor = self.values.min(), self.values.max()

        Log.info(f'Values important for comparison of approximated values in 3_Train section: normMinMaxColor = [{normMinMaxColor[0]}, {normMinMaxColor[1]}].')

        approximationVizualization(
            xlim=xlim,
            ylim=ylim,
            predict_func=self.intervalAvgValue,
            indicatorNames=['Indicator1', 'Indicator2'],
            samples=samples,
            manualSamples=manualSamples,
            showGrid=showGrid,
            normMinMaxColor=normMinMaxColor,
            Title=None
        )

    def intervalAvgValue(self, vectors: list) -> float:
        """ 
        Return average value based on intervals. In handy when adding colors
        Only for 2D vectors.
        """
        vectors = np.array(vectors)
        vectors.shape = (-1, vectors.shape[0])

        if vectors.shape[1] != 4:
            Log.error(f'Wrong dimensional vector provided to intervalAvgValue() function. Provided dimension [{len(vectors)}].')
            return None
        else:
            return_value = []
            for vector in vectors:
                xmin, xmax, ymin, ymax = vector

                filtered_indices = np.where((self.axes[0] >= xmin) & (self.axes[0] <= xmax) & (self.axes[1] >= ymin) & (self.axes[1] <= ymax))
                filtered_values = self.values[filtered_indices]

                average_value = np.mean(filtered_values)
                return_value.append(average_value)

            return return_value


    def toFile(self, subsetAxes=None, subsetValues=None, fileName='rootSamples'):
        """
        Saves generated function to a file (specifically dense sampled grid and subset).
        """
        if (len(subsetAxes) + len(subsetValues)) == 0:
            subsetAxes = self.axes
            subsetValues = self.values

        try:
            np.savez(
            'generated_data/' + fileName + '.npz',
            axes=subsetAxes,
            values=subsetValues
            )
            Log.info('Root function has been saved.')
        except:
            Log.error('Error when saving root file.')
        

    def loadFromFile(self, fileToLoad=''):
        data = np.load(fileToLoad)
        Axes = data['axes']
        Values = data['values']
        return Axes, Values
    
    def save(self, name='_name'):
        Log.setLogLevel(1)

        self.toFile(subsetAxes=self.axes, subsetValues=self.values, fileName='rootSamples'+name)
        self.toFile(subsetAxes=self.subset_axes, subsetValues=self.subset_values, fileName='subsetSamples'+name)
        
        Log.resetLogLevel()
        Log.info('Function has been saved!')



if __name__ == '__main__':
    Function = SyntheticFunction(dimension=2)

    def f(x, y):
        return x * y * np.exp(-x**2 - y**2)


    def noise(x, y, noise_level=0.05):
        return noise_level * np.random.randn(*x.shape) 

    Function.setFunction(f)
    Function.setNoise(noise)