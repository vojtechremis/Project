"""
Script: pca.py
Desc: custom Principal Component Analysis (4_PCA) library.
Described in thesis.
"""

root_directory = '/Users/vojtechremis/Desktop/bachelorproject/'

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

plt.style.use(root_directory+'Inc/MatPlotLib_styles/classicChart.mplstyle')

class PCA:
    """
    Implementation of principal component analysis method.
    """
    eigenvalues = None
    eigenvectors = None
    topN = None
    originalDim = None
    P = None # Transformation matrix from standard basis to eigen vector basis

    scaler_mean = None
    scaler_std = None

    def __init__(self, sourceDir : str = None):
        """
        If sourceDir is not None, method 'init' calls 'load' method.

        param: str sourceDir : location to the directory where data are saved.If sourceDir is None, no data will be loaded.
        return: None
        """
        if sourceDir is not None:
            self.load(sourceDir)

    def standardScalerFit(self, X : np.array):
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)

        self.scaler_mean = mean
        self.scaler_std = std

        return np.multiply(X - mean, 1/std)

    def standardScalerTransform(self, X : np.array):
        if self.scaler_mean is not None and self.scaler_std is not None:
            return np.multiply(X - self.scaler_mean, 1/self.scaler_std)
        else:
            print('Error while requesting obligatory parameters for standardScaling inversion!')
            return None

    def standardScalerInverseTransform(self, X_pcSpace : np.array):
        if self.scaler_mean is not None and self.scaler_std is not None:
            X_ = np.multiply(X_pcSpace, self.scaler_std) + self.scaler_mean
            return X_
        else:
            print('Error while requesting obligatory parameters for standardScaling inversion!')
            return None
    def missingNumbers(self, X : np.array) -> bool:
        return (X == None).any() or np.isnan(X).any()
    def fit(self, X_ : np.array):
        """
        Method 'fit' calculates eigenvalues and eigenvectors of covariance matrix, sorts them by eigenvalues, plots them and store them (each eigenvector is normalized).

        param: np.array X : matrix of data points (columns as features, rows as datapoints)
        return: None
        """

        self.originalDim = X_.shape[1]

        if self.missingNumbers(X_) == True:
            print('Dataset has missing values.')
            return None

        X = self.standardScalerFit(X_)


        # m = number of samples
        m = X.shape[0]
        factor = 1 / (m - 1)

        # Mean is allready 0
        mean = np.mean(X, axis=0)

        S = factor * (X - mean).T @ (X - mean)

        # Calculate eigen values and vectors
        try:
            eigenvalues, eigenvectors = np.linalg.eig(S)
            eigenvectors_norm = eigenvectors / np.linalg.norm(eigenvectors, axis=0)

        except Exception as e:
            print(f'While computing eigen values/vectors error occurred : {e}.')
            return None

        indcs = np.argsort(-eigenvalues)

        eigenvalues_sorted = eigenvalues.take(indcs)
        eigenvectors_sorted = eigenvectors_norm.take(indcs, axis=1)

        # Force first coordinate to be positive
        for i in range(eigenvectors_sorted.shape[1]):
            if eigenvectors_sorted[0, i] < 0:
                eigenvectors_sorted[:, i] = eigenvectors_sorted[:, i] * (-1)

        # Store eigenvalues and eigenvectors
        self.eigenvalues, self.eigenvectors = np.array(eigenvalues_sorted), np.array(eigenvectors_sorted)
        self.plotFitResult(cumulative=True)

        # Set transformation matrix P
        self.P = np.array(self.eigenvectors)
    def plotFitResult(self, cumulative : bool = False, saveFig_path : str = None):
        """
        Method 'plotFitResult' plots eigenvalues (cumulative sum if needed).

        param: bool cumulative : matrix of data points
               str saveFig_path : location of plot to file (name included). Plot won't be saved if saveFig_path is None.
        return: None
        """

        eigenvalues = self.eigenvalues
        if eigenvalues is not None:
            x_axis = ['$\lambda_'+str(i+1)+'$' for i in range(len(eigenvalues))]

            if cumulative:
                maxVariance = np.sum(eigenvalues)
                ratio = eigenvalues / maxVariance
                cumulative_ratio = np.cumsum(ratio)
                plt.plot(
                    x_axis,
                    cumulative_ratio
                )
                plt.title('Eigenvalues (cumulative)')
                plt.xlabel('$\lambda_i$')
                plt.ylabel('$\lambda_i$ / $\sum_{i} \lambda_i$')

                plt.ylim([0, 1])

                plt.xticks(range(len(x_axis)), x_axis, size='small')

                plt.grid(True)

                plt.show()
            else:
                plt.scatter(
                    x=range(len(x_axis)),
                    y=eigenvalues,
                    s=2
                )
                plt.title('Eigenvalues')
                plt.xlabel('$\lambda_i$')
                plt.ylabel('$\lambda_i$ value')

                plt.xticks(range(len(x_axis)), x_axis, size='small')

                plt.show()

            if saveFig_path is not None:
                plt.savefig(f'{saveFig_path}.svg')
        else:
            print('There are no eigenvalues to plot!')

    def setN(self, topN : int):
        """
        Method 'setN' allows you to choose first most important principal components.

        param: int topN : first N most important principal components
        return: None
        """

        if topN in range(1, len(self.eigenvectors)+1):
            self.topN = topN

        else:
            print('N is not in allowed range.')

    def transform(self, X_ : np.array) -> np.array:
        """
        Method 'transform' transforms dataset matrix into a basis of principal components.

        param: np.array X_ : dataset matrix to be transformed (columns as features, rows as datapoints)
        return: dataset matrix in basis of principal components
        """

        X = self.standardScalerTransform(X_)

        if self.P is not None:

            if self.topN is not None:
                topN = self.topN
            else:
                print('The dataset is transformed according to all components because the parameter TopN, which selects only the principal components, was not set.')
                topN = self.originalDim

            Transformed = X @ self.P[:, :topN]

            return Transformed

        else:
            print('No transformation matrix set.')
            return None

    def inverse_transform(self, Y : np.array, standardInverseTransform : bool = True):
        """
        Method 'inverse_transform' transforms dataset matrix from basis of principal components to its origin basis.

        param: np.array X_ : dataset matrix to be transformed
        return: dataset matrix in its origin basis
        """

        if self.P is not None:

            if self.topN is not None:
                topN = self.topN
            else:
                print('The dataset is transformed according to all components because the parameter TopN, which selects only the principal components, was not set.')
                topN = self.originalDim

            Transformed = Y @ self.P[:, :topN].T

            if standardInverseTransform:
                return self.standardScalerInverseTransform(Transformed)
            else:
                return Transformed
        else:
            print('No transformation matrix set.')
            return None

    def getPrincipalEigenvectors(self, scaled : bool):
        """
        Method 'getPrincipalEigenvectors' returns eigenvectors corresponding to selected principal components.

        param: bool scaled : if true the eigenvectors length is scaled correspondingly to its eigenvalue value (max vector size is 1)
        return: eigenvectors corresponding to selected principal components
        """

        if self.eigenvectors is not None:

            # Crop eigenvectors, if topN was set
            if self.topN is not None:
                eigenvectors_topN = self.eigenvectors[:self.topN, :self.topN]
                eigenvalues_topN = self.eigenvalues[:self.topN]
            else:
                eigenvectors_topN = self.eigenvectors
                eigenvalues_topN = self.eigenvalues

            if scaled is True:
                eigenvalues_sum = np.sum(eigenvalues_topN)

                return np.multiply((eigenvalues_topN.reshape(1, -1) / eigenvalues_sum), eigenvectors_topN)
            else:
                return eigenvectors_topN
        else:
            print('4_PCA analysis haven\'t been performed yet. Use method fit, to initialize the method.')
    

    def save(self, outputDir : str):
        """
        Method 'save' saves the eigenvalues and eigenvectors.

        param: str outputDir : location to the directory where data shall be saved
        return: None
        """

        try:
            if not os.path.exists(outputDir):
                os.makedirs(outputDir)

            # Save data
            data = {
                'eigenvalues': self.eigenvalues,
                'eigenvectors': self.eigenvectors,
                'topN': self.topN,
                'originalDim': self.originalDim,
                'scaler_mean': self.scaler_mean,
                'scaler_std': self.scaler_std
            }

            with open(outputDir + '/pca_coreData.pkl', 'wb') as file:
                pickle.dump(data, file)

            print(f'4_PCA basic data have been saved to \'{outputDir}\'.')

        except Exception as e:
            print(f'Error when saving 4_PCA basic data. Error: {e}')

    def load(self, sourceDir : str):
        """
        Method 'load' loads the eigenvalues, eigenvectors and performs fit except computation.

        param: str sourceDir : location to the directory where data are saved. Eigenvectors are supposed to be normalized and sorted by eigenvalues.
        return: None
        """

        try:
            with open(sourceDir + '/pca_coreData.pkl', 'rb') as file:
                loaded_data = pickle.load(file)

            self.eigenvalues = loaded_data['eigenvalues']
            self.eigenvectors = loaded_data['eigenvectors']
            self.topN = loaded_data['topN']
            self.originalDim = loaded_data['originalDim']
            self.scaler_mean = loaded_data['scaler_mean']
            self.scaler_std = loaded_data['scaler_std']

            self.P = np.array(self.eigenvectors)

            print(f'Eigenvectors and eigenvalues have been loaded from \'{sourceDir}\'.')
        except Exception as e:
            print(f'Error when loading 4_PCA basic data. Error: {e}.')




if __name__ == '__main__':
    print('Jobs done!')





