"""
Script: pca.py
Desc: Perform Principal Component Analysis (4_PCA) on a dataset using Custom 4_PCA Library (from Inc/pca.py). Save all essential data to enable transformation of the entire dataset at a later time.
For first, it has to remove all non-numeric columns, all columns with all values equal to zero and all Null values.
"""

root_directory = '/Users/vojtechremis/Desktop/bachelorproject/'

import sys
sys.path += [root_directory+'Inc']
import numpy as np
from matplotlib import pyplot as plt

from Inc import pca as pca_
import log
import database as db
Log = log.log()

plt.style.use(root_directory+'Inc/MatPlotLib_styles/classicChart.mplstyle')

dataset_name = 'dataset_clean_B10' # Dataset name will be used for loading data and naming stored files

# Loading dataset
path_to_database =f'.../0_Preprocessing/_output/{dataset_name}.db'
query = 'SELECT * FROM table_name'

Log.info('Loading database...')
connectionToSourceDB = db.Connection(path_to_database)
researchwizzDB = connectionToSourceDB.tableToDF(query)
if researchwizzDB is None:
    sys.exit(0)


# Selecting only valid columns
columns_all = researchwizzDB.columns.tolist()

columns_nonNumeric = ['Company', 'Ticker', 'NERD', 'export_date'] # Non-numeric columns

db_summed = researchwizzDB.sum()
const_zero_columns = db_summed[ db_summed == 0].index.tolist() # Columns with all values = 0

for column in columns_nonNumeric+const_zero_columns:
    columns_all.remove(column)
columns_forPCA = columns_all

dataset = researchwizzDB[columns_forPCA]


# Removing None values
dataset_size_before = len(dataset)
dataset.dropna(how='any', inplace=True)

dataset_size_after = len(dataset)
Log.info(f'Dataset size before cleaning: {dataset_size_before} vs. after cleaning: {dataset_size_after}.')


# Performing Principal Component Analysis
Log.info('Calculating Principal Component Analysis.')

path_to_output = f'.../0_Preprocessing/4_PCA/{dataset_name}/'

# Getting X, y
target_column = 'relative_profit'
columns_forPCA.remove(target_column)
X = dataset[columns_forPCA].to_numpy()
y = dataset[target_column].to_numpy()

# Selecting two indicators which will be visible on visualizations
feature1 = columns_forPCA[1]
feature2 = columns_forPCA[2]
indx1 = 1
indx2 = 2

# Number of principal components to keep.
N = 2

# Run 4_PCA
pca_instance = pca_.PCA()
pca_instance.fit(X)
pca_instance.setN(N)

pca_instance.plotFitResult(cumulative = True, saveFig_path = path_to_output + 'explainedVariance') # Save explained variance plot

X_pcaSpace = pca_instance.transform(X)

eigen_vectors = pca_instance.getPrincipalEigenvectors(scaled=True).T
Log.info(f'Eigenvectors: \n{eigen_vectors}')


# Limit sizes to 1000 samples for visualizations
random_indices = np.random.choice(len(X), 1000, replace=False)
X = X[random_indices]
X_pcaSpace = X_pcaSpace[random_indices]


# PLOTS

plt.clf()
plt.figure(figsize=(8, 8))
plt.scatter(X[:, indx1], X[:, indx2], label='Projekce dat do 2D', color='lightseagreen', alpha=0.7, edgecolors='black')
plt.title('Projekce dat do prostoru 2D')
plt.xlabel(feature1)
plt.ylabel(feature2)
plt.grid(True)

origin = np.mean(X[:, indx1:indx2+1], axis=0)
colors = ['orange', 'darkseagreen']
alphas = [1, 0.7]
for i in range(eigen_vectors.shape[0]):
    plt.quiver(*origin, eigen_vectors[i, 0], eigen_vectors[i, 1], color=colors[i], scale=3, label=f'Vlastní vektor {i+1}', alpha=alphas[i])
    
plt.legend()
plt.savefig(path_to_output + 'original_space.svg')
plt.show()

    
plt.clf()
plt.figure(figsize=(8, 8))
plt.scatter(X_pcaSpace[:, 0], X_pcaSpace[:, 1], label='Data v prostoru 4_PCA', color='lightseagreen', alpha=0.7, edgecolors='black')
plt.title('Transformace dat do prostoru hlavních komponent')
plt.xlabel('Hlavní komponenta 1')
plt.ylabel('Hlavní komponenta 2')
plt.legend()

plt.grid(True)
plt.savefig(path_to_output + 'PrincipalSpace_final.svg')
plt.show()

pca_instance.save(path_to_output + 'Output')