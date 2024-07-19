"""
Script: train_neural.py
Desc: This class provides complete control over the setup and training of neural network experiments in Thesis.
"""

root_directory = '/Users/vojtechremis/Desktop/bachelorproject/'

# Neural training
import tensorflow as tf
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow import keras

# Data
import numpy as np
import pandas as pd
import joblib
import json

# Plots
import matplotlib.pyplot as plt
plt.style.use(root_directory+'Inc/MatPlotLib_styles/classicChart.mplstyle')

# Other
import os
import datetime
import re

# Custom class
import sys
sys.path.append(root_directory+'Inc')
import log

sys.path.append(root_directory+'3_Train/inc')
import callbacks as Callbacks

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

# Identity function definition
def identity_func(X):
    return X

# Class Body
class NeuralNetwork:
    def __init__(self, inputDim, outputDim, logRoot='log'):

        self.Log = log.log(logRoot + '/training_log')

        self.logRoot = logRoot
        np.random.seed(42)  # Very Important

        self.scalerObject_X = None
        self.scalerObject_Y = None
        self.model = None

        self.callbacks_list = []
        self.customCallbacks_list = []
        self.customMetrics_names = []

        self.outputDim = outputDim
        self.inputDim = inputDim

        self.target_name = 'relative_profit'

    def createDirectory(self, pathToDirectory):
        try:
            if not os.path.exists(pathToDirectory):
                self.Log.info(f'Folder has been created [{pathToDirectory}].')
                os.makedirs(pathToDirectory)
        except Exception as e:
            self.Log.error(f'Error when creating folders. Error: {e}')

    def loadData(self, pathToCombinations):
        # Load source data
        Xdata_pd = pd.read_csv(pathToCombinations)

        self.numberOfDatapoints = len(Xdata_pd)
        self.indicators = list(Xdata_pd.columns)[0:-1] # Every indicator except last relative_profit
        self.indicatorNames = self.indicatorNames_()

        if self.inputDim != len(self.indicators): # min and max for each indicator
            self.Log.error('Indicator dimension does not correspond to provided inputDim value!')
            return None

        self.X = Xdata_pd.drop(columns=[self.target_name]).values
        self.Y = Xdata_pd[self.target_name].values

        self.dataScaled = False

    def indicatorNames_(self):
        cleaned_indicators_dict = {}
        for indicator in self.indicators:
            cleaned_indicator = indicator.replace('_min', '').replace('_max', '')
            cleaned_indicators_dict[cleaned_indicator] = None

        return list(cleaned_indicators_dict.keys())


    def setHyperParameters(self, args=None):
        # Setup
        if args is None:
            self.args = {
                'scaler': 'standardscaler',
                'test_size': 0.3,
                'epochs': 10,
                'batch_size': 50,
                'dense_units': [100, 100],
                'learning_rate': 1e-2,
                'learning_rate_final': 1e-4
            }
        else:
            self.args = args

        print_scaled = True

        # Creating directories
        self.logdir = os.path.join(self.logRoot, "{}-{}-{}".format(
            os.path.basename(globals().get("__file__", "notebook")),
            datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
            ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(self.args.items())))
        ))

        self.plotsDir = f'{self.logdir}/Plots'
        self.datasetsDir = f'{self.logdir}/Datasets'
        self.kerasDir = f'{self.logdir}/Keras'
        self.tensorboardDir = f'{self.logdir}/TensorboardLog'

        self.createDirectory(self.plotsDir)
        self.createDirectory(self.datasetsDir)
        self.createDirectory(self.kerasDir)
        self.createDirectory(self.tensorboardDir)



        # Set features scaling (defined in args['scaler'])
        self.setScaler()
    
    def setScaler(self, scaler=None):
        if scaler is None:
            scaler = self.args['scaler']
        
        # We won't scale Y
        self.scalerObject_Y = preprocessing.FunctionTransformer(func=identity_func) # preprocessing.MinMaxScaler()

        if scaler == 'minmax':
            self.scalerObject_X = preprocessing.MinMaxScaler()
        elif scaler == 'standardscaler':
            self.scalerObject_X = preprocessing.StandardScaler()
        else:
            self.scalerObject_X = preprocessing.FunctionTransformer(func=identity_func)

    def performScaling(self):
        self.X = self.scalerObject_X.fit_transform(self.X)
        self.Y = self.scalerObject_Y.fit_transform(self.Y.reshape(-1, 1))

        self.dataScaled = True

    def splitDataToTrainTest(self):
        # Splitting data into Train / Test
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=self.args['test_size'], random_state=42)

    def plotSampling(self):
        pass

    def defineArchitecture(self):
        layer_units = self.args['dense_units']

        if len(self.args['dense_units']) != 0:

            # Setting input layers
            input_layer = tf.keras.Input(shape=(self.inputDim,))
            d = input_layer
            
            # Setting hidden layers
            for units in layer_units:
                d = tf.keras.layers.Dense(
                    units=units,
                    activation='relu',
                    kernel_initializer=tf.keras.initializers.HeNormal(),
                    bias_initializer=tf.keras.initializers.Constant(0)
                )(d)
            
            # Output layer
            output_layer = tf.keras.layers.Dense(
                units=1,
                activation='linear',
                kernel_initializer=tf.keras.initializers.HeNormal(),
                bias_initializer=tf.keras.initializers.Constant(0)
            )(d)
            
            # Define model
            model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
            
            return model
        else:
            return None
    
    def defineDecay(self, num_of_backprops):
        """ 
        Equals to learning_rate if no decay.
        """
        return tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate = self.args['learning_rate'],
            decay_steps = num_of_backprops,
            alpha=self.args['learning_rate_final']
        )

    def defineLoss(self):
        return tf.keras.losses.MeanSquaredError()
    
    def setCustomMetric_callback(self, X_evaluate, Y_evaluate, metricName):
        """
        Setting custom metric which will be evaluate after every batch end.
        """ 
        
        if metricName not in self.customMetrics_names:
            Callback = Callbacks.Metrics_callback(log_dir=self.tensorboardDir, X_eval=X_evaluate.reshape(-1, self.inputDim), Y_eval=Y_evaluate.reshape(-1, self.outputDim), metric_name=metricName, scalerObject_X=self.scalerObject_X, scalerObject_Y=self.scalerObject_Y)
            self.customCallbacks_list.append(Callback)
            self.customMetrics_names.append(metricName)
        else:
            self.Log.warning(f'Metrics [{metricName}] already exists. Skipping it.')
    
    def setDefaultCallbacks(self):
        """
        Setting up predefined callbacks:
        - learning rate
        - earlyStopping
        """

        file_writer = tf.summary.create_file_writer(self.logdir)
        file_writer.set_as_default()

        tb_callback = tf.keras.callbacks.TensorBoard(log_dir=self.logdir, histogram_freq = 1)
        self.callbacks_list.append(tb_callback)

        # Learning rate callback
        self.callbacks_list.append(Callbacks.LearningRate_callback(self.tensorboardDir))

        # Early stopping callback
        if self.args['EarlyStopping_patience'] is not None:
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                min_delta=self.args['EarlyStopping_minDelta'],
                patience=self.args['EarlyStopping_patience']
            )
            self.callbacks_list.append(early_stopping)


    def trainModel(self):
        # Scale features
        self.performScaling()

        print(self.X.shape, self.Y.shape)

        # Split data to Test / Train
        self.splitDataToTrainTest()

        model = self.defineArchitecture()

        num_of_backprops = self.X_train.shape[0] / self.args['batch_size'] * self.args['epochs']
        decayObject = self.defineDecay(num_of_backprops)
        lossObject = self.defineLoss()

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=decayObject
            ),
            loss=lossObject
        )

        model.summary()

        self.setDefaultCallbacks()

        # sys.exit(0)

        fitted_model = model.fit(
            self.X_train,
            self.Y_train,
            validation_data=(self.X_test, self.Y_test),
            callbacks = self.callbacks_list + self.customCallbacks_list,
            epochs=self.args['epochs'],
            batch_size=self.args['batch_size']
        )

        self.model = model
        self.fitted_model = fitted_model
        self.loss = self.fitted_model.history['loss']
        self.val_loss = self.fitted_model.history['val_loss']

        self.custom_metrics = {}
        for metricHistory_dict in [metric_history.get_history() for metric_history in self.customCallbacks_list]:
            self.custom_metrics = {**self.custom_metrics, **metricHistory_dict}

        self.plotLossFunction()
        self.plotPrediction()

        # Save experiment
        self.saveExperiment()

    def loadExperiment(self, logdir):
        # Define Paths
        self.logdir = logdir
        self.plotsDir = f'{self.logdir}/Plots'
        self.datasetsDir = f'{self.logdir}/Datasets'
        self.kerasDir = f'{self.logdir}/Keras'

        # Load Val and Train loss for plotting
        epochs, self.loss, self.val_loss = np.load(f'{self.datasetsDir}/Loss_ValTrain.npy')

        # Load custom metrics
        with open(f'{self.datasetsDir}/CustomMetrics.json', 'r') as customMetrics_file:
            custom_metrics = json.load(customMetrics_file)
            self.custom_metrics  = custom_metrics
            self.customMetrics_names = custom_metrics.keys()

        # Load model

        # Load scalers
        self.scalerObject_X = joblib.load(f'{self.datasetsDir}/scalerObject_X.joblib')
        self.scalerObject_Y = joblib.load(f'{self.datasetsDir}/scalerObject_Y.joblib')
 
        # Load X_train/test, Y_train/test
        self.X_train = np.load(f'{self.datasetsDir}/X_train.npy')
        self.X_test = np.load(f'{self.datasetsDir}/X_test.npy')
        self.Y_train = np.load(f'{self.datasetsDir}/Y_train.npy')
        self.Y_test = np.load(f'{self.datasetsDir}/Y_test.npy')

        # Load model
        self.model = keras.models.load_model(f'{self.kerasDir}/model.h5')

        # Load metadata
        with open(f'{self.logdir}/metadata.json', 'r') as metadataFile:
            metadata = json.load(metadataFile)
        
            self.indicatorNames = metadata['indicatorNames']
         

    def saveExperiment(self):
        # Save loss
        loss = self.fitted_model.history['loss']
        val_loss = self.fitted_model.history['val_loss']
        epochs = range(1, (len(loss))+1)
        np.save(f'{self.datasetsDir}/Loss_ValTrain.npy', [epochs, loss, val_loss])

        # Save custom metrics
        with open(f'{self.datasetsDir}/CustomMetrics.json', 'w') as customMetrics_file:
            json.dump(self.custom_metrics, customMetrics_file)

        # Save scaler
        joblib.dump(self.scalerObject_X, f'{self.datasetsDir}/scalerObject_X.joblib')
        joblib.dump(self.scalerObject_Y, f'{self.datasetsDir}/scalerObject_Y.joblib')

        # Save X_train/test, Y_train/test
        np.save(f'{self.datasetsDir}/X_train.npy', self.X_train)
        np.save(f'{self.datasetsDir}/X_test.npy', self.X_test)
        np.save(f'{self.datasetsDir}/Y_train.npy', self.Y_train)
        np.save(f'{self.datasetsDir}/Y_test.npy', self.Y_test)

        # Save model
        self.model.save(f'{self.kerasDir}/model.h5')

        # Save meta data 
        metadata = {
            'indicatorNames': self.indicatorNames
        }

        with open(f'{self.logdir}/metadata.json', 'w') as metadataFile:
            json.dump(metadata, metadataFile)
        
    def predict(self, X, inverseScale_Y=True):
        """ 
        This method takes vector X, scales it, predicts and return inverse scaled output Y (optional).
        X can be np.array / list / variable
        """

        # Checking X type
        if isinstance(X, list):
            X = np.array(X)
        elif isinstance(X, (int, float)):
            X = np.array([X])
        else:
            pass

        if (self.model is not None) and (self.scalerObject_X is not None) and (self.scalerObject_Y is not None):
            X_scaled = self.scalerObject_X.transform(X.reshape(-1, self.inputDim))
            Y_scaled = self.model.predict(X_scaled)

            if inverseScale_Y:
                return self.scalerObject_Y.inverse_transform(Y_scaled).reshape(-1, )
            else:
                return Y_scaled.reshape(-1, )
        else:
            self.Log.error('Neither the Model or scaler Object_X or scaler Object_Y is available!')

    # # # # # # # # # # # # # # #
    # Plots
    # # # # # # # # # # # # # # #
            
    def plotLossFunction(self):
        # Get loss values
        epochs = range(1, (len(self.loss))+1)

        fig, ax = plt.subplots(1, 1, figsize=(8, 8), constrained_layout=True)

        # Loss function
        ax.plot(epochs, self.loss, label='Trénovací ztráta', color='royalblue', linewidth=1, linestyle='solid')
        ax.plot(epochs, self.val_loss, label='Validační ztráta', color='lightcoral', linewidth=1, linestyle='solid')
        ax.set_xlabel('Epochy')
        ax.set_ylabel('Ztráta')
        ax.legend(loc='upper right')

        fig.set_constrained_layout_pads(w_pad=4/72, h_pad=4/72, hspace=0.2, wspace=0.2)

        plt.savefig(f'{self.plotsDir}/losses_epochs_loss_val_loss.pdf')

        plt.show()

    def plotPrediction(self, print_scaled=False, test=True, train=True, n_samples=100):
        # Vectors x_train, x_test, y_train, y_test are scaled!

        # Print X_test
        if test:
            fig, ax = plt.subplots(figsize=(9,9))  

            # Printing only subset of X_test (in case of huge datasets)
            test_indicies = np.random.choice(self.X_test.shape[0], size=n_samples, replace=False)
            X_test_subset = self.X_test[test_indicies, :]
            Y_test_subset = self.Y_test[test_indicies]

            # X_test_subset_unscaled = self.scalerObject_X.inverse_transform(X_test_subset.reshape(-1, 1))
            
            Y_test_subset_predicted_unscaled = self.scalerObject_Y.inverse_transform(
                np.array(
                    self.model.predict(X_test_subset)
                ).reshape(-1, self.outputDim)
            )
            
            Y_test_subset_unscaled = self.scalerObject_Y.inverse_transform(Y_test_subset)

            # Plot
            ax.plot(Y_test_subset_unscaled, Y_test_subset_predicted_unscaled, 'o', markersize=1, color='#800000')
            ax.set_title('Porovnání predikovaných a zlatých dat (testovací dataset)')
            ax.set_xlabel("Skutečné Y", fontsize=15)
            ax.set_ylabel("Predikované Y", fontsize=15)
            ax.tick_params(axis='x', labelsize=14)
            ax.tick_params(axis='y', labelsize=14)

            plt.savefig(f'{self.plotsDir}/real_vs_predicted_samples_{n_samples}_test.pdf')
            plt.show()
            plt.clf()
                

        if train:
            fig, ax = plt.subplots(figsize=(9,9))  
            
            # Printing only subset of X_rain (in case of huge datasets)
            train_indicies = np.random.choice(self.X_train.shape[0], size=n_samples, replace=False)
            X_train_subset = self.X_train[train_indicies, :]
            Y_train_subset = self.Y_train[train_indicies]

            # X_test_subset_unscaled = self.scalerObject_X.inverse_transform(X_test_subset.reshape(-1, 1))
            
            Y_train_subset_predicted_unscaled = self.scalerObject_Y.inverse_transform(
                np.array(
                    self.model.predict(X_train_subset)
                ).reshape(-1, self.outputDim)
            )
            
            Y_train_subset_unscaled = self.scalerObject_Y.inverse_transform(Y_train_subset)

            # Plot
            ax.plot(Y_train_subset_unscaled, Y_train_subset_predicted_unscaled, 'o', markersize=1, color='#3090C7')
            ax.set_title('Porovnání predikovaných a zlatých dat (trénovací dataset)')
            ax.set_xlabel("Skutečné Y", fontsize=15)
            ax.set_ylabel("Predikované Y", fontsize=15)
            ax.tick_params(axis='x', labelsize=14)
            ax.tick_params(axis='y', labelsize=14)

            plt.savefig(f'{self.plotsDir}/real_vs_predicted_samples_{n_samples}_train.pdf')
            plt.show()

    # Predict class for relative prediction
    class relativePredict:
        def __init__(self, predict_func, X_toCompare, Y_toCompare) -> None:
            self.predict_func = predict_func

            self.X_toCompare = X_toCompare
            self.Y_toCompare = Y_toCompare

        def valueToCompare(self, X_value):
            if isinstance(X_value, list):
                x = np.array(X_value)
                
            index = np.where(np.all(self.X_toCompare == x, axis=1))
            return self.Y_toCompare[index][0]

        def predict(self, X, inverseScale_Y=True):
            predictedValue = self.predict_func(X)
            valueToReturn = abs(self.valueToCompare(X) - predictedValue[0]) / abs(self.valueToCompare(X))
            return [valueToReturn]