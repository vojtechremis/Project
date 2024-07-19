import tensorflow as tf

class LearningRate_callback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir):
        super(LearningRate_callback, self).__init__()
        self.log_dir = log_dir

    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate.numpy()
        
        tf.summary.scalar('learning_rate', data=lr, step=self.model.optimizer.iterations)

class Metrics_callback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, X_eval, Y_eval, metric_name, scalerObject_X, scalerObject_Y):
        super(Metrics_callback, self).__init__()
        self.log_dir = log_dir

        # Raw (unscaled) data to evaluate on
        self.X_eval = X_eval
        self.Y_eval = Y_eval

        # Setup scaler objects. Transformation will be performed when the data is requested for the first time, since scaler objects are fitted after method trainModel() is triggered
        self.scalerObject_X = scalerObject_X
        self.scalerObject_Y = scalerObject_Y

        # Initialization of scaled data to evaluate on
        self.X_eval_scaled = None
        self.Y_eval_scaled = None

        self.metric_name = 'CustomMetrics_'+metric_name

        self.History_ = {}
        self.History_[self.metric_name] = []

    def on_epoch_end(self, epoch, logs=None):

        # Transform data if it is not scaled yet
        if self.X_eval_scaled is None and self.Y_eval_scaled is None:

            if (self.scalerObject_X is not None) and (self.scalerObject_Y is not None):
                self.X_eval_scaled = self.scalerObject_X.transform(self.X_eval)
                self.Y_eval_scaled = self.scalerObject_Y.transform(self.Y_eval)
            else:
                print('Neither the scaler Object_X or scaler Object_Y is available!')
                return None


        Y_predicted_scaled = self.model.predict(self.X_eval_scaled, verbose=0)

        # Evaluation MSE on transformed data
        metric_value = tf.keras.losses.mean_squared_error(self.Y_eval_scaled.reshape(1, -1), Y_predicted_scaled.reshape(1, -1))
        self.History_[self.metric_name].append(float(metric_value.numpy()[0]))

        tf.summary.scalar(self.metric_name, data=metric_value.numpy()[0], step=self.model.optimizer.iterations)

    def get_history(self):
        return self.History_
