import tensorflow as tf

class Modeler (tf.keras.Model):
    def __init__(self, x_train, y_train, x_test, y_test):
        super(Modeler, self).__init__()
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        print("Modeler initialized")
        self.model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, return_sequences=True),
            tf.keras.layers.LSTM(50, return_sequences=True),
            tf.keras.layers.LSTM(50, return_sequences=True),
            tf.keras.layers.LSTM(50),
            tf.keras.layers.Dense(1, activation='sigmoid'),
        ])
        print("Model created")

    def call(self, x_train):
        return self.model(x_train)

    def train(self, batch_size, epochs=1, verbose=1):
        print('Training...')
        callback = tf.keras.callbacks.ModelCheckpoint('./checkpoints/model_checkpoint2',
            save_best_only=True)
        self.model.compile(optimizer='adam', loss='binary_crossentropy', \
            metrics=[tf.keras.metrics.BinaryAccuracy()], run_eagerly=True)
        self.model.fit(self.x_train, self.y_train, epochs=epochs, \
            batch_size=batch_size, verbose=verbose, callbacks=[callback], validation_split=0.1)
        print('Finished training')
    
    def test(self, batch_size, verbose=1):
        print('Testing...')
        self.model.evaluate(self.x_test, self.y_test, batch_size=batch_size, \
            verbose=verbose)
        print('Finished testing')
    
        
