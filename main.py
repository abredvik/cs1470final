from preprocess import get_data, process
from model import Modeler
import tensorflow as tf

def main():
    data = get_data('./data/data.csv')
    X, Y = process(data)
    batch_size = 32
    # split training/testing data (80/20)
    split = int(0.8*len(Y))
    train_x = X[:split]; train_y = Y[:split]
    test_x = X[split:]; test_y = Y[split:]

    model = Modeler(train_x, train_y, test_x, test_y)
    model.train(batch_size=batch_size, epochs=5)
    model.test(batch_size=batch_size)

    # # Load Model
    # loaded = tf.keras.models.load_model('./checkpoints/model_checkpoint')
    # loaded.compile(optimizer='adam', loss='binary_crossentropy', \
    #         metrics=[tf.keras.metrics.BinaryAccuracy()], run_eagerly=True)
    # loaded.evaluate(test_x, test_y, batch_size=batch_size, verbose=1)


if __name__ == '__main__':
    main()
