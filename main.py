from preprocess import get_data, process
from model import Modeler

def main():
    data = get_data('./data/data.csv')
    X, Y = process(data)
    batch_size = 30
    divisible = int(len(Y) // batch_size * batch_size)
    X = X[:divisible]; Y = Y[:divisible]
    # split training/testing data (80/20)
    split = int(0.8*len(Y))
    train_x = X[:split]; train_y = Y[:split]
    test_x = X[split:]; test_y = Y[split:]
    model = Modeler(train_x, train_y, test_x, test_y)
    model.train(batch_size=batch_size, epochs=3)
    model.test(batch_size=batch_size)

if __name__ == '__main__':
    main()
