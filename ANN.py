import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))


class ANN:
    def __init__(self, W):
        self.W = W

    def predict(self, x, t):
        z = x
        for i in range(len(self.W)):
            z = sigmoid(np.dot(z, self.W[i]))
        z = softmax(z)
        z = np.argmax(z, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(z == t) / float(x.shape[0])

        return accuracy


def fitness(Weight, x, t):
    accuracy = np.empty(shape=Weight.shape[0])

    for index in range(Weight.shape[0]):
        current_w = Weight[index, :]

        ann = ANN(current_w)

        accuracy[index] = ann.predict(x, t)
    return accuracy
