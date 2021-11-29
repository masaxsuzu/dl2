import numpy as np


class Sigmoid:

    def __init__(self):
        self.params = []

    def forward(self, x):
        return 1 / (1 + np.exp(-x))


class Affine:

    def __init__(self, w, b):
        self.params = [w, b]

    def forward(self, x):
        w, b = self.params
        out = np.dot(x, w) + b
        return out


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size):
        i, h, o = input_size, hidden_size, output_size

        w1 = np.random.randn(i, h)
        b1 = np.random.randn(h)

        w2 = np.random.randn(h, o)
        b2 = np.random.randn(o)

        self.layers = [
            Affine(w1, b1),
            Sigmoid(),
            Affine(w2, b2),
        ]

        self.params = []
        for layer in self.layers:
            self.params += layer.params

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
