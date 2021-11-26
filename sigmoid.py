import numpy as np

def sigmoid(z):
    g = 1.0 / (1.0 + np.exp(-z))
    return g

def sigmoid_gradient(z):
    g0 = sigmoid(z)
    g = g0 * (1 - g0)
    return g