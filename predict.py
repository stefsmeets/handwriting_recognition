import numpy as np
from sigmoid import sigmoid

def predict(Θ1, Θ2, X):
    m = X.shape[0]

    h1 = sigmoid(np.c_[np.ones((m, 1)), X] @ Θ1.T)
    h2 = sigmoid(np.c_[np.ones((m, 1)), h1] @ Θ2.T)

    p = np.argmax(h2, axis=1)
    conf = np.take_along_axis(h2, np.expand_dims(p, axis=1), axis=1)

    return p.reshape(-1,1), conf.reshape(-1,1)