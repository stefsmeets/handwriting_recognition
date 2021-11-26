import numpy as np
from cost import nn_cost_function

def check_nn_gradients(λ=0):
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    theta_1 = debug_initialize_weights(hidden_layer_size, input_layer_size)
    theta_2 = debug_initialize_weights(num_labels, hidden_layer_size)

    X = debug_initialize_weights(m, input_layer_size - 1)
    y = np.mod(np.arange(0, m), num_labels).reshape(-1,1)

    nn_params = np.hstack([theta_1.ravel(), theta_2.ravel()])

    from functools import partial

    cost_func = partial(nn_cost_function,     
        input_layer_size=input_layer_size, 
        hidden_layer_size=hidden_layer_size, 
        num_labels=num_labels,
        X=X,
        y=y,
        λ=λ,
    )

    cost, grad = cost_func(nn_params=nn_params)

    numgrad = compute_numerical_gradient(cost_func, nn_params)

    print('Checking backprop')
    print('left: backprop, right: numerical')
    print(np.vstack([grad, numgrad]).T)

    diff = np.linalg.norm(numgrad - grad) / np.linalg.norm(numgrad + grad)
    print(f'{diff=}')
    assert diff < 1e-9


def debug_initialize_weights(fan_out, fan_in):
    W = np.zeros((fan_out, 1+fan_in))
    s = np.sin(np.arange(1, W.size+1))
    return np.reshape(s, W.shape) / 10


def compute_numerical_gradient(J, theta):
    numgrad = np.zeros(theta.shape)
    perturb = np.zeros(theta.shape)
    
    e = 1e-4
    
    for p in range(theta.size):
        perturb[p] = e
        loss1, _ = J(theta - perturb)
        loss2, _ = J(theta + perturb)
        
        numgrad[p] = (loss2 - loss1) / (2*e)
        perturb[p] = 0
    
    return numgrad
