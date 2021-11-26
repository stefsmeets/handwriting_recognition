import numpy as np
from sigmoid import sigmoid, sigmoid_gradient


def unroll(*args):
    return np.hstack([arr.ravel() for arr in args])


def rand_initialize_weights(L_in, L_out):
    ε_init = 0.12
    W = np.random.random(size=(L_out, 1 + L_in)) * 2 * ε_init - ε_init
    return W


def nn_cost_function(nn_params, 
        input_layer_size, 
        hidden_layer_size, 
        num_labels,
        X,
        y,
        λ=0,     
    ):
    """Cost function for neural network."""
    m = X.shape[0]

    # Repack Theta1 / Theta2

    Θ1 = nn_params[:hidden_layer_size * (input_layer_size + 1)]
    Θ1 = Θ1.reshape(hidden_layer_size, input_layer_size + 1)

    Θ2 = nn_params[(hidden_layer_size * (input_layer_size + 1)):]
    Θ2 = Θ2.reshape(num_labels, (hidden_layer_size + 1))

    # Calculate cost (J)

    a1 = np.c_[np.ones(m), X]

    z2 = a1 @ Θ1.T

    a2 = sigmoid(z2)
    a2 = np.c_[np.ones(a2.shape[0]), a2]

    z3 = a2 @ Θ2.T

    h0 = a3 = sigmoid(z3)

    y = np.int16(np.arange(1, num_labels+1) == y)

    J = (1/m) * np.sum(np.sum((-y * np.log(h0)) - ((1 - y) * np.log(1 - h0))))

    # Regularization

    Θ1_regl = np.sum(Θ1[:,1:]**2)
    Θ2_regl = np.sum(Θ2[:,1:]**2)

    regl = (λ / (2*m)) * (Θ1_regl + Θ2_regl) 

    J = J + regl

    # Back propagation

    Δ3 = a3 - y

    a2_grad = sigmoid_gradient(z2)
    a2_grad = np.c_[np.ones(a2_grad.shape[0]), a2_grad]

    Δ2 = (Δ3 @ Θ2) * a2_grad
    Δ2 = Δ2[:, 1:]  # remove bias

    Θ1_grad = (1/m) * (Δ2.T @ a1)
    Θ2_grad = (1/m) * (Δ3.T @ a2)

    # Regularization for back propagation
    
    Θ1_grad[:, 1:] = Θ1_grad[:, 1:] + (λ / m) * Θ1[:, 1:]
    Θ2_grad[:, 1:] = Θ2_grad[:, 1:] + (λ / m) * Θ2[:, 1:]

    # Unroll gradients

    grad = unroll(Θ1_grad, Θ2_grad)

    return J, grad


def nn_cost(nn_params, 
        input_layer_size, 
        hidden_layer_size, 
        num_labels,
        X,
        y,
        λ=0,     
    ):
    m = X.shape[0]

    # Repack Theta1 / Theta2

    Θ1 = nn_params[:hidden_layer_size * (input_layer_size + 1)]
    Θ1 = Θ1.reshape(hidden_layer_size, input_layer_size + 1)

    Θ2 = nn_params[(hidden_layer_size * (input_layer_size + 1)):]
    Θ2 = Θ2.reshape(num_labels, (hidden_layer_size + 1))

    # Calculate cost (J)

    a1 = np.c_[np.ones(m), X]

    z2 = a1 @ Θ1.T

    a2 = sigmoid(z2)
    a2 = np.c_[np.ones(a2.shape[0]), a2]

    z3 = a2 @ Θ2.T

    h0 = a3 = sigmoid(z3)

    y = np.int16(np.arange(1, num_labels+1) == y)

    J = (1/m) * np.sum(np.sum((-y * np.log(h0)) - ((1 - y) * np.log(1 - h0))))

    # Regularization

    Θ1_regl = np.sum(Θ1[:,1:]**2)
    Θ2_regl = np.sum(Θ2[:,1:]**2)

    regl = (λ / (2*m)) * (Θ1_regl + Θ2_regl) 

    J = J + regl

    return J


def nn_fprime(nn_params, 
        input_layer_size, 
        hidden_layer_size, 
        num_labels,
        X,
        y,
        λ=0,     
    ):
    m = X.shape[0]

    # Repack Theta1 / Theta2

    Θ1 = nn_params[:hidden_layer_size * (input_layer_size + 1)]
    Θ1 = Θ1.reshape(hidden_layer_size, input_layer_size + 1)

    Θ2 = nn_params[(hidden_layer_size * (input_layer_size + 1)):]
    Θ2 = Θ2.reshape(num_labels, (hidden_layer_size + 1))

    # Calculate cost (J)

    a1 = np.c_[np.ones(m), X]

    z2 = a1 @ Θ1.T

    a2 = sigmoid(z2)
    a2 = np.c_[np.ones(a2.shape[0]), a2]

    z3 = a2 @ Θ2.T

    a3 = sigmoid(z3)

    y = np.int16(np.arange(1, num_labels+1) == y)

    # Back propagation

    Δ3 = a3 - y

    a2_grad = sigmoid_gradient(z2)
    a2_grad = np.c_[np.ones(a2_grad.shape[0]), a2_grad]

    Δ2 = (Δ3 @ Θ2) * a2_grad
    Δ2 = Δ2[:, 1:]  # remove bias

    Θ1_grad = (1/m) * (Δ2.T @ a1)
    Θ2_grad = (1/m) * (Δ3.T @ a2)

    # Regularization for back propagation
    
    Θ1_grad[:, 1:] = Θ1_grad[:, 1:] + (λ / m) * Θ1[:, 1:]
    Θ2_grad[:, 1:] = Θ2_grad[:, 1:] + (λ / m) * Θ2[:, 1:]

    # Unroll gradients

    grad = unroll(Θ1_grad, Θ2_grad)

    return grad
