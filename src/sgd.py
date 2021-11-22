import numpy as np
from src.tools import get_batches, get_stochastic_batch, get_stochastic_batches 

def gradient_descent(model_matrix, targets, parameters, derivative, learning_rate=0.001, learning_rate_decay=0, momentum=0.0, max_iterations=200, convergence=10e-12):

    step = 0
    for i in range(max_iterations):

        grad = derivative(model_matrix, targets, parameters)
        step = learning_rate * grad + momentum * step
        parameters = parameters - step

        if np.sum(np.abs(step)) < convergence:
            break

    return parameters

def stochastic_gradient_descent(model_matrix, targets, parameters, derivative, batch_size=32, learning_rate=0.001, momentum=0.0, max_iterations=200, convergence=10e-10):

    step = 0
    for i in range(max_iterations):

        model_matrix_batch, targets_batch = get_stochastic_batch(model_matrix, targets, batch_size)

        grad = derivative(model_matrix_batch, targets_batch, parameters)
        step = learning_rate * grad + momentum * step
        parameters = parameters - step

        if np.sum(np.abs(step)) < convergence:
            break

    return parameters


def batch_gradient_descent(model_matrix, targets, parameters, derivative, batch_size=32, learning_rate=0.001, momentum=0.0, max_iterations=200, convergence=10e-10):

    step = 0
    for i in range(max_iterations):

        model_matrix_batches, targets_batches = get_batches(model_matrix, targets, batch_size)

        for j in range(len(targets_batches)):

            grad = derivative(model_matrix_batches[j], targets_batches[j], parameters)
            step = learning_rate * grad + momentum * step
            parameters = parameters - step


        if np.sum(np.abs(step)) < convergence:
            break

    return parameters

def stochastic_batch_gradient_descent(model_matrix, targets, parameters, derivative, batch_size=32, learning_rate=0.001, momentum=0.0, max_iterations=200, convergence=10e-10):

    step = 0
    for i in range(max_iterations):

        model_matrix_batches, targets_batches = get_stochastic_batches(model_matrix, targets, batch_size)

        for j in range(len(targets_batches)):

            grad = derivative(model_matrix_batches[j], targets_batches[j], parameters)
            step = learning_rate * grad + momentum * step
            parameters = parameters - step


        if np.sum(np.abs(step)) < convergence:
            break

    return parameters