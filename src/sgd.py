import numpy as np
from src.tools import get_batches, get_stochastic_batch, get_stochastic_batches 

def gradient_descent(model_matrix, targets, parameters, derivative, alpha=0, learning_rate=0.001, lr_decay=0.0, momentum=0.0, max_iterations=200, convergence=10e-4):

    """Gradient descent implementation where the full gradient is calculated in each generation.

    Returns:
        np.array: The optimised parameters.
    """

    step = 0
    for i in range(max_iterations):

        learning_rate *= (1 / (1 + lr_decay * max_iterations))
        grad = derivative(model_matrix, targets, parameters, alpha=alpha)
        step = learning_rate * grad + momentum * step
        parameters = parameters - step

        if np.sum(np.abs(step)) < convergence:
            break

    return parameters

def stochastic_gradient_descent(model_matrix, targets, parameters, derivative, alpha=0, batch_size=32, learning_rate=0.001, lr_decay=0.0, momentum=0.0, max_iterations=200, convergence=10e-4):

    """Stochastic gradient descent implementation where in each iteration the whole gradient is approximated by a single randomly sampled batch.

    Returns:
        np.array: The optimised parameters.
    """

    step = 0
    for i in range(max_iterations):

        learning_rate *= (1 / (1 + lr_decay * max_iterations))
        model_matrix_batch, targets_batch = get_stochastic_batch(model_matrix, targets, batch_size)

        grad = derivative(model_matrix_batch, targets_batch, parameters, alpha=alpha)
        step = learning_rate * grad + momentum * step
        parameters = parameters - step

        if np.sum(np.abs(step)) < convergence:
            break

    return parameters


def batch_gradient_descent(model_matrix, targets, parameters, derivative, alpha=0, batch_size=32, learning_rate=0.001, lr_decay=0.0, momentum=0.0, max_iterations=200, convergence=10e-4):

    """Stochastic gradient descent implementation that goes through the dataset in batches. The dataset is shuffled each iteration so that the batches are different in each iteration.

    Returns:
        np.array: The optimised parameters.
    """

    step = 0
    for i in range(max_iterations):

        learning_rate *= (1 / (1 + lr_decay * max_iterations))
        model_matrix_batches, targets_batches = get_batches(model_matrix, targets, batch_size)

        for j in range(len(targets_batches)):

            grad = derivative(model_matrix_batches[j], targets_batches[j], parameters, alpha=alpha)
            step = learning_rate * grad + momentum * step
            parameters = parameters - step


        if np.sum(np.abs(step)) < convergence:
            break

    return parameters

def stochastic_batch_gradient_descent(model_matrix, targets, parameters, derivative, alpha=0, batch_size=32, learning_rate=0.001, lr_decay=0.0, momentum=0.0, max_iterations=200, convergence=10e-4):

    """Stochastic gradient descent implementation that goes through the dataset in batches where each batch is sampled with replacement from the whole dataset.

    Returns:
        np.array: The optimised parameters.
    """

    step = 0
    for i in range(max_iterations):

        learning_rate *= (1 / (1 + lr_decay * max_iterations))
        model_matrix_batches, targets_batches = get_stochastic_batches(model_matrix, targets, batch_size)

        for j in range(len(targets_batches)):

            grad = derivative(model_matrix_batches[j], targets_batches[j], parameters, alpha=alpha)
            step = learning_rate * grad + momentum * step
            parameters = parameters - step


        if np.sum(np.abs(step)) < convergence:
            break

    return parameters