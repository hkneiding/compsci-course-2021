import matplotlib
from matplotlib.pyplot import axis
import numpy as np

from .regressors import logistic
from .tools import get_train_test_split, calculate_accuracy, calculate_cost_mse

matplotlib.rcParams.update({'font.size': 12})

def get_bootstrap_sample(data):

    """Draws a single bootstrap sample.

    Returns:
        dict: Data dict of bootstrap sample.
    """

    bootstrap_indices = np.random.choice(len(data['targets']) - 1, len(data['targets']))
    bootstrap_sample_data = { 'inputs': [item[bootstrap_indices] for item in data['inputs']], 'targets': data['targets'][bootstrap_indices] }

    return bootstrap_sample_data

def bias_variance_decomposition(data, regressor, regressor_parameters, n_pol, n_samples, train_ratio=0.8):

    """Estimates the test bias and variance using bootstrap.

    Returns:
        float: Average error.
        float: Average bias.
        float: Average variance.
    """

    test_predictions = []

    # partion into train and test set
    train_data, test_data = get_train_test_split(data, train_ratio=train_ratio)

    for i in range(n_samples):

        # get a bootstrap sample
        bootstrap_sample_data_train = get_bootstrap_sample(train_data)

        # perform regression and append to output variables
        train_prediction, test_prediction = regressor(regressor_parameters, bootstrap_sample_data_train, test_data, n_pol)
        
        test_predictions.append(test_prediction)

    test_predictions = np.array(test_predictions)
    average_test_predictions = np.mean(test_predictions, axis=0)

    error = calculate_cost_mse(average_test_predictions, test_data['targets'])
    bias = np.mean( (average_test_predictions - test_data['targets']) ** 2)
    variance = np.mean(np.var(test_predictions, axis=0))

    return error, bias, variance

def bootstrap(data, regressor, regressor_parameters, n_pol, n_samples, train_ratio=0.8):

    """Performs bootstrapping for a supplied regressor.

    Returns:
        np.array: Sample-wise train predictions.
        np.array: Sample-wise test predictions.
    """

    # lists to store output
    train_losses = []
    test_losses = []

    # partion into train and test set
    train_data, test_data = get_train_test_split(data, train_ratio=train_ratio)

    for i in range(n_samples):

        # get a bootstrap sample
        bootstrap_sample_data_train = get_bootstrap_sample(train_data)

        # perform regression and append to output variables
        train_prediction, test_prediction = regressor(regressor_parameters, bootstrap_sample_data_train, test_data, n_pol)
        
        if regressor == logistic:
            train_losses.append(calculate_accuracy(np.array(train_prediction > 0.5, dtype='int64'), bootstrap_sample_data_train['targets']))
            test_losses.append(calculate_accuracy(np.array(test_prediction > 0.5, dtype='int64'), test_data['targets']))
        else:
            train_losses.append(calculate_cost_mse(train_prediction, bootstrap_sample_data_train['targets']))
            test_losses.append(calculate_cost_mse(test_prediction, test_data['targets']))

    return train_losses, test_losses

def cross_validation(data, regressor, regressor_parameters, n_pol, n_folds):

    """Performs k-fold cross validation for a supplied regressor.

    Returns:
        np.array: Sample-wise train predictions.
        np.array: Sample-wise test predictions.
    """

    # get fold length
    fold_length = int(np.ceil(len(data['targets']) / n_folds))

    # lists to store output
    train_losses = []
    test_losses = []

    for i in range(n_folds):
        
        # get indices of validation data points for this fold
        start = i * fold_length
        stop = min([(i + 1) * fold_length, len(data['targets']) - 1])
        test_indices = list(range(start, stop, 1))

        # assign train and test data for this fold
        test_data = { 'inputs': [item[test_indices] for item in data['inputs']], 'targets': data['targets'][test_indices] }
        train_data = { 'inputs': [np.delete(item, test_indices) for item in data['inputs']], 'targets': np.delete(data['targets'], test_indices) }

        # perform regression and append to output variables
        train_prediction, test_prediction = regressor(regressor_parameters, train_data, test_data, n_pol)
        
        if regressor == logistic:
            train_losses.append(calculate_accuracy(np.array(train_prediction > 0.5, dtype='int64'), train_data['targets']))
            test_losses.append(calculate_accuracy(np.array(test_prediction > 0.5, dtype='int64'), test_data['targets']))
        else:
            train_losses.append(calculate_cost_mse(train_prediction, train_data['targets']))
            test_losses.append(calculate_cost_mse(test_prediction, test_data['targets']))

    return train_losses, test_losses
