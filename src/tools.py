import numpy as np
from numpy.core.fromnumeric import size


def calculate_r_squared(predictions, targets):

    target_mean = np.mean(targets)
    return 1 - ( np.sum( np.power(targets - predictions, 2) ) / np.sum( np.power(targets - target_mean, 2) ) )

def calculate_cost_mse(predictions, targets):

    # assert equal length
    assert len(predictions) == len(targets)

    return np.mean(np.power(predictions - targets, 2))

def calculate_accuracy(predictions, targets):
    return 1 - np.sum(np.absolute(predictions - targets))/len(targets)

def calculate_cost_derivative_mse(X, Y, beta, alpha=0):

    return 1 / len(X) * (X @ beta - Y) @ X + 2 * alpha * beta

def calculate_cost_derivative_ridge(X, Y, beta, alpha):

    return 1 / len(X) * (X @ beta - Y) @ X + 2 * alpha * beta

def calculate_sigmoid(array_data):
    return 1/(1 + np.exp(-array_data))

def calculate_cost_derivative_logistic(X, Y, beta, alpha=0):
    weights_applied =  np.dot(X, beta)
    prediction = calculate_sigmoid(weights_applied)
    return 1 / X.shape[0] * (X.T @ (prediction - Y)) + 2 * alpha * beta

def scale_min_max(x):

    return (x - np.min(x)) / (np.max(x) - np.min(x))

def shuffle(data):

    # check that dimensions of inputs and targets match
    for i in range(len(data['inputs'])):
        assert len(data['inputs'][i]) == len(data['targets'])

    permuation = np.random.permutation(len(data['targets']))

    data['inputs'] = [item[permuation] for item in data['inputs']]
    data['targets'] = data['targets'][permuation]

    return data

def get_train_test_split(data, train_ratio):

    # check that ration r is between 0 an 1
    assert train_ratio >= 0 and train_ratio <= 1

    # check that dimensions of inputs and targets match
    for i in range(len(data['inputs'])):
        assert len(data['inputs'][i]) == len(data['targets'])

    # get train data last index
    train_last_index = int(np.floor(train_ratio * len(data['targets'])))
    
    # get train data
    train_split = { 'inputs': [item[:train_last_index] for item in data['inputs']], 
                    'targets': data['targets'][:train_last_index]}

    # get test data
    test_split = { 'inputs': [item[train_last_index:] for item in data['inputs']], 
                    'targets': data['targets'][train_last_index:]}

    return train_split, test_split

def get_batches(model_matrix, targets, batch_size):

    # shuffle data
    perm = np.random.permutation(len(targets))

    model_matrix = model_matrix[perm]
    targets = targets[perm]

    n_batches = int(np.ceil(len(targets) / batch_size))

    model_matrix_batches = []
    targets_batches =[]

    running_index = 0
    for i in range(n_batches):

        model_matrix_batches.append(model_matrix[running_index:running_index + batch_size])
        targets_batches.append(targets[running_index:running_index + batch_size])

    return model_matrix_batches, targets_batches

def get_stochastic_batches(model_matrix, targets, batch_size):

    n_batches = int(np.ceil(len(targets) / batch_size))

    model_matrix_batches = []
    targets_batches =[]
    for i in range(n_batches):

        model_matrix_batch, targets_batch = get_stochastic_batch(model_matrix, targets, batch_size)

        model_matrix_batches.append(model_matrix_batch)
        targets_batches.append(targets_batch)


    return model_matrix_batches, targets_batches


def get_stochastic_batch(model_matrix, targets, batch_size):

    indices = np.random.randint(low=0, high=len(targets), size=batch_size)

    return model_matrix[indices], targets[indices]