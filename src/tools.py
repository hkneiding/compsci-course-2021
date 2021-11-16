import numpy as np

def calculate_r_squared(predictions, targets):

    target_mean = np.mean(targets)
    return 1 - ( np.sum( np.power(targets - predictions, 2) ) / np.sum( np.power(targets - target_mean, 2) ) )

def calculate_cost_mse(predictions, targets):

    # assert equal length
    assert len(predictions) == len(targets)

    return np.mean(np.power(predictions - targets, 2))

def calculate_cost_derivative_mse(X, Y, beta):

    return 1 / len(X) * (X @ beta - Y) @ X

def calculate_cost_derivative_ridge(X, Y, beta):

    return 1 / len(X) * (X @ beta - Y) @ X + 2 * beta

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

def get_bootstrap_sample(data):

    bootstrap_indices = np.random.choice(len(data['targets']) - 1, len(data['targets']))
    bootstrap_sample_data = { 'inputs': [item[bootstrap_indices] for item in data['inputs']], 'targets': data['targets'][bootstrap_indices] }

    return bootstrap_sample_data

def bootstrap(data, regressor, regressor_parameters, n_pol, n_samples, train_ratio=0.8):

    # lists to store output
    train_losses = []
    test_losses = []

    for i in range(n_samples):

        # partion into train and test set
        train_data, test_data = get_train_test_split(data, train_ratio=train_ratio)

        # get a bootstrap sample
        bootstrap_sample_data_train = get_bootstrap_sample(train_data)

        # perform regression and append to output variables
        train_prediction, test_prediction = regressor(regressor_parameters, bootstrap_sample_data_train, test_data, n_pol)
        train_losses.append(calculate_cost_mse(train_prediction, bootstrap_sample_data_train['targets']))
        test_losses.append(calculate_cost_mse(test_prediction, test_data['targets']))

    return train_losses, test_losses

def cross_validation(data, regressor, regressor_parameters, n_pol, n_folds):

    # check that data can be partioned into n_folds folds
    assert len(data['targets']) % n_folds == 0
    fold_length = len(data['targets']) // n_folds

    # lists to store output
    train_losses = []
    test_losses = []

    for i in range(n_folds):
        
        # get indices of validation data points for this fold
        test_indices = list(range(i * fold_length, (i + 1) * fold_length, 1))

        # assign train and test data for this fold
        test_data = { 'inputs': [item[test_indices] for item in data['inputs']], 'targets': data['targets'][test_indices] }
        train_data = { 'inputs': [np.delete(item, test_indices) for item in data['inputs']], 'targets': np.delete(data['targets'], test_indices) }

        # perform regression and append to output variables
        train_prediction, test_prediction = regressor(regressor_parameters, train_data, test_data, n_pol)
        train_losses.append(calculate_cost_mse(train_prediction, train_data['targets']))
        test_losses.append(calculate_cost_mse(test_prediction, test_data['targets']))

    return train_losses, test_losses

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
