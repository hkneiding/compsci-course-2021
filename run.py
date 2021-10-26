from matplotlib.pyplot import axis, plot
import numpy as np
from numpy.core.fromnumeric import mean
from franke_function import franke_function, plot_3d
import itertools

def get_xy_grid_data(lower_bound, upper_bound, steps, flatten=False):

    x = np.linspace(lower_bound, upper_bound, steps)
    y = np.linspace(lower_bound, upper_bound, steps)
    x, y = np.meshgrid(x,y)

    if flatten:
        x = x.flatten()
        y = y.flatten()

    return [x, y]

def calculate_beta(model_matrix, targets):
    return np.linalg.pinv(model_matrix.T @ model_matrix) @ model_matrix.T @ targets

def get_prediction(model_matrix, beta, intercept=0):
    return model_matrix @ beta + intercept

def calculate_r_squared(predictions, targets):

    target_mean = np.mean(targets)
    return 1 - ( np.sum( np.power(targets - predictions, 2) ) / np.sum( np.power(targets - target_mean, 2) ) )

def calculate_cost_mse(predictions, targets):

    # assert equal length
    assert len(predictions) == len(targets)

    return np.mean(np.power(predictions - targets, 2))

def get_model_matrix(data_points, n_pol, include_intercept=True):

    # check that dimensions of features are equal
    for i in range(len(data_points)):
        assert len(data_points[0]) == len(data_points[1])

    # check that the polynom value is of type int
    assert type(n_pol) == int

    model_matrix_columns = []
    if include_intercept:
        model_matrix_columns.append(np.ones(len(data_points[0])))

    # list of data row indices
    indices = list(range(0, len(data_points), 1))

    # intitialize list to store polynomial combinations
    index_combinations = []
    for i in range(1, n_pol + 1, 1):
        index_combinations.extend(list(itertools.combinations_with_replacement(indices, i)))

    # iterate through polynomial combinations
    for i in range(len(index_combinations)):
        
        # iterate through each combination and build appropriate element wise products
        product = None
        for j in range(len(index_combinations[i])):

            # set if first item
            if product is None:
                product = data_points[index_combinations[i][j]]
            # otherwise multiply
            else:
                product = product * data_points[index_combinations[i][j]]

        # append to list
        model_matrix_columns.append(product)

    model_matrix = np.array(model_matrix_columns).T
    return model_matrix

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


def center(matrix):

    m_mean = np.mean(matrix, axis=0)
    divisor = np.max(matrix, axis=0) - np.min(matrix, axis=0)

    for i in range(len(matrix)):
        matrix[i] = (matrix[i] - m_mean) / divisor
    
    return matrix, m_mean, divisor

def min_max_scale(matrix):

    for i in range(len(matrix)):
        print(matrix[i])

    return matrix

def ols(train_data, test_data, n_pol, center_matrix=True):

    if center_matrix:
        # TRAIN
        # set up model matrix for train
        train_model_matrix = get_model_matrix(train_data['inputs'], n_pol, include_intercept=False)
        # center model matrix
        centered_train_model_matrix, train_mean, divisor_scaler = center(train_model_matrix)
        # calculate beta
        beta = calculate_beta(centered_train_model_matrix, train_data['targets'])
        # get train prediction and account for intercept
        train_prediction = get_prediction(centered_train_model_matrix, beta, intercept=np.mean(train_data['targets']))

        # TEST
        # set up model matrix for test
        test_model_matrix = get_model_matrix(test_data['inputs'], n_pol, include_intercept=False)
        # center model matrix according to train center
        centered_test_model_matrix = (test_model_matrix - train_mean) / divisor_scaler
        # get test prediction and account for intercept
        test_prediction = get_prediction(centered_test_model_matrix, beta, intercept=np.mean(train_data['targets']))
    else:
        # TRAIN
        # set up model matrix for train
        train_model_matrix = get_model_matrix(train_data['inputs'], n_pol)
        # calculate beta
        beta = calculate_beta(train_model_matrix, train_data['targets'])
        # get train prediction
        train_prediction = get_prediction(train_model_matrix, beta)

        # TEST
        # set up model matrix for test
        test_model_matrix = get_model_matrix(test_data['inputs'], n_pol)
        # get test prediction
        test_prediction = get_prediction(test_model_matrix, beta)

        # calculate losses
        train_loss = calculate_cost_mse(predictions=train_prediction, targets=train_data['targets'])
        test_loss = calculate_cost_mse(predictions=test_prediction, targets=test_data['targets'])

    # calculate losses
    train_loss = calculate_cost_mse(predictions=train_prediction, targets=train_data['targets'])
    test_loss = calculate_cost_mse(predictions=test_prediction, targets=test_data['targets'])

    train_r_squared = calculate_r_squared(predictions=train_prediction, targets=train_data['targets'])
    test_r_squared = calculate_r_squared(predictions=test_prediction, targets=test_data['targets'])

    return train_loss, test_loss, train_r_squared, test_r_squared

def get_bootstrap_sample(data):

    bootstrap_indices = np.random.choice(len(data['targets']) - 1, len(data['targets']))
    bootstrap_sample_data = { 'inputs': [item[bootstrap_indices] for item in data['inputs']], 'targets': data['targets'][bootstrap_indices] }

    return bootstrap_sample_data

def bootstrap(data, regressor, n_pol, n_samples, train_ratio=0.8, center_matrix=True):

    # lists to store output
    train_losses = []
    test_losses = []

    for i in range(n_samples):

        # get a bootstrap sample
        bootstrap_sample_data = get_bootstrap_sample(data)

        # partion into train and test set
        train_data, test_data = get_train_test_split(bootstrap_sample_data, train_ratio=train_ratio)

        # perform regression and append to output variables
        train_loss, test_loss, train_r_squared, test_r_squared = regressor(train_data, test_data, n_pol, center_matrix=center_matrix)
        train_losses.append(train_loss)
        test_losses.append(test_loss)

    return train_losses, test_losses

def cross_validation(data, regressor, n_pol, n_folds, center_matrix=True):

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
        train_loss, test_loss, train_r_squared, test_r_squared = regressor(train_data, test_data, n_pol, center_matrix=center_matrix)
        train_losses.append(train_loss)
        test_losses.append(test_loss)

    return train_losses, test_losses

def main():

    f_range = 1.22

    # get grid points
    x, y = get_xy_grid_data(0, 1, 10)

    # full data object
    data = { 'inputs': [x.flatten(), y.flatten()], 'targets': franke_function(x, y, noise_std=0).flatten()}
    # data = shuffle(data)

    for i in range(1, 6, 1):

        # train_losses, test_losses = cross_validation(data, regressor=ols, n_pol=i, n_folds=4)
        train_losses, test_losses = bootstrap(data, regressor=ols, n_pol=i, n_samples=10)

        print('Polynom order ' + str(i))
        print('Train Mean:     %.2e' % np.mean(train_losses))
        print('Train Variance: %.2e' % np.var(train_losses))
        print('Test Mean:      %.2e' % np.mean(test_losses))
        print('Test Variance:  %.2e' % np.var(test_losses))
        print()

    exit()
    # split in train and test sets
    train_data, test_data = get_train_test_split(data, train_ratio=0.8)

    print(ols(train_data, test_data, 5, center_matrix=True))
    # print(ols(train_data, test_data, 1, center_matrix=False))
    



    # r_squared = calculate_r_squared(predictions=pred, targets=data['targets'])
    # print(model_matrix)
    # # print(beta)
    # print(data['inputs'])
    # print(data['targets'])
    # print(pred)

    # plot_3d(test_data['inputs'][0], test_data['inputs'][1], test_prediction.reshape(x.shape))
    # plot_3d(data['inputs'][0].reshape(x.shape), data['inputs'][1].reshape(x.shape), data['targets'].reshape(x.shape))

    # plot_3d(x, y, franke_function(x, y, noise_std=0))


    exit()



if __name__ == "__main__":
    main()
