import itertools
import numpy as np

def calculate_beta(model_matrix, targets):
    return np.linalg.pinv(model_matrix.T @ model_matrix) @ model_matrix.T @ targets

def get_prediction(model_matrix, beta, intercept=0):
    return model_matrix @ beta + intercept

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

def center(matrix):

    m_mean = np.mean(matrix, axis=0)
    divisor = np.max(matrix, axis=0) - np.min(matrix, axis=0)

    for i in range(len(matrix)):
        matrix[i] = (matrix[i] - m_mean) / divisor
    
    return matrix, m_mean, divisor

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

    return train_prediction, test_prediction