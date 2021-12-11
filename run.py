import math

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from sklearn import svm
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score

from mlxtend.evaluate import bias_variance_decomp

from src.resampling import bootstrap, cross_validation, bias_variance_decomposition
from src.tools import calculate_cost_mse, get_train_test_split, shuffle, scale_min_max
from src.franke_function import franke_function, get_xy_grid_data
from src.regressors import lasso, logistic, ols, ols_sgd, regressor, ridge, ridge_sgd


def get_wisconsin_data():
    
    """Reads the Wisconsin dataset from file applies min-max scaling and returns data as dict."""

    f = open('wdbc.csv', 'r')
    lines = f.readlines()
    f.close()

    features = []
    labels = []
    for i in range(len(lines)):
        line_split = lines[i].split(',')

        if line_split[1] == 'M':
            labels.append(1)
        else:
            labels.append(0)

        features.append(list(map(float, line_split[2:32])))
    
    labels = np.array(labels)
    features = np.array(features)

    data = { 'inputs': [scale_min_max(features[:,i]) for i in range(features.shape[1])], 'targets': labels}

    return data

def regression_bias_variance_decomposition():

    # get grid points
    x, y = get_xy_grid_data(0, 1, 50)

    # full data object
    data = { 'inputs': [x.flatten(), y.flatten()], 'targets': franke_function(x, y, noise_std=0).flatten()}
    data = shuffle(data)

    regressor_parameters = { 'fit_intercept': True, 'alpha': 0 }

    average_error, average_bias, average_variance = bias_variance_decomposition(data, ols, regressor_parameters, n_pol=1, n_samples=1000)
    
    print('Average error: ', average_error)
    print('Average bias: ', average_bias)
    print('Average variance: ', average_variance)

def regression_bias_variance_decomposition_model_data_points():

    regressor_parameters = { 'fit_intercept': True, 'alpha': 0 }

    data_sizes = [10, 20, 30, 40, 50]

    errors = []
    biases = []
    variances = []
    for data_size in data_sizes:

        np.random.seed(2021)

        # get grid points
        x, y = get_xy_grid_data(0, 1, data_size)

        # full data object
        data = { 'inputs': [x.flatten(), y.flatten()], 'targets': franke_function(x, y, noise_std=0).flatten()}
        data = shuffle(data)

        average_error, average_bias, average_variance = bias_variance_decomposition(data, ols, regressor_parameters, n_pol=5, n_samples=1000)
        errors.append(average_error)
        biases.append(average_bias)
        variances.append(average_variance)

    # plt.plot(n_pols, biases)
    plt.show()
    plt.figure(figsize=(10, 6))
    plt.plot(data_sizes, biases, label='Bias')
    plt.plot(data_sizes, variances, label='Variance')
    plt.xlabel('$\sqrt{N_{Data}}$')
    # plt.xticks(range(min(data_sizes), math.ceil(max(data_sizes))+1))
    plt.legend(loc='upper center')
    plt.show()


def regression_bias_variance_decomposition_model_complexity():

    # get grid points
    x, y = get_xy_grid_data(0, 1, 50)

    # full data object
    data = { 'inputs': [x.flatten(), y.flatten()], 'targets': franke_function(x, y, noise_std=0.1).flatten()}
    data = shuffle(data)

    regressor_parameters = { 'fit_intercept': True, 'alpha': 0 }

    n_pols = list(range(1, 21, 1))

    errors = []
    biases = []
    variances = []
    for n_pol in n_pols:

        average_error, average_bias, average_variance = bias_variance_decomposition(data, ols, regressor_parameters, n_pol=n_pol, n_samples=1000)
        errors.append(average_error)
        biases.append(average_bias)
        variances.append(average_variance)

    # plt.plot(n_pols, biases)
    plt.show()
    plt.figure(figsize=(10, 6))
    plt.plot(n_pols, biases, label='Bias')
    plt.plot(n_pols, variances, label='Variance')
    plt.xlabel('Polynomial degree')
    plt.xticks(range(min(n_pols), math.ceil(max(n_pols))+1))
    plt.legend(loc='upper center')
    plt.show()

def regression_lambda_scan():

    np.random.seed(2021)

    # get grid points
    x, y = get_xy_grid_data(0, 1, 50)

    # full data object
    data = { 'inputs': [x.flatten(), y.flatten()], 'targets': franke_function(x, y, noise_std=0.5).flatten()}
    data = shuffle(data)

    errors = []
    lambdas = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]

    for l in lambdas:

        regressor_parameters = { 'fit_intercept': True, 'alpha': l }
        train_losses, test_losses = cross_validation(data, regressor=ridge, regressor_parameters=regressor_parameters, n_pol=5, n_folds=5)
        errors.append(np.mean(test_losses))

    print(errors)
    plt.figure(figsize=(10, 6))
    plt.plot(lambdas, errors, 'o')
    plt.xscale('log')
    plt.xlabel('$\lambda$')
    plt.ylabel('MSE')
    # plt.legend(loc='upper right')
    plt.show()

def regression_cross_validation():

    np.random.seed(2021)

    # get grid points
    x, y = get_xy_grid_data(0, 1, 50)

    # full data object
    data = { 'inputs': [x.flatten(), y.flatten()], 'targets': franke_function(x, y, noise_std=0).flatten()}
    data = shuffle(data)

    errors = []
    variances = []

    fold_numbers = [5, 6, 7, 8, 9, 10]
    for fold_number in fold_numbers:

        regressor_parameters = { 'fit_intercept': True, 'alpha': 0 }
        train_losses, test_losses = cross_validation(data, regressor=ols, regressor_parameters=regressor_parameters, n_pol=10, n_folds=fold_number)
        errors.append(np.mean(test_losses))
        variances.append(np.var(test_losses))

    print(errors)
    plt.figure(figsize=(10, 6))
    plt.plot(fold_numbers, errors, 'o', label='Test error')
    # plt.plot(fold_numbers, variances, 'o', label='Variance')
    # plt.legend(loc='upper right')
    plt.xlabel('Number of folds')
    plt.ylabel('Test error')
    plt.show()

def regression_bias_variance_decomposition_mlxtend():

    # get grid points
    x, y = get_xy_grid_data(0, 1, 50)

    # full data object
    data = { 'inputs': [x.flatten(), y.flatten()], 'targets': franke_function(x, y, noise_std=0).flatten()}
    data = shuffle(data)

    train_data, test_data = get_train_test_split(data, train_ratio=0.8)

    model = LinearRegression()
    mse, bias, var = bias_variance_decomp(model, 
                                        X_train=np.array(train_data['inputs']).T, 
                                        y_train=train_data['targets'],
                                        X_test=np.array(test_data['inputs']).T,
                                        y_test=test_data['targets'],
                                        loss='mse',
                                        num_rounds=1000)

    print(mse)
    print(bias)
    print(var)

def main_regression():

    # X = np.array([[2,1]])
    # Y = np.array([1.2])
    # beta = np.array([[1,1]])

    # print(gradient_descent(X, Y, beta, calculate_cost_derivative_mse, max_iterations=100, momentum=0))
    # print(stochastic_gradient_descent(X, Y, beta, calculate_cost_derivative_mse, max_iterations=100, momentum=0))


    # np.random.seed(1)

    # get grid points
    x, y = get_xy_grid_data(0, 1, 50)

    # full data object
    data = { 'inputs': [x.flatten(), y.flatten()], 'targets': franke_function(x, y, noise_std=0).flatten()}
    data = shuffle(data)
    train_data, test_data = get_train_test_split(data, 0.9)

    for i in range(1,6,1):
        regressor_parameters = { 'fit_intercept': True, 'alpha': 0.0001 }
        
        train_pred, test_pred = lasso(regressor_parameters, train_data, test_data, n_pol=i)
        print(calculate_cost_mse(test_pred, test_data['targets']))

    exit()

    pol_train_loss = []
    pol_test_loss = []
    n_pols=[i for i in range(1, 6, 1)]
    for i in n_pols:

        # alpha is only needed for LASSO or ridge where you need additional hyperparameter to 
        # control how much the magnitude of the parameter vector should be penalised
        regressor_parameters = { 'fit_intercept': True, 'alpha': 0.5 }


        regressor_parameters = { 'fit_intercept': True, 
                                 'alpha': 0.5,
                                 'learning_rate': 0.1,
                                 'max_iterations': 10000,
                                 'momentum': 0,
                                 'batch_size': 10
                               }
        
        # train_prediction, test_prediction = ols_sgd(regressor_parameters=regressor_parameters, train_data=train_data, test_data=test_data, n_pol=i)

        # train_loss = calculate_cost_mse(train_prediction, train_data['targets'])
        # test_loss = calculate_cost_mse(test_prediction, test_data['targets'])

        # pol_train_loss.append(train_loss)
        # pol_test_loss.append(test_loss)
        
        
        train_losses, test_losses = cross_validation(data, regressor=ols, regressor_parameters=regressor_parameters, n_pol=i, n_folds=10)
        # train_losses, test_losses = bootstrap(data, regressor=lasso, regressor_parameters=regressor_parameters, n_pol=i, n_samples=30)


        pol_train_loss.append(np.mean(train_losses))
        pol_test_loss.append(np.mean(test_losses))

        # print('Polynom order ' + str(i))
        # print('Train Mean:     %.2e' % np.mean(train_losses))
        # print('Train Variance: %.2e' % np.var(train_losses))
        # print('Test Mean:      %.2e' % np.mean(test_losses))
        # print('Test Variance:  %.2e' % np.var(test_losses))
        # print()


    print(pol_train_loss)
    plt.plot(n_pols, pol_train_loss, label='Train error')
    plt.plot(n_pols, pol_test_loss, label='Test error')
    plt.legend(loc='upper right')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylabel('Cost')
    plt.xlabel('Polynomial order')
    plt.show()

    exit()


def main_logistic():
    
    data = get_wisconsin_data()
    data = shuffle(data)

    regressor_parameters = { 'fit_intercept': False, 
                                'alpha': 0.1,
                                'learning_rate': 0.01,
                                'max_iterations': 10000,
                                'momentum': 0.1,
                                'batch_size': 20
                            }

    train_data, test_data = get_train_test_split(data, train_ratio=0.66)
    train_prediction, test_prediction = logistic(regressor_parameters, train_data, test_data, 1)

    test_prediction_binary = np.array(test_prediction > 0.5, dtype='int64')

    # print(test_prediction_binary)
    # print(test_data['targets'])
    accuracy = 1 - np.sum(np.absolute(test_prediction_binary - test_data['targets']))/len(test_data['targets'])

    print(accuracy)

    exit()

def logistic_lr_scan():

    np.random.seed(1)

    data = get_wisconsin_data()
    data = shuffle(data)

    # lrs = [0.0001, 0.001, 0.01, 0.1, 1.0]
    lrs = np.arange(0.005, 0.105, 0.005)
    lrs = np.concatenate(([0.001], lrs))
    lrs = np.concatenate((lrs, [0.125, 0.15, 0.175, 0.2, 0.3, 0.4, 0.5] ))

    print(len(lrs))

    train_losses_full = []
    test_losses_full = []
    for i in range(len(lrs)):


        regressor_parameters = { 
                                    'fit_intercept': True, 
                                    'alpha': 0,
                                    'learning_rate': lrs[i],
                                    'max_iterations': 1000,
                                    'momentum': 0,
                                    'batch_size': 20
                                }


        train_losses, test_losses = cross_validation(data, regressor=logistic, regressor_parameters=regressor_parameters, n_pol=1, n_folds=6)

        train_losses_full.append(np.mean(train_losses))
        test_losses_full.append(np.mean(test_losses))

    print(train_losses_full)
    print(test_losses_full)

def logistic_momentum_scan():

    np.random.seed(1)

    data = get_wisconsin_data()
    data = shuffle(data)

    mom = [0.0001, 0.001, 0.01, 0.1, 1.0]
    mom = np.arange(0.001, 0.0101, 0.001)
    mom = np.concatenate(([0.0001, 0.0002, 0.0003, 0.0004, 0.0005], mom))

    print(len(mom))

    train_losses_full = []
    test_losses_full = []
    for i in range(len(mom)):


        regressor_parameters = { 
                                    'fit_intercept': True, 
                                    'alpha': 0,
                                    'learning_rate': 0.085,
                                    'max_iterations': 1000,
                                    'momentum': mom[i],
                                    'batch_size': 20
                                }


        train_losses, test_losses = cross_validation(data, regressor=logistic, regressor_parameters=regressor_parameters, n_pol=1, n_folds=6)

        train_losses_full.append(np.mean(train_losses))
        test_losses_full.append(np.mean(test_losses))

    print(train_losses_full)
    print(test_losses_full)

def logistic_alpha_scan():

    np.random.seed(1)

    data = get_wisconsin_data()
    data = shuffle(data)

    alphas = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
    alphas = np.arange(0.00001, 0.00011, 0.00001)

    print(len(alphas))

    train_losses_full = []
    test_losses_full = []
    for i in range(len(alphas)):

        np.random.seed(1)
        regressor_parameters = { 
                                    'fit_intercept': True, 
                                    'alpha': alphas[i],
                                    'learning_rate': 0.085,
                                    'max_iterations': 1000,
                                    'momentum': 0,
                                    'batch_size': 20
                                }


        train_losses, test_losses = cross_validation(data, regressor=logistic, regressor_parameters=regressor_parameters, n_pol=1, n_folds=6)

        train_losses_full.append(np.mean(train_losses))
        test_losses_full.append(np.mean(test_losses))

    print(train_losses_full)
    print(test_losses_full)

def logistic_scikit_compare():
    
    data = get_wisconsin_data()
    data = shuffle(data)

    # setup skl predictor
    predictor = LogisticRegression(C=1 / (8 * 10 ** (-5)), solver='sag', max_iter=10000)
    # setup regressor parameters
    regressor_parameters = { 
                                'fit_intercept': True, 
                                'alpha': 8 * 10 ** (-5),
                                'learning_rate': 0.085,
                                'max_iterations': 10000,
                                'momentum': 0.001,
                                'batch_size': 1
                            }

    skl_score_collection = []
    own_score_collection = []
    for i in range(20):
    
        data = shuffle(data)
        print(i)
        skl_scores = cross_val_score(predictor, np.array(data['inputs']).T, data['targets'], cv=6)
        train_losses, test_losses = cross_validation(data, regressor=logistic, regressor_parameters=regressor_parameters, n_pol=1, n_folds=6)

        skl_score_collection.append(np.mean(skl_scores))
        own_score_collection.append(np.mean(test_losses))
    
    print(skl_score_collection)
    print(own_score_collection)

def svm_scikit():

    data = get_wisconsin_data()

    # define SVM
    predictor = svm.SVC(kernel='linear', C=1, tol=10e-4)
    # predictor = svm.SVC(kernel='poly', degree=5, C=1, tol=10e-4)
        
    score_collection = []
    for i in range(20):

        data = shuffle(data)
        scores = cross_val_score(predictor, np.array(data['inputs']).T, data['targets'], cv=6)

        score_collection.append(np.mean(scores))

    print(score_collection)
    print(np.mean(score_collection))

def svm_scikit_c_scan():

    data = get_wisconsin_data()
    data = shuffle(data)

    C_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

    score_collection = []
    for C_value in C_values:

        # define SVM
        predictor = svm.SVC(kernel='linear', C=C_value, tol=10e-4)

        scores = cross_val_score(predictor, np.array(data['inputs']).T, data['targets'], cv=6)
        score_collection.append(np.mean(scores))

    print(score_collection)

    plt.figure(figsize=(10, 6))
    plt.plot(C_values, score_collection, 'o', label='scikit-learn')
    plt.xscale('log')
    plt.xlabel('C value')
    plt.ylabel('Accuracy')
    plt.show()

def svm_scikit_poly_scan():

    data = get_wisconsin_data()
    data = shuffle(data)

    p_values = [1, 2, 3, 4, 5]

    score_collection = []
    for p_value in p_values:

        # define SVM
        predictor = svm.SVC(kernel='poly', degree=p_value, C=1, tol=10e-4)

        scores = cross_val_score(predictor, np.array(data['inputs']).T, data['targets'], cv=6)
        score_collection.append(np.mean(scores))

    print(score_collection)
    
    plt.figure(figsize=(10, 6))
    plt.plot(p_values, score_collection, 'o', label='scikit-learn')
    plt.xlabel('Polynomial degree')
    plt.xticks(range(min(p_values), math.ceil(max(p_values))+1))
    plt.ylabel('Accuracy')
    plt.show()

if __name__ == "__main__":

    # # regression
    # regression_bias_variance_decomposition()
    # regression_bias_variance_decomposition_mlxtend()
    regression_bias_variance_decomposition_model_data_points()
    regression_bias_variance_decomposition_model_complexity()
    # regression_cross_validation()
    # regression_lambda_scan()

    # # logistic regression
    # logistic_lr_scan()
    # logistic_momentum_scan()
    # logistic_alpha_scan()
    # logistic_scikit_compare()

    # # support vector machines
    # svm_scikit()
    # svm_scikit_c_scan()
    # svm_scikit_poly_scan()