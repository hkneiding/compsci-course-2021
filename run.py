import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from src.tools import bootstrap, calculate_cost_mse, cross_validation, get_train_test_split, shuffle, scale_min_max
from src.franke_function import franke_function, get_xy_grid_data
from src.regressors import lasso, logistic, ols, ols_sgd, regressor, ridge, ridge_sgd


def main_regression():

    # X = np.array([[2,1]])
    # Y = np.array([1.2])
    # beta = np.array([[1,1]])

    # print(gradient_descent(X, Y, beta, calculate_cost_derivative_mse, max_iterations=100, momentum=0))
    # print(stochastic_gradient_descent(X, Y, beta, calculate_cost_derivative_mse, max_iterations=100, momentum=0))



    # np.random.seed(1)

    # get grid points
    x, y = get_xy_grid_data(0, 1, 60)

    # full data object
    data = { 'inputs': [x.flatten(), y.flatten()], 'targets': franke_function(x, y, noise_std=0).flatten()}
    data = shuffle(data)
    train_data, test_data = get_train_test_split(data, 0.8)


    pol_train_loss = []
    pol_test_loss = []
    n_pols=[i for i in range(1, 6, 1)]
    for i in n_pols:

        # alpha is only needed for LASSO or ridge where you need additional hyperparameter to 
        # control how much the magnitude of the parameter vector should be penalised
        regressor_parameters = { 'fit_intercept': True, 'alpha': 0.5 }


        regressor_parameters = { 'fit_intercept': True, 
                                 'alpha': 0.1,
                                 'learning_rate': 0.1,
                                 'max_iterations': 1000,
                                 'momentum': 0,
                                 'batch_size': 100
                               }
        
        # train_prediction, test_prediction = ols_sgd(regressor_parameters=regressor_parameters, train_data=train_data, test_data=test_data, n_pol=i)

        # train_loss = calculate_cost_mse(train_prediction, train_data['targets'])
        # test_loss = calculate_cost_mse(test_prediction, test_data['targets'])

        # pol_train_loss.append(train_loss)
        # pol_test_loss.append(test_loss)
        
        
        train_losses, test_losses = cross_validation(data, regressor=ridge, regressor_parameters=regressor_parameters, n_pol=i, n_folds=10)
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
    data = shuffle(data)

    regressor_parameters = { 'fit_intercept': False, 
                                'alpha': 0.1,
                                'learning_rate': 0.005,
                                'max_iterations': 40000,
                                'momentum': 0,
                                'batch_size': 10
                            }

    train_data, test_data = get_train_test_split(data, train_ratio=0.66)
    train_prediction, test_prediction = logistic(regressor_parameters, train_data, test_data, 1)

    test_prediction_binary = np.array(test_prediction > 0.5, dtype='int64')

    # print(test_prediction_binary)
    # print(test_data['targets'])
    accuracy = 1 - np.sum(np.absolute(test_prediction_binary - test_data['targets']))/len(test_data['targets'])

    print(accuracy)

    exit()

if __name__ == "__main__":
    # main_regression()
    main_logistic()
    #f_range = 1.22
