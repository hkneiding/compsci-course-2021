import numpy as np

from src.tools import bootstrap, calculate_cost_derivative_mse, calculate_cost_mse, cross_validation, get_train_test_split, shuffle
from src.franke_function import franke_function, get_xy_grid_data
from src.regressors import lasso, ols, ols_sgd, regressor, ridge
from src.sgd import gradient_descent, stochastic_gradient_descent

import matplotlib.pyplot as plt

def main():

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
    train_data, test_data = get_train_test_split(data, 0.8)


    pol_train_loss = []
    pol_test_loss = []
    n_pols=[i for i in range(1, 6, 1)]
    for i in n_pols:

        # alpha is only needed for LASSO or ridge where you need additional hyperparameter to 
        # control how much the magnitude of the parameter vector should be penalised
        regressor_parameters = { 'fit_intercept': True, 'alpha': 0.5 }


        regressor_parameters = { 'fit_intercept': True, 
                                 'alpha': 0.5,
                                 'learning_rate': 0.001,
                                 'max_iterations': 10000,
                                 'momentum': 0,
                                 'batch_size': 100
                               }
        
        # train_prediction, test_prediction = ols_sgd(regressor_parameters=regressor_parameters, train_data=train_data, test_data=test_data, n_pol=i)

        # train_loss = calculate_cost_mse(train_prediction, train_data['targets'])
        # test_loss = calculate_cost_mse(test_prediction, test_data['targets'])

        # pol_train_loss.append(train_loss)
        # pol_test_loss.append(test_loss)
        
        
        train_losses, test_losses = cross_validation(data, regressor=ols_sgd, regressor_parameters=regressor_parameters, n_pol=i, n_folds=4)
        # train_losses, test_losses = bootstrap(data, regressor=ridge, regressor_parameters=regressor_parameters, n_pol=i, n_samples=10)


        pol_train_loss.append(np.mean(train_losses))
        pol_test_loss.append(np.mean(test_losses))

        # print('Polynom order ' + str(i))
        # print('Train Mean:     %.2e' % np.mean(train_losses))
        # print('Train Variance: %.2e' % np.var(train_losses))
        # print('Test Mean:      %.2e' % np.mean(test_losses))
        # print('Test Variance:  %.2e' % np.var(test_losses))
        # print()


    print(pol_train_loss)
    plt.plot(n_pols, pol_train_loss, label='train')
    plt.plot(n_pols, pol_test_loss, label='test')
    plt.legend(loc='upper center')
    plt.show()

    exit()


if __name__ == "__main__":
    main()
    #f_range = 1.22
