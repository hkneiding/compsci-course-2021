import numpy as np

from src.tools import bootstrap, cross_validation, get_train_test_split, shuffle
from src.franke_function import franke_function, get_xy_grid_data
from src.regressors import lasso, ols, ridge


def main():

    # get grid points
    x, y = get_xy_grid_data(0, 1, 10)

    # full data object
    data = { 'inputs': [x.flatten(), y.flatten()], 'targets': franke_function(x, y, noise_std=0.1).flatten()}
    data = shuffle(data)
    train_data, test_data = get_train_test_split(data, 0.8)

    for i in range(1, 6, 1):

        # alpha is only needed for LASSO or ridge where you need additional hyperparameter to 
        # control how much the magnitude of the parameter vector should be penalised
        regressor_parameters = { 'fit_intercept': True, 'alpha': 0.5 }

        train_losses, test_losses = cross_validation(data, regressor=ridge, regressor_parameters=regressor_parameters, n_pol=i, n_folds=4)
        # train_losses, test_losses = bootstrap(data, regressor=ridge, regressor_parameters=regressor_parameters, n_pol=i, n_samples=10)

        print('Polynom order ' + str(i))
        print('Train Mean:     %.2e' % np.mean(train_losses))
        print('Train Variance: %.2e' % np.var(train_losses))
        print('Test Mean:      %.2e' % np.mean(test_losses))
        print('Test Variance:  %.2e' % np.var(test_losses))
        print()

    exit()


if __name__ == "__main__":
    main()
    #f_range = 1.22
