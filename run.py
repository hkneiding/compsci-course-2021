import numpy as np

from src.tools import bootstrap, cross_validation, get_train_test_split, shuffle
from src.franke_function import franke_function, get_xy_grid_data
from src.regressors import ols


def main():

    # get grid points
    x, y = get_xy_grid_data(0, 1, 10)

    # full data object
    data = { 'inputs': [x.flatten(), y.flatten()], 'targets': franke_function(x, y, noise_std=0).flatten()}
    data = shuffle(data)
    train_data, test_data = get_train_test_split(data, 0.8)
    
    # train_pred, test_pred = ols(train_data, test_data, 3, center_matrix=False)
    # print(test_pred)

    for i in range(1, 6, 1):

        train_losses, test_losses = cross_validation(data, regressor=ols, n_pol=i, n_folds=4)
        # train_losses, test_losses = bootstrap(data, regressor=ols, n_pol=i, n_samples=10)

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
    #f_range = 1.22
