import numpy as np
import matplotlib.pyplot as plt

from src.tools import get_stochastic_batches, calculate_accuracy, shuffle, get_train_test_split
from src.resampling import cross_validation
from src.regressors import ridge, get_model_matrix
from src.franke_function import franke_function, get_xy_grid_data
from src.sgd import stochastic_batch_gradient_descent

from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDRegressor


def verbose_stochastic_batch_gradient_descent(model_matrix, targets, parameters, derivative, loss_function, alpha=0, batch_size=32, learning_rate=0.001, momentum=0.0, max_iterations=200, convergence=10e-10, scale_learning_rate=False, tau=200):
    step = 0
    loss_data_iteration = []
    loss_data_epoch = []
    accuracy_epoch = []
    learning_rate_zero = learning_rate
    learning_rate_stop = learning_rate * 0.01
    for i in range(max_iterations):
        model_matrix_batches, targets_batches = get_stochastic_batches(model_matrix, targets, batch_size)
        if scale_learning_rate and i <= int(tau):
            learning_rate = (1.0 - float(i)/tau)*learning_rate_zero + learning_rate_stop*float(i)/tau
        for j in range(len(targets_batches)):
            grad = derivative(model_matrix_batches[j], targets_batches[j], parameters, alpha=alpha)
            step = learning_rate * grad + momentum * step
            parameters = parameters - step
            loss_data = loss_function(alpha, model_matrix,targets, parameters)
            loss_data_iteration.append(loss_data)

        loss_data_epoch.append(loss_data)
        accuracy_epoch.append(calculate_accuracy( model_matrix@parameters, targets ))
        if np.sum(np.abs(step)) < convergence:
            break

    return parameters, loss_data_iteration, loss_data_epoch, accuracy_epoch


def cost_derivative_ridge(X, Y, beta, alpha):
    return -2 * X.T @ (Y - (X @ beta)) + 2 * alpha * beta


def loss_function_ridge(alpha, model_matrix, targets, parameters):
    prediction = model_matrix @ parameters
    return (targets - prediction).T @ (targets - prediction) + alpha * (parameters.T @ parameters)


def find_lambda_ridge():
    # set lambdas to test
    lambdas = np.logspace(-7, 3, num=100)

    # get grid points
    x, y = get_xy_grid_data(0, 1, 60)

    # full data object
    data = {'inputs': [x.flatten(), y.flatten()], 'targets': franke_function(x, y, noise_std=0.1).flatten()}
    data = shuffle(data)
    train_data, test_data = get_train_test_split(data, 0.8)

    num_folds = 10
    lambdas_mse = []
    regressor_parameters = {
        'fit_intercept': False,
        'alpha': 0.0
    }

    for i in range(len(lambdas)):
        regressor_parameters['alpha'] = lambdas[i]
        train_losses, test_losses = cross_validation(train_data, ridge, regressor_parameters, 10, num_folds)
        lambdas_mse.append(np.mean(test_losses))

    regressor_parameters['alpha'] = 0
    train_losses, test_losses = cross_validation(train_data, ridge, regressor_parameters, 10, num_folds)
    ols_mse = np.mean(test_losses)
    print("RESULTS: ")
    print("MSE: ", lambdas_mse)
    print("MSE OLS: ", ols_mse)
    return lambdas_mse


def find_lambda_ridge_skl():

    lambdas = np.logspace(-7, 3, num=100)

    # get grid points
    x, y = get_xy_grid_data(0, 1, 60)

    # full data object
    data = {'inputs': [x.flatten(), y.flatten()], 'targets': franke_function(x, y, noise_std=0.1).flatten()}
    data = shuffle(data)
    train_data, test_data = get_train_test_split(data, 0.8)
    train_model_matrix = get_model_matrix(train_data['inputs'], 10)
    lambdas_mse = []

    for i in range(len(lambdas)):
        ridge_sk = Ridge(alpha=lambdas[i], solver='cholesky', fit_intercept=False)
        lambda_mse = cross_val_score(ridge_sk, train_model_matrix, train_data['targets'], scoring='neg_mean_squared_error', cv=10, n_jobs=-1)
        lambdas_mse.append(-1*np.mean(lambda_mse))

    ridge_sk = Ridge(alpha=0, solver='cholesky', fit_intercept=False)
    ols_mse = cross_val_score(ridge_sk, train_model_matrix, train_data['targets'], scoring='neg_mean_squared_error', cv=10, n_jobs=-1)
    ols_mse = -1 * np.mean(ols_mse)

    print("RESULTS: ")
    print("MSE: ", lambdas_mse)
    print("MSE OLS: ", ols_mse)
    return lambdas_mse


def plot_mse_lambdas():
    lambdas = np.logspace(-7, 3, num=100)
    lambdas_mse = find_lambda_ridge()
    lambdas_mse_sk = find_lambda_ridge_skl()
    fig, ax = plt.subplots()
    ax.plot(np.log10(lambdas), lambdas_mse_sk, label='scikit-learn implementation')
    ax.plot(np.log10(lambdas), lambdas_mse, label='own implementation')
    ax.set_ylabel("MSE")
    ax.set_xlabel("Regularisation parameter - log scale")
    ax.legend()
    plt.title('Regularisation parameter for Ridge regression')
    plt.show()


def search_learning_rate(alpha):
    learning_rates = [0.01, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
    batches_size = [16, 32, 64, 128, 256, 512]
    x, y = get_xy_grid_data(0, 1, 60)
    data = {'inputs': [x.flatten(), y.flatten()], 'targets': franke_function(x, y, noise_std=0).flatten()}
    data = shuffle(data)
    train_data, test_data = get_train_test_split(data, 0.8)
    train_model_matrix = get_model_matrix(train_data['inputs'], 6)
    test_model = get_model_matrix(test_data['inputs'], 6)

    fig, ax = plt.subplots()
    ax.set_ylabel("loss")
    ax.set_xlabel("epoch")
    ax.set_ylim([0, 1000])
    ax.set_xlim([0, 200])

    beta = np.zeros(train_model_matrix.shape[1])

    for learning_rate in learning_rates:
        parameters, loss_data, loss_epoch, accuracy_epoch = verbose_stochastic_batch_gradient_descent(train_model_matrix,train_data['targets'], beta , cost_derivative_ridge, loss_function_ridge, batch_size=32, alpha=alpha, momentum=0.0, learning_rate=learning_rate, max_iterations=400, scale_learning_rate=False)
        prediction = train_model_matrix@parameters
        prediction_test = test_model@parameters
        mse = calculate_accuracy(prediction, train_data['targets'])
        mse_test = calculate_accuracy(prediction_test, test_data['targets'])
        print("ACCURACY: ", mse)
        print("ACCURACY TEST: ", mse_test)
        label = "Learning rate: " + "{:.6f}".format(learning_rate)
        ax.plot(range(len(loss_epoch)), loss_epoch, label=label)
        #ax.plot(range(len(accuracy_epoch)), accuracy_epoch, label=label)

    ax.legend()
    plt.show()

    fig, ax = plt.subplots()
    ax.set_ylabel("loss")
    ax.set_xlabel("epoch")
    ax.set_ylim([0, 1000])
    ax.set_xlim([0, 200])
    for batch_size in batches_size:
        parameters, loss_data, loss_epoch, accuracy_epoch = verbose_stochastic_batch_gradient_descent(train_model_matrix,train_data['targets'], beta , cost_derivative_ridge, loss_function_ridge, batch_size=batch_size, alpha=alpha, momentum=0.0, learning_rate=0.001, max_iterations=400, scale_learning_rate=False)
        prediction = train_model_matrix@parameters
        prediction_test = test_model@parameters
        mse = calculate_accuracy(prediction, train_data['targets'])
        mse_test = calculate_accuracy(prediction_test, test_data['targets'])
        print("ACCURACY: ", mse)
        print("ACCURACY TEST: ", mse_test)
        label = "Batch size: " + str(batch_size)
        ax.plot(range(len(loss_epoch)), loss_epoch, label=label)
        #ax.plot(range(len(accuracy_epoch)), accuracy_epoch, label=label)

    ax.legend()
    plt.show()


def plot_grid(data, lambdas, learning_rates, title):
    for i in range(len(learning_rates)):
        learning_rates[i] = '{:0.6f}'.format(learning_rates[i])
    for i in range(len(lambdas)):
        lambdas[i] = '{:0.7f}'.format(lambdas[i])

    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(data, interpolation='nearest', cmap=plt.cm.hot)
    plt.xlabel('regularization parameter')
    plt.ylabel('learning rate')
    plt.colorbar()
    plt.yticks(np.arange(len(learning_rates)), learning_rates)
    plt.xticks(np.arange(len(lambdas)), lambdas, rotation=45)
    plt.title(title)
    plt.show()


def grid_search_learning_rate_regularization():
    learning_rates = np.linspace(0.000001, 0.001, num=10)
    lambdas = np.logspace(-6, 2, num=9)
    combinations = np.array(np.meshgrid(learning_rates, lambdas)).T.reshape(-1, 2)

    x, y = get_xy_grid_data(0, 1, 60)
    data = {'inputs': [x.flatten(), y.flatten()], 'targets': franke_function(x, y, noise_std=0).flatten()}
    data = shuffle(data)
    train_data, test_data = get_train_test_split(data, 0.8)
    train_model_matrix = get_model_matrix(train_data['inputs'], 6)
    test_model_matrix = get_model_matrix(test_data['inputs'], 6)
    beta = np.zeros(train_model_matrix.shape[1])
    data_to_plot = []

    for combination in combinations:
        parameters = stochastic_batch_gradient_descent(train_model_matrix,train_data['targets'], beta, cost_derivative_ridge, batch_size=64, alpha=combination[1], learning_rate=combination[0], max_iterations=400)
        accuracy_train = calculate_accuracy(train_model_matrix@parameters,train_data['targets'])
        accuracy_test = calculate_accuracy(test_model_matrix@parameters, test_data['targets'])
        data_to_plot.append([combination[0], combination[1], accuracy_train, accuracy_test])

    # grid test
    accuracy_test = np.array(data_to_plot)
    accuracy_test = accuracy_test[:, 3]
    accuracy_test = accuracy_test.reshape(len(learning_rates), len(lambdas))
    plot_grid(accuracy_test, lambdas, learning_rates, 'ACCURACY TEST')

    # grid train
    accuracy_train = np.array(data_to_plot)
    accuracy_train = accuracy_train[:, 2]
    accuracy_train = accuracy_train.reshape(len(learning_rates), len(lambdas))
    plot_grid(accuracy_train, lambdas, learning_rates, 'ACCURACY TRAIN')


def sklearn_grid_search():

    x, y = get_xy_grid_data(0, 1, 60)
    data = {'inputs': [x.flatten(), y.flatten()], 'targets': franke_function(x, y, noise_std=0).flatten()}
    data = shuffle(data)
    train_data, test_data = get_train_test_split(data, 0.8)
    train_model_matrix = get_model_matrix(train_data['inputs'], 6)
    test_model_matrix = get_model_matrix(test_data['inputs'], 6)

    learning_rates = np.linspace(0.000001, 0.001, num=10)
    lambdas = np.logspace(-6, 2, num=9)
    combinations = np.array(np.meshgrid(learning_rates, lambdas)).T.reshape(-1, 2)
    data_to_plot = []

    for combination in combinations:
        model = SGDRegressor(loss='squared_error', alpha=combination[1], max_iter=400, eta0=combination[0], learning_rate='constant', random_state=42, tol=10e-10, fit_intercept=False)
        model.fit(train_model_matrix, train_data['targets'])
        prediction = model.predict(test_model_matrix)
        accuracy_train = calculate_accuracy(model.predict(train_model_matrix), train_data['targets'])
        accuracy_test = calculate_accuracy(prediction, test_data['targets'])
        data_to_plot.append([combination[0], combination[1], accuracy_train, accuracy_test])

    accuracy_test = np.array(data_to_plot)
    accuracy_test = accuracy_test[:, 3]
    accuracy_test = accuracy_test.reshape(len(learning_rates), len(lambdas))
    plot_grid(accuracy_test, lambdas, learning_rates, 'ACCURACY TEST')

    accuracy_train = np.array(data_to_plot)
    accuracy_train = accuracy_train[:,2]
    accuracy_train = accuracy_train.reshape(len(learning_rates),len(lambdas))
    plot_grid(accuracy_train, lambdas, learning_rates, 'ACCURACY TRAIN')


def plot_curves(xlim, ylim, xlabel, ylabel, y_array, labels):
    fig, ax = plt.subplots()
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_ylim([0, ylim])
    ax.set_xlim([1, xlim])

    for i in range(len(y_array)):
        ax.plot(range(xlim), y_array[i], label=labels[i])
    ax.legend()
    plt.show()


def sklearn_analysis_sgd(type='ols'):
    epochs = range(1, 201)
    learning_rates = [0.01, 0.001, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
    alpha = 0
    loss = 'squared_error'

    x, y = get_xy_grid_data(0, 1, 60)
    data = {'inputs': [x.flatten(), y.flatten()], 'targets': franke_function(x, y, noise_std=0).flatten()}
    data = shuffle(data)
    train_data, test_data = get_train_test_split(data, 0.8)
    train_model_matrix = get_model_matrix(train_data['inputs'], 6)

    if type == "ridge":
        alpha = 0.1

    y_plots = []
    labels = []
    for learning_rate in learning_rates:
        accuracy = []
        for epoch in epochs:
            model = SGDRegressor(loss=loss, alpha=alpha, max_iter=epoch, eta0=learning_rate, learning_rate='constant', random_state=42, tol=10e-10, fit_intercept=True)
            model.fit(train_model_matrix, train_data['targets'])
            prediction = model.predict(train_model_matrix)
            accuracy.append(calculate_accuracy(prediction, train_data['targets']))
        y_plots.append(accuracy)
        labels.append("Learning rate: " + "{:.6f}".format(learning_rate))
        print("Learning rate: ", learning_rate)
    plot_curves(200, 1, 'epoch', 'accuracy', y_plots, labels)
