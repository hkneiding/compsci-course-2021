import unittest
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from parameterized import parameterized

from src.tools import get_train_test_split, shuffle
from src.franke_function import franke_function, get_xy_grid_data
from src.regressors import regressor
from src.enums.regressor_type import RegressorType

class TestOls(unittest.TestCase):

    def setUp(self) -> None:
        # set random seed for reproducibility
        np.random.seed(1)
    
        return super().setUp()

    @parameterized.expand([

        [ {'fit_intercept': True}, 1 ], 
        [ {'fit_intercept': True}, 2 ], 
        [ {'fit_intercept': True}, 3 ], 
        [ {'fit_intercept': True}, 4 ], 
        [ {'fit_intercept': True}, 5 ]

    ])
    def test_ols(self, regressor_parameters, n_pol):

        # build data
        x, y = get_xy_grid_data(0, 1, 10)
        data = { 'inputs': [x.flatten(), y.flatten()], 'targets': franke_function(x, y, noise_std=0).flatten()}
        # shuffle
        data = shuffle(data)
        # split into train and test data
        train_data, test_data = get_train_test_split(data, 0.8)

        # get prediction from own implementation
        train_pred_own, test_pred_own = regressor(RegressorType.OLS, regressor_parameters, train_data, test_data, n_pol)

        # get reference prediction from sklearn
        X = np.array(train_data['inputs']).T
        vector = train_data['targets']
        predict= np.array(test_data['inputs']).T

        poly = PolynomialFeatures(degree=n_pol)
        X_ = poly.fit_transform(X)
        predict_ = poly.fit_transform(predict)

        clf = linear_model.LinearRegression()
        clf.fit(X_, vector)
    
        test_pred_skl = clf.predict(predict_)

        # check equal
        for i in range(len(test_pred_skl)):
            self.assertAlmostEqual(test_pred_skl[i], test_pred_own[i])

    @parameterized.expand([

        [ { 'fit_intercept': False, 'alpha': 0.5 }, 1 ],
        [ { 'fit_intercept': False, 'alpha': 0.5 }, 2 ],
        [ { 'fit_intercept': False, 'alpha': 0.5 }, 3 ],
        [ { 'fit_intercept': False, 'alpha': 0.5 }, 4 ],
        [ { 'fit_intercept': False, 'alpha': 0.5 }, 5 ],

        [ { 'fit_intercept': True, 'alpha': 0.5 }, 1 ],
        [ { 'fit_intercept': True, 'alpha': 0.5 }, 2 ],
        [ { 'fit_intercept': True, 'alpha': 0.5 }, 3 ],
        [ { 'fit_intercept': True, 'alpha': 0.5 }, 4 ],
        [ { 'fit_intercept': True, 'alpha': 0.5 }, 5 ]

    ])
    def test_ridge(self, regressor_parameters, n_pol):

        # build data
        x, y = get_xy_grid_data(0, 1, 10)
        data = { 'inputs': [x.flatten(), y.flatten()], 'targets': franke_function(x, y, noise_std=0).flatten()}
        # shuffle
        data = shuffle(data)
        # split into train and test data
        train_data, test_data = get_train_test_split(data, 0.8)

        # get prediction from own implementation
        train_pred_own, test_pred_own = regressor(RegressorType.RIDGE, regressor_parameters, train_data, test_data, n_pol)

        # get reference prediction from sklearn
        X = np.array(train_data['inputs']).T
        vector = train_data['targets']
        predict= np.array(test_data['inputs']).T

        poly = PolynomialFeatures(degree=n_pol)
        X_ = poly.fit_transform(X)
        predict_ = poly.fit_transform(predict)

        clf = linear_model.Ridge(alpha=regressor_parameters['alpha'], fit_intercept=regressor_parameters['fit_intercept'])
        clf.fit(X_, vector)
    
        test_pred_skl = clf.predict(predict_)

        # check equal
        for i in range(len(test_pred_skl)):
            self.assertAlmostEqual(test_pred_skl[i], test_pred_own[i])

    @parameterized.expand([

        [ { 'fit_intercept': False, 'alpha': 0.5 }, 2 ],
        [ { 'fit_intercept': False, 'alpha': 0.5 }, 3 ],
        [ { 'fit_intercept': False, 'alpha': 0.5 }, 4 ],
        [ { 'fit_intercept': False, 'alpha': 0.5 }, 5 ],

        [ { 'fit_intercept': False, 'alpha': 0.1 }, 3 ],
        [ { 'fit_intercept': False, 'alpha': 0.2 }, 3 ],
        [ { 'fit_intercept': False, 'alpha': 0.3 }, 3 ],
        [ { 'fit_intercept': False, 'alpha': 0.4 }, 3 ],

        [ { 'fit_intercept': True, 'alpha': 0.5 }, 2 ],
        [ { 'fit_intercept': True, 'alpha': 0.5 }, 3 ],
        [ { 'fit_intercept': True, 'alpha': 0.5 }, 4 ],
        [ { 'fit_intercept': True, 'alpha': 0.5 }, 5 ]

    ])
    def test_lasso(self, regressor_parameters, n_pol):

        # build data
        x, y = get_xy_grid_data(0, 1, 10)
        data = { 'inputs': [x.flatten(), y.flatten()], 'targets': franke_function(x, y, noise_std=0).flatten()}
        # shuffle
        data = shuffle(data)
        # split into train and test data
        train_data, test_data = get_train_test_split(data, 0.8)

        # get prediction from own implementation
        train_pred_own, test_pred_own = regressor(RegressorType.LASSO, regressor_parameters, train_data, test_data, n_pol)

        # get reference prediction from sklearn
        X = np.array(train_data['inputs']).T
        vector = train_data['targets']
        predict= np.array(test_data['inputs']).T

        poly = PolynomialFeatures(degree=n_pol)
        X_ = poly.fit_transform(X)
        predict_ = poly.fit_transform(predict)

        clf = linear_model.Lasso(alpha=regressor_parameters['alpha'], fit_intercept=regressor_parameters['fit_intercept'])
        clf.fit(X_, vector)
    
        test_pred_skl = clf.predict(predict_)

        # check equal
        for i in range(len(test_pred_skl)):
            self.assertAlmostEqual(test_pred_skl[i], test_pred_own[i], places=5)