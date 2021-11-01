import unittest
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from parameterized import parameterized

from src.tools import get_train_test_split, shuffle
from src.franke_function import franke_function, get_xy_grid_data
from src.regressors import ols, ridge


class TestOls(unittest.TestCase):

    def setUp(self) -> None:
        # set random seed for reproducibility
        np.random.seed(11111)
    
        return super().setUp()

    @parameterized.expand([

        [ 1 ], [ 2 ], [ 3 ], [ 4 ], [ 5 ]

    ])
    def test_ols(self, n_pol):

        # build data
        x, y = get_xy_grid_data(0, 1, 10)
        data = { 'inputs': [x.flatten(), y.flatten()], 'targets': franke_function(x, y, noise_std=0).flatten()}
        # shuffle
        data = shuffle(data)
        # split into train and test data
        train_data, test_data = get_train_test_split(data, 0.8)

        # get prediction from own implementation
        train_pred_own, test_pred_own = ols(train_data, test_data, n_pol, fit_intercept=True)

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

        [ 1, 0.5, True ], [ 2, 0.5, True ], [ 3, 0.5, True ], [ 4, 0.5, True ], [ 5, 0.5, True ],
        [ 1, 0.5, False ], [ 2, 0.5, False ], [ 3, 0.5, False ], [ 4, 0.5, False ], [ 5, 0.5, False ]

    ])
    def test_ridge(self, n_pol, alpha, fit_intercept):

        # build data
        x, y = get_xy_grid_data(0, 1, 10)
        data = { 'inputs': [x.flatten(), y.flatten()], 'targets': franke_function(x, y, noise_std=0).flatten()}
        # shuffle
        data = shuffle(data)
        # split into train and test data
        train_data, test_data = get_train_test_split(data, 0.8)

        # get prediction from own implementation
        train_pred_own, test_pred_own = ridge(train_data, test_data, n_pol, alpha=alpha, fit_intercept=fit_intercept)

        # get reference prediction from sklearn
        X = np.array(train_data['inputs']).T
        vector = train_data['targets']
        predict= np.array(test_data['inputs']).T

        poly = PolynomialFeatures(degree=n_pol)
        X_ = poly.fit_transform(X)
        predict_ = poly.fit_transform(predict)

        clf = linear_model.Ridge(alpha=alpha, fit_intercept=fit_intercept)
        clf.fit(X_, vector)
    
        test_pred_skl = clf.predict(predict_)

        # check equal
        for i in range(len(test_pred_skl)):
            self.assertAlmostEqual(test_pred_skl[i], test_pred_own[i])
