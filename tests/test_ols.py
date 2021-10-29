import unittest
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from parameterized import parameterized

from src.tools import get_train_test_split, shuffle
from src.franke_function import franke_function, get_xy_grid_data
from src.ols import ols

class TestOls(unittest.TestCase):

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
        train_pred_own, test_pred_own = ols(train_data, test_data, n_pol, center_matrix=True)

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

