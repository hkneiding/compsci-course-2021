import unittest
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from parameterized import parameterized

from src.tools import calculate_cost_mse, get_train_test_split, shuffle
from src.franke_function import franke_function, get_xy_grid_data

class TestTools(unittest.TestCase):

    @parameterized.expand([

        [ np.array([1, 6, 4, 2]), np.array([2, 5, 2, 1]), 1.75 ]

    ])
    def test_calculate_cost_mse(self, x, y, expected):

        result = calculate_cost_mse(x, y)


        self.assertAlmostEqual(result, expected)
