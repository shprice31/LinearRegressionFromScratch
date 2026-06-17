import unittest
import pandas as pd
import numpy as np
from ..src.normal_equation import *
from ..src.data import *

class TestAddIntercept(unittest.TestCase):
    def setUp(self):
        self.X, self.y = fetch_california_housing()
        self.X_w_int = add_intercept_column(self.X)

    def test_something(self):
        self.assertEqual(1, 1)
    
    def test_intercept_col_is_1s(self):
        """
        Every value in the Intercept column is = to 1
        """
        self.assertTrue((self.X_w_int['Intercept'] == 1).all())

    def test_0th_index_col_is_intercept(self):
        """
        The column in the 0th index is named Intercept
        """
        self.assertTrue(self.X_w_int.columns[0] == 'Intercept')

    def test_rest_of_df_unchanged(self):
        pass

    def test_intercept_col_expected_length(self):
        pass


class TestNormalEquation(unittest.TestCase):
    def setUp(self):
        pass

if __name__ == '__main__':
    unittest.main()