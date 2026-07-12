import unittest
import pandas as pd
import numpy as np
from src.normal_equation import add_intercept_column
from src.data import load_california_housing_data

class TestAddIntercept(unittest.TestCase):
    def setUp(self):
        self.cali_df_split = load_california_housing_data()
        self.X_w_int = add_intercept_column(self.cali_df_split.X)

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
        """
        The dataframe past the 0th column is unchanged from the original 
        """
        pd.testing.assert_frame_equal(self.cali_df_split.X, self.X_w_int.iloc[:, 1:])

    def test_intercept_col_is_expected_length(self):
        """
        The new intercept column is the same length as the rest of the dataframe
        """
        self.assertEqual(len(self.X_w_int['Intercept']), len(self.cali_df_split.X.iloc[:, 0]))


class TestNormalEquation(unittest.TestCase):
    def setUp(self):
        pass

    def test_placeholder(self):
        self.skipTest("tbd")

if __name__ == '__main__':
    unittest.main()