import unittest
import pandas as pd
import pandas.testing as pdt
import numpy as np
from src.normal_equation import add_intercept_column, fit_using_normal_equation, _validate_normal_equation_inputs, predict_using_lin_reg
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
        self.cali_df_split = load_california_housing_data()
        self.X_w_int = add_intercept_column(self.cali_df_split.X)
        self.beta_hat = fit_using_normal_equation(self.X_w_int, self.cali_df_split.y)

    def test_beta_hat_same_length_as_num_cols(self):
        """
        Beta_hat has 1 coefficient for each feature in the design matrix
        """
        self.assertEqual(len(self.beta_hat), len(self.X_w_int.columns))

    def test_coefficients_are_expected(self):
        """
        Small toy example to verify coefficient outputs are expected
        """
        X = pd.DataFrame([[1, 2], [1, 2], [1, 0]])
        y = pd.Series([3, 5, 2])

        beta_hat = fit_using_normal_equation(X, y)
        expected_result = pd.Series([2., 1.])

        pdt.assert_series_equal(beta_hat, expected_result)

    def test_beta_hat_returned_as_column_vector(self):
        self.assertEqual(self.beta_hat.shape[1], 1)

    #test _validate_normal_equation_inputs()
    def test_different_Xy_sizes_raises_value_error(self):
        X = pd.DataFrame([[1, 2], [1, 2], [1,0]])
        y = pd.Series([3, 5, 2, 1])

        with self.assertRaisesRegex(ValueError, "X and y have different lengths"):
            beta_hat = fit_using_normal_equation(X, y)

    def test_missing_X_values_raises_value_error(self):
        X = pd.DataFrame([[1, 2], [1, None], [1,0]])
        y = pd.Series([3, 5, 2])

        with self.assertRaisesRegex(ValueError, "X has at least 1 missing value"):
            beta_hat = fit_using_normal_equation(X, y)

    def test_missing_y_values_raises_value_error(self):
        X = pd.DataFrame([[1, 2], [1, 2], [1,0]])
        y = pd.Series([3, 5, None])

        with self.assertRaisesRegex(ValueError, "y has at least 1 missing value"):
            beta_hat = fit_using_normal_equation(X, y)

    def test_one_col_all_zeroes_raises_value_error(self):
        X = pd.DataFrame([[0], [0], [0]])
        y = pd.Series([3, 5, 2])

        with self.assertRaisesRegex(ValueError, "If X has a single column, it must have at least 1 non-zero value"):
            beta_hat = fit_using_normal_equation(X, y)

class TestPredictUsingLinReg(unittest.TestCase):
    def setUp(self):
        pass


    def test_pred_shape_correct(self):
        pass

    def test_ex_pred_values_are_expected(self):
        beta_hat = pd.Series([3, 2, 5, 1]).to_frame()
        X = pd.DataFrame([[1, 2, 3, 5],
                          [1, 3, 1, 7],
                          [1, 5, 2, 1],
                          [1, 10, 0, 2]])

        preds_true = pd.DataFrame([[27],
                                   [21],
                                   [24],
                                   [25]])
        
        preds = predict_using_lin_reg(X, beta_hat)

        pd.testing.assert_frame_equal(preds, preds_true)
    
    def test_ex_pred_shape_is_expected(self):
        beta_hat = pd.Series([3, 2, 5, 1]).to_frame()
        X = pd.DataFrame([[1, 2, 3, 5],
                               [1, 3, 1, 7],
                               [1, 5, 2, 1],
                               [1, 10, 0, 2]])
        
        preds = predict_using_lin_reg(X, beta_hat)

        self.assertEqual(preds.shape[0], X.shape[0]) #same number of rows as design matrix
        self.assertEqual(preds.shape[1], 1) #1 column

    #test _validate_predict_using_lin_reg_inputs()
    def test_misaligned_beta_hat_X_shape_raises_value_error(self):
        pass

    def test_beta_hat_0_d_shape_raises_value_error(self):
        pass


if __name__ == '__main__':
    unittest.main()