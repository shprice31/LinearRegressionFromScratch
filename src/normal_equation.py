import pandas as pd
import numpy as np

#make your own linear regression class

def add_intercept_column(X: pd.DataFrame) -> np.ndarray:
    X = np.array(X) #type:ignore

    X_w_intercept = np.insert(X, 0, 1, axis = 1)

    return X_w_intercept

def fit_using_normal_equation(X, y) -> np.ndarray:

    X = np.array(X)
    y = np.array(y)
    
    #_validate_normal_equation_inputs(X, y)

    X_t = X.T
    beta_hat = np.linalg.solve(X_t @ X, X_t @ y)

    return beta_hat

def _validate_normal_equation_inputs(X: np.ndarray, y: np.ndarray):
    
    #X has the same number of rows as y
    if len(X) != len(y):
        raise ValueError("X and y have different lengths")

    #Neither x nor y has missing values
    if X.isna().any(axis = None):
        raise ValueError("X has at least 1 missing value")
    
    if y.isna().any():
        raise ValueError("y has at least 1 missing value")

    #If x has a single columm, it has to have at least one non-zero value
    if X.shape[1] == 1:
        if not X.any().item():
            raise ValueError("If X has a single column, it must have at least 1 non-zero value")


def predict_using_lin_reg(X: pd.DataFrame, beta_hat: pd.DataFrame):
    """
    Multiply coefficients by feature matrix to generate predictions
    """

    preds = X @ beta_hat

    return preds


def calculate_residuals():
    pass

def calculate_rmse():
    pass

def calculate_mae():
    pass
