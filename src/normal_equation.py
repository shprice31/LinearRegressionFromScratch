import pandas as pd
import numpy as np

#make your own linear regression class

def add_intercept_column(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()

    X.insert(0, 'Intercept', 1)

    return X

def fit_using_normal_equation(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    
    _validate_normal_equation_inputs(X, y)

    #reshape
    X = np.array(X) #type: ignore
    y = np.array(y) #type: ignore

    X_t = X.T
    beta_hat = np.linalg.solve(X_t @ X, X_t @ y)

    return pd.Series(beta_hat)
def _validate_normal_equation_inputs(X: pd.DataFrame, y: pd.Series):
    
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


def predict_using_lin_reg():
    pass

def calculate_residuals():
    pass

def calculate_rmse():
    pass

def calculate_mae():
    pass
