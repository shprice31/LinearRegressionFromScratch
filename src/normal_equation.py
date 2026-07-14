import pandas as pd
import numpy as np

#make your own linear regression class

def add_intercept_column(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()

    X.insert(0, 'Intercept', 1)

    return X

def fit_using_normal_equation(X: np.array, y: np.array):
    
    #verify the design matrix's 1st column is 1s 
    # if not (X.iloc[:, 0] == 1).all():
    #     raise ValueError("The design matrix must have an intercept column")

    X_t = X.T
    beta_hat = np.linalg.solve(X_t @ X, X_t @ y)

    return pd.Series(beta_hat)

def predict_using_lin_reg():
    pass

def calculate_residuals():
    pass

def calculate_rmse():
    pass

def calculate_mae():
    pass
