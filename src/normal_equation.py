import pandas as pd
import numpy as np

#make your own linear regression class

def add_intercept_column(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()

    X.insert(0, 'Intercept', 1)

    return X

def fit_using_normal_equation():
    pass

def predict_using_lin_reg():
    pass

def calculate_residuals():
    pass

def calculate_rmse():
    pass

def calculate_mae():
    pass

