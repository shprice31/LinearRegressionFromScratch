import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing

def load_california_housing_data() -> tuple[pd.DataFrame, pd.Series]:
    cali_housing = fetch_california_housing(as_frame = True)

    X = cali_housing.data
    y = cali_housing.data

    return X, y
