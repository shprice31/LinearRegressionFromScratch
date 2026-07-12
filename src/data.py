import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from src.utils.custom_types import df_X_y_split

def load_california_housing_data() -> df_X_y_split:
    cali_housing = fetch_california_housing(as_frame = True)

    return df_X_y_split(
        X = cali_housing.data,
        y = cali_housing.target
    )


if __name__ == "__main__":
    cali_df_split = load_california_housing_data()
    print(cali_df_split.X.head())