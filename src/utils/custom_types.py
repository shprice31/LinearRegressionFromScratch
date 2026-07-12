from typing import NamedTuple
import pandas as pd

class df_X_y_split(NamedTuple):
    X: pd.DataFrame
    y: pd.Series

