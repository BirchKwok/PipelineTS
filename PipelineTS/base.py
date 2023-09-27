from spinesUtils.asserts import *
import pandas as pd


class BasePipeline:
    @ParameterTypeAssert({
        'df': pd.DataFrame,
        'target_col': str,
        'date_col': str,
        'forward_windows': int,
        'random_state': int
    })
    def __init__(
            self,
            df,
            target_col,
            date_col,
            forward_windows,
            random_state=0,
    ):
        self.df = df
        self.target_col = target_col
        self.date_col = date_col
        self.forward_windows = forward_windows
        self.backward_windows = forward_windows
        self.random_state = random_state

    def fit(self):
        ...

    def predict(self):
        ...

    def partial_predict(self, n):
        ...
