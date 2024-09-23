import numpy as np
import pandas as pd
from corr_test import test_corr
from functions import encode
from train_val_test import split

df = pd.DataFrame({'col1': ['a', 'b', 'c'], 'col2': ['d', 'e', 'f'], 'col3': [0, 1, 2]})
df_new = df.copy()
en_l = ['col1', 'col2']
assert set(encode(en_l, df, df_new, encoded=None)[0].keys()) == {0, 1, 2}

X_train, X_val, X_test, y_train, y_val, y_test = split([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 0.2, 0.2)
assert (len(X_train) == len(y_train) == 3) & (len(X_val) == len(X_test) == len(y_val) == len(y_test) == 1)

df = pd.DataFrame({'x1': np.array([1, 2, 3, 4]), 'x2': np.array([1, 2, 3, 4])*2, 'x3': np.array([1, 2, 3, 4])*3})
assert test_corr(df) == {'x1': [1.0, 'x2', 1.0, 'x3'], 'x2': [1.0, 'x1', 1.0, 'x3'], 'x3': [1.0, 'x1', 1.0, 'x2']}