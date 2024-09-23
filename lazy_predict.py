from lazypredict.Supervised import LazyRegressor, LazyClassifier
from train_val_test import *
import pandas as pd

pd.set_option('display.max_columns', None)


def LazyReg(x, y):
    X_train, X_val, X_test, y_train, y_val, y_test = split(x, y, 0.2, 0.01)
    reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None, random_state=42)
    models, predictions = reg.fit(X_train, X_test, y_train, y_test)
    return models


def LazyCla(x, y):
    X_train, X_val, X_test, y_train, y_val, y_test = split(x, y, 0.2, 0.01)
    reg = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None, random_state=42)
    models, predictions = reg.fit(X_train, X_test, y_train, y_test)
    return models

print(LazyReg(x, y_viability))
print(LazyCla(x, y_class))