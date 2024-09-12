import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from Multicollinearity_test import df_independent_variables, data_normalized
import lazypredict.Supervised
from lazypredict.Supervised import LazyRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
import plotly.graph_objects as go

model = LinearRegression(fit_intercept=True)
x = df_independent_variables
y_viability = data_normalized['viability']

X_train_viability, X_test_viability, y_train_viability, y_test_viability = train_test_split(x, y_viability, test_size=0.2, random_state=42)
#reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
#models_viability, predictions_viability = reg.fit(X_train_viability, X_test_viability, y_train_viability, y_test_viability)
#print(models_viability)

GBregr = GradientBoostingRegressor(random_state=42, n_estimators=44)
GBregr.fit(X_train_viability, y_train_viability)

y_test_pred_viability = GBregr.predict(X_test_viability)
r2_test_viability = r2_score(y_test_viability, y_test_pred_viability)
print(f'R2 test: {r2_test_viability}')

y_train_pred_viability = GBregr.predict(X_train_viability)
r2_train_viability = r2_score(y_train_viability, y_train_pred_viability)
print(f'R2 train: {r2_train_viability}')

y_class = data_normalized['class']
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(x, y_class, test_size=0.2, random_state=42)
reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
models_class, predictions_class = reg.fit(X_train_class, X_test_class, y_train_class, y_test_class)
print(models_class[models_class['R-Squared'] == max(models_class['R-Squared'])])

