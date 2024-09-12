import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from Multicollinearity_test import df_independent_variables, data_normalized
from lazypredict.Supervised import LazyRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
import plotly.graph_objects as go

model = LinearRegression(fit_intercept=True)
x = df_independent_variables
y_viability = data_normalized['viability']

X_train_viability, X_test_viability, y_train_viability, y_test_viability = train_test_split(x, y_viability, test_size=0.2, random_state=42)
reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
models_viability, predictions_viability = reg.fit(X_train_viability, X_test_viability, y_train_viability, y_test_viability)
print(models_viability)

GBregr_viability = GradientBoostingRegressor(random_state=42, n_estimators=44)
GBregr_viability.fit(X_train_viability, y_train_viability)

y_test_pred_viability = GBregr_viability.predict(X_test_viability)
r2_test_viability = r2_score(y_test_viability, y_test_pred_viability)
print(f'R2 test viability: {r2_test_viability}')

y_train_pred_viability = GBregr_viability.predict(X_train_viability)
r2_train_viability = r2_score(y_train_viability, y_train_pred_viability)
print(f'R2 train viability: {r2_train_viability}')

#==============================================================================================================

y_class = data_normalized['class']
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(x, y_class, test_size=0.2, random_state=42)
reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
models_class, predictions_class = reg.fit(X_train_class, X_test_class, y_train_class, y_test_class)
print(models_class)

GBregr_class = GradientBoostingRegressor(random_state=42, n_estimators=44)
GBregr_class.fit(X_train_class, y_train_class)

y_test_pred_class = GBregr_class.predict(X_test_class)
r2_test_class = r2_score(y_test_class, y_test_pred_class)
print(f'R2 test class: {r2_test_class}')

y_train_pred_class = GBregr_class.predict(X_train_class)
r2_train_class = r2_score(y_train_class, y_train_pred_class)
print(f'R2 train class: {r2_train_class}')