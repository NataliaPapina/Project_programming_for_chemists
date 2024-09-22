from sklearn.model_selection import train_test_split
from normalization import df_independent_variables, data_normalized
from lazypredict.Supervised import LazyRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
import plotly.express as px
import plotly.graph_objects as go


def LazyReg(X_train, X_test, y_train, y_test):
    reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
    models, predictions = reg.fit(X_train, X_test, y_train, y_test)
    return models[models['R-Squared'] == max(models['R-Squared'])].index[0]


models = {'GradientBoostingRegressor': GradientBoostingRegressor, 'AdaBoostRegressor': AdaBoostRegressor,
          'ExtraTreesRegressor': ExtraTreesRegressor, 'MLPRegressor': MLPRegressor}

x = df_independent_variables
y_viability = data_normalized['viability']
y_class = data_normalized['class']
X_train, X_test, y_train, y_test = train_test_split(x, y_viability, test_size=0.2, random_state=42)


def model(x, y, name, n):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    MODEL = models[LazyReg(X_train, X_test, y_train, y_test)](random_state=42, n_estimators=n)
    MODEL.fit(X_train, y_train)
    y_test_pred = MODEL.predict(X_test)
    r2_test = r2_score(y_test, y_test_pred)
    print(f'R2 test {name}: {r2_test}')
    y_train_pred = MODEL.predict(X_train)
    r2_train = r2_score(y_train, y_train_pred)
    print(f'R2 train {name}: {r2_train}')
    print(*sorted(list(zip(MODEL.feature_importances_, MODEL.feature_names_in_)), reverse=True),
          sep="\n")


model(x, y_viability, 'viability', 44)
#model(x, y_class, 'class')