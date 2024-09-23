from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, root_mean_squared_error
from train_val_test import x, y_viability, split
from visualization import one_box_plot, scatter
from preprocessing import data


def model_reg(x, y, name, n):
    X_train, X_val, X_test, y_train, y_val, y_test = split(x, y, 0.1, 0.1)
    MODEL = GradientBoostingRegressor(random_state=42, n_estimators=n)
    MODEL.fit(X_train, y_train)
    y_val_pred = MODEL.predict(X_val)
    r2_val = r2_score(y_val, y_val_pred)
    RMSE_val = root_mean_squared_error(y_val, y_val_pred)
    print(f'R2 val {name}: {r2_val}')
    print(f'RMSE val {name}: {RMSE_val}')
    y_test_pred = MODEL.predict(X_test)
    r2_test = r2_score(y_test, y_test_pred)
    RMSE_test = root_mean_squared_error(y_test, y_test_pred)
    scatter(y_test, y_test_pred)
    print(f'R2 test {name}: {r2_test}')
    print(f'RMSE test {name}: {RMSE_test}')
    y_train_pred = MODEL.predict(X_train)
    r2_train = r2_score(y_train, y_train_pred)
    RMSE_train = root_mean_squared_error(y_train, y_train_pred)
    print(f'R2 train {name}: {r2_train}')
    print(f'RMSE train {name}: {RMSE_train}')
    print(*sorted(list(zip(MODEL.feature_importances_, MODEL.feature_names_in_)), reverse=True),
          sep="\n")


model_reg(x, y_viability, 'viability', 49)

one_box_plot(data, 'dosage')
one_box_plot(data, 'surfarea')
one_box_plot(data, 'Expotime')
one_box_plot(data, 'coresize')