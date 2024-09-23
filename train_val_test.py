from sklearn.model_selection import train_test_split
from normalization import df_independent_variables, data_normalized

x = df_independent_variables
y_viability = data_normalized['viability']
y_class = data_normalized['class']


def split(x, y, test, val):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val, random_state=1)
    return X_train, X_val, X_test, y_train, y_val, y_test