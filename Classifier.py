from train_val_test import x, y_class, split
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from visualization import roc_


def model_cla(x, y, name):
    X_train, X_val, X_test, y_train, y_val, y_test = split(x, y, 0.2, 0.0001)
    MODEL = BaggingClassifier()
    MODEL.fit(X_train, y_train)
    y_test_pred = MODEL.predict(X_test)
    r2_test = accuracy_score(y_test, y_test_pred)
    ROC_AUC_test = roc_auc_score(y_test, y_test_pred, average=None)
    print(f'Accuracy test {name}: {r2_test}')
    print(f'ROC AUC test {name}: {ROC_AUC_test}')
    y_train_pred = MODEL.predict(X_train)
    r2_train = accuracy_score(y_train, y_train_pred)
    ROC_AUC_train = roc_auc_score(y_test, y_test_pred, average=None)
    print(f'Accuracy train {name}: {r2_train}')
    print(f'ROC AUC train {name}: {ROC_AUC_train}')
    fpr, tpr, thresholds = roc_curve(y_test, y_test_pred)
    roc_(fpr, tpr)


model_cla(x, y_class, 'class')