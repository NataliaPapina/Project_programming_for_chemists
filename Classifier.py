from train_val_test import x, y_class, split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn import set_config
from visualization import roc_


set_config(enable_metadata_routing=True)


def model_cla(x, y, name):
    X_train, X_val, X_test, y_train, y_val, y_test = split(x, y, 0.1, 0.1)
    MODEL = GaussianNB()
    MODEL.fit(X_train, y_train)
    y_val_pred = MODEL.predict(X_val)
    r2_val = accuracy_score(y_val, y_val_pred)
    ROC_AUC_val = roc_auc_score(y_val, y_val_pred, average=None)
    print(f'Accuracy val {name}: {r2_val}')
    print(f'ROC AUC val {name}: {ROC_AUC_val}')
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