import numpy as np
from sklearn.preprocessing import LabelEncoder

def encode(lst_to_encode, lst_res, df, df_new):
    for i in lst_to_encode:
        le = LabelEncoder()
        le.fit(df[i])
        df_new[i] = le.transform(df[i])
        lst_res.append(dict(zip(df_new[i], df[i])))


def norm_log(df, var):
    if max(df[var]) >= 200:
        df[var] = df[var].map(lambda x: np.log10(x))
    else:
        df[var] = df[var].map(lambda x: np.log(x))


def normalize(df, variables):
    for var in variables:
        if min(df[var]) > -1 and max(df[var]) < 11:
            continue
        elif min(df[var]) <= 0:
            df[var] = df[var].map(lambda x: x + abs(min(df[var])) + 1)
            norm_log(df, var)
        else:
            norm_log(df, var)