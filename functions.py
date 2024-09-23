from sklearn.preprocessing import LabelEncoder


def encode(lst_to_encode, df, df_new, encoded=None):
    if encoded is None:
        encoded = []
    for i in lst_to_encode:
        le = LabelEncoder()
        le.fit(df[i])
        df_new[i] = le.transform(df[i])
        encoded.append(dict(zip(df_new[i], df[i])))
    return encoded