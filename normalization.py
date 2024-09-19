from sklearn.preprocessing import MinMaxScaler
from feature_selection import data_normalized
import pandas as pd
from visualization import box_plot


scaler = MinMaxScaler(feature_range=(0, 1)).fit(data_normalized.drop(['NPs', 'Celltype', 'class'], axis=1))
data_normalized[data_normalized.drop(['NPs', 'Celltype', 'class'], axis=1).columns] = (
    pd.DataFrame(scaler.transform(data_normalized.drop(['NPs', 'Celltype', 'class'], axis=1)),
                        columns=data_normalized.drop(['NPs', 'Celltype', 'class'], axis=1).columns))

df_independent_variables = pd.DataFrame(data_normalized.drop(['NPs', 'class', 'viability'], axis=1))
data_normalized = data_normalized.dropna()
df_independent_variables = df_independent_variables.dropna()
box_plot(data_normalized.drop(['NPs', 'Cellline', 'Celltype', 'class'], axis=1), 'after normalization')
print(data_normalized.info())