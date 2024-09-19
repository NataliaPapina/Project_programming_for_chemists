from preprocessing import data, data_normalized, df_independent_variables
import pandas as pd
from OUTLIERS import smirnov_grubbs as grubbs
from corr_test import test_corr
from visualization import box_plot
from visualization import heatmap

indexes = set(df_independent_variables.index)
print(df_independent_variables.drop(['Cellline', 'Celltype'], axis=1).columns)

for i in df_independent_variables.drop(['Cellline', 'Celltype', 'total_electrons'], axis=1).columns:
    indexes = indexes & set(grubbs.test(df_independent_variables[i], alpha=0.05).index)

df_independent_variables = pd.DataFrame(df_independent_variables.loc[list(indexes)])
data_normalized = pd.DataFrame(data_normalized.loc[list(indexes)])
print(df_independent_variables.info())


box_plot(data_normalized.drop(['NPs', 'Cellline', 'Celltype', 'class'], axis=1), 'after normalization')

heatmap(df_independent_variables)
test_corr(df_independent_variables)