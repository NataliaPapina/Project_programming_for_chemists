import pandas as pd
import numpy as np
from Data_analysis import data_normalized, df_independent_variables
#from Data_analysis import box_plot



def test_corr(df):
    cor = df.corr(method='spearman', numeric_only=True)    #method='spearman'
    high_corr = dict()
    for i in range(len(cor.columns)):
        for j in range(len(cor.columns)):
            if (cor.loc[cor.columns[i], cor.columns[j]] > 0.8 or cor.loc[cor.columns[i], cor.columns[j]] < -0.8) and cor.columns[i] != cor.columns[j]:
                high_corr[cor.columns[i]] = high_corr.get(cor.columns[i], list()) + ([cor.loc[cor.columns[i], cor.columns[j]], cor.columns[j]])
    print(*sorted([(len(value)//2, key, value) for key, value in high_corr.items()]), sep='\n')