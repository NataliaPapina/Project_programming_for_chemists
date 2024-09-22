from preprocessing import data, data_normalized, df_independent_variables
from visualization import heatmap

def test_corr(df):
    cor = df.corr(method='spearman', numeric_only=True)
    high_corr = dict()
    for i in range(len(cor.columns)):
        for j in range(len(cor.columns)):
            if (cor.loc[cor.columns[i], cor.columns[j]] > 0.8 or cor.loc[cor.columns[i], cor.columns[j]] < -0.8) and \
                    cor.columns[i] != cor.columns[j]:
                high_corr[cor.columns[i]] = high_corr.get(cor.columns[i], list()) + (
                [cor.loc[cor.columns[i], cor.columns[j]], cor.columns[j]])
    print(*sorted([(len(value) // 2, key, value) for key, value in high_corr.items()]), sep='\n')


heatmap(df_independent_variables.drop(['Cellline', 'Celltype'], axis=1))
test_corr(df_independent_variables)