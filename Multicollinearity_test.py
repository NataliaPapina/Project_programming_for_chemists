import pandas as pd
from preprocessing import data_normalized, df_independent_variables
from statsmodels.stats.outliers_influence import variance_inflation_factor

def test_corr(df):
    cor = df.corr(method='spearman',numeric_only=True)
    high_cor = dict()
    for i in range(len(cor.columns)):
        for j in range(len(cor.columns)):
            if (cor.loc[cor.columns[i], cor.columns[j]] > 0.8 or cor.loc[cor.columns[i], cor.columns[j]] < -0.8) and cor.columns[i] != cor.columns[j]:
                high_cor[cor.columns[i]] = high_cor.get(cor.columns[i], list()) + ([cor.loc[cor.columns[i], cor.columns[j]], cor.columns[j]])
    print(*sorted([(len(value)//2, key, value) for key, value in high_cor.items()]), sep='\n')


def vif_test(df):
    vif_data = pd.DataFrame()
    vif_data["feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i)
                          for i in range(len(df.columns))]
    print(vif_data)


test_corr(df_independent_variables)
#vif_test(df_independent_variables)

drop = ['enthalpy_of_formation_of_cation', 'polarization_ratio', 'ratio_of_esum_to_Noxygen',
        'ox', 'average_electroneg', 'standard_enthalpy_of_formation', 'pauling_electronegativity',
        'NOxygen', 'num_atoms', 'to_weight', 'summation_of_electronegativity', 'metal_atomic_mass',
        'atomic_radius_calculated', 'Z', 'group', 'ionization_energy', 'Mulliken_electronegativity',
        'electrical_resistivity', 'ionic_radii', 'NMetal']

data_normalized = data_normalized.drop(drop, axis=1)

df_independent_variables = df_independent_variables.drop(drop, axis=1)

print('after drop')
test_corr(df_independent_variables)
#vif_test(df_independent_variables)