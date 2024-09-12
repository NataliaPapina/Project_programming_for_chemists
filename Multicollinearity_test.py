import pandas as pd
from preprocessing import data_normalized, df_independent_variables
from statsmodels.stats.outliers_influence import variance_inflation_factor

def test_corr(df):
    cor = df.corr(numeric_only=True)
    high_cor = []
    [[high_cor.append([cor.loc[i,j], i, j]) for i in cor.columns if (cor.loc[i,j] > 0.65 or cor.loc[i,j] < -0.65) and i != j] for j in cor.columns]
    print(*sorted(high_cor), sep='\n')

def vif_test(df):
    vif_data = pd.DataFrame()
    vif_data["feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i)
                          for i in range(len(df.columns))]
    print(vif_data)


test_corr(df_independent_variables)
vif_test(df_independent_variables)

drop = ['electrical_resistivity', 'ox', 'ratio_of_esum_to_Noxygen', 'polarization_ratio', 'Z', 'to_weight',
        'total_electrons', 'average_electroneg', 'NMetal', 'group', 'num_atoms', 'ionic_radii',
        'standard_enthalpy_of_formation', 'atomic_radius_calculated', 'ionization_energy',
        'summation_of_electronegativity', 'NOxygen', 'metal_atomic_mass', 'Mulliken_electronegativity',
        'enthalpy_of_formation_of_cation', 'valence_band_energy', 'MW', 'van_der_waals_radius', 'hydrosize',
        'pauling_electronegativity']

data_normalized = data_normalized.drop(drop, axis=1)

df_independent_variables = df_independent_variables.drop(drop, axis=1)

test_corr(df_independent_variables)
vif_test(df_independent_variables)