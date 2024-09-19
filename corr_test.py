import pandas as pd
import numpy as np
from preprocessing import data_normalized, df_independent_variables
from Data_analysis import box_plot
from sklearn.preprocessing import MinMaxScaler


def test_corr(df):
    cor = df.corr(method='spearman', numeric_only=True)    #method='spearman'
    high_corr = dict()
    for i in range(len(cor.columns)):
        for j in range(len(cor.columns)):
            if (cor.loc[cor.columns[i], cor.columns[j]] > 0.8 or cor.loc[cor.columns[i], cor.columns[j]] < -0.8) and cor.columns[i] != cor.columns[j]:
                high_corr[cor.columns[i]] = high_corr.get(cor.columns[i], list()) + ([cor.loc[cor.columns[i], cor.columns[j]], cor.columns[j]])
    print(*sorted([(len(value)//2, key, value) for key, value in high_corr.items()]), sep='\n')


test_corr(df_independent_variables)

drop = ['enthalpy_of_formation_of_cation', 'polarization_ratio', 'ratio_of_esum_to_Noxygen',
         'average_electroneg', 'standard_enthalpy_of_formation', 'pauling_electronegativity',
        'num_atoms', 'to_weight', 'summation_of_electronegativity', 'metal_atomic_mass',
        'atomic_radius_calculated', 'Z', 'group', 'ionization_energy', 'Mulliken_electronegativity',
        'electrical_resistivity', 'ionic_radii', 'boiling_point', 'melting_point',
        'coefficient_of_linear_thermal_expansion', 'brinell_hardness', 'atomic_radius', 'density_of_solid',
        'liquid_range', 'ox', 'NOxygen', 'electron_affinity', 'MW', 'van_der_waals_radius', 'total_electrons', 'NMetal']


data_normalized = data_normalized.drop(drop, axis=1)
df_independent_variables = df_independent_variables.drop(drop, axis=1)


print('after drop')
test_corr(df_independent_variables)

#box_plot(data_normalized.drop(['NPs'], axis=1), 'after drop')
print(data_normalized.info())

scaler = MinMaxScaler(feature_range=(0, 1)).fit(data_normalized.drop(['NPs', 'Celltype', 'class'], axis=1))
data_normalized[data_normalized.drop(['NPs', 'Celltype', 'class'], axis=1).columns] = pd.DataFrame(scaler.transform(data_normalized.drop(['NPs', 'Celltype', 'class'], axis=1)),
                        columns=data_normalized.drop(['NPs', 'Celltype', 'class'], axis=1).columns)

df_independent_variables = pd.DataFrame(data_normalized.drop(['NPs', 'class', 'viability'], axis=1))
data_normalized = data_normalized.dropna()
df_independent_variables = df_independent_variables.dropna()
#box_plot(data_normalized.drop(['NPs', 'Cellline', 'Celltype', 'class'], axis=1), 'after normalization')
print(data_normalized.info())