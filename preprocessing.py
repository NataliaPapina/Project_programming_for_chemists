import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("nanotox_dataset.tsv", sep="\t", header='infer')

data = data.rename(columns={'Hsf': 'standard_enthalpy_of_formation',
                     'Ec': 'conduction_band_energy',
                     'Ev': 'valence_band_energy',
                     'MeO': 'Mulliken_electronegativity',
                     'ratio': 'polarization_ratio',
                     'e': 'pauling_electronegativity',
                     'esum': 'summation_of_electronegativity',
                     'esumbyo': 'ratio_of_esum_to_Noxygen',
                     'enthalpy': 'enthalpy_of_formation_of_cation'})

data_normalized = data.copy()

data_normalized['conduction_band_energy'] = data_normalized['conduction_band_energy'].map(lambda x: np.log(x + 6.7))
data_normalized['dosage'] = data_normalized['dosage'].map(lambda x: np.log10(x))
data_normalized['Expotime'] = data_normalized['Expotime'].map(lambda x: np.log2(x))
data_normalized['surfarea'] = data_normalized['surfarea'].apply(lambda x: (x-data['surfarea'].mean())/data['surfarea'].std())
data_normalized['standard_enthalpy_of_formation'] = data_normalized['standard_enthalpy_of_formation'].map(lambda x: np.log(x+18.345))
data_normalized['surfcharge'] = data_normalized['surfcharge'].map(lambda x: np.log(x+42.6))
data_normalized['valence_band_energy'] = data_normalized['valence_band_energy'].map(lambda x: np.log(x+10.81))

variables = ['coresize', 'hydrosize', 'Mulliken_electronegativity', 'enthalpy_of_formation_of_cation', 'polarization_ratio', 'pauling_electronegativity', 'summation_of_electronegativity', 'ratio_of_esum_to_Noxygen', 'MW', 'NMetal', 'NOxygen', 'ox']

for var in variables:
    data_normalized[var] = data_normalized[var].map(lambda x: np.log(x))

independent_variables = data[['coresize', 'hydrosize', 'surfcharge',
                              'surfarea', 'standard_enthalpy_of_formation',
                              'conduction_band_energy', 'valence_band_energy',
                              'Mulliken_electronegativity', 'Expotime',
                              'dosage', 'enthalpy_of_formation_of_cation', 'polarization_ratio',
                              'pauling_electronegativity',
                              'summation_of_electronegativity',
                              'ratio_of_esum_to_Noxygen', 'MW',
                              'NMetal', 'NOxygen', 'ox']]

independent_variables_normalized = data_normalized[['coresize', 'hydrosize', 'surfcharge',
                                                    'surfarea', 'standard_enthalpy_of_formation',
                                                    'conduction_band_energy', 'valence_band_energy',
                                                    'Mulliken_electronegativity', 'Expotime',
                                                    'dosage', 'enthalpy_of_formation_of_cation', 'polarization_ratio',
                                                    'pauling_electronegativity',
                                                    'summation_of_electronegativity',
                                                    'ratio_of_esum_to_Noxygen', 'MW',
                                                    'NMetal', 'NOxygen', 'ox']]
