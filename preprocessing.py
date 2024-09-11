import pandas as pd
import numpy as np
from functions import normalize, encode
import pymatgen.core as pmg
from pymatgen.core import composition
from pymatgen.core import Element

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

data = data.assign(average_electroneg = [pmg.Composition(i).average_electroneg for i in data['NPs']])
data = data.assign(num_atoms = [pmg.Composition(i).num_atoms for i in data['NPs']])
data = data.assign(total_electrons = [pmg.Composition(i).total_electrons for i in data['NPs']])
data = data.assign(to_weight = [pmg.Composition(i).to_weight_dict[pmg.Composition(i).elements[0].name] for i in data['NPs']])
data = data.assign(metal_atomic_mass = [pmg.Composition(i).elements[0].atomic_mass for i in data['NPs']])
data = data.assign(Z = [pmg.Composition(i).elements[0].Z for i in data['NPs']])
data = data.assign(atomic_radius_calculated = [pmg.Composition(i).elements[0].atomic_radius_calculated for i in data['NPs']])
data = data.assign(van_der_waals_radius = [pmg.Composition(i).elements[0].van_der_waals_radius for i in data['NPs']])
data = data.assign(electrical_resistivity = [pmg.Composition(i).elements[0].electrical_resistivity for i in data['NPs']])
data = data.assign(ionic_radii = [pmg.Composition(i).elements[0].ionic_radii[int(pmg.Composition(i).oxi_state_guesses()[0][pmg.Composition(i).elements[0].name])] for i in data['NPs']])
data = data.assign(group = [pmg.Composition(i).elements[0].group for i in data['NPs']])
data = data.assign(ionization_energy = [pmg.Composition(i).elements[0].ionization_energy for i in data['NPs']])
data = data.assign(electron_affinity = [pmg.Composition(i).elements[0].electron_affinity for i in data['NPs']])

data_normalized = data.copy()

data_normalized['viability'] = data_normalized['viability'].map(lambda x: x if x >= 0 else 0)

encoded = []

encode(['Cellline', 'Celltype', 'class'], encoded, data, data_normalized)

print(*encoded, sep='\n')

variables = ['coresize', 'hydrosize', 'surfcharge', 'surfarea',
       'standard_enthalpy_of_formation', 'conduction_band_energy',
       'valence_band_energy', 'Mulliken_electronegativity', 'Cellline',
       'Celltype', 'Expotime', 'dosage', 'enthalpy_of_formation_of_cation',
       'polarization_ratio', 'pauling_electronegativity', 'summation_of_electronegativity',
       'ratio_of_esum_to_Noxygen', 'MW',
       'NMetal', 'NOxygen', 'ox', 'viability','average_electroneg', 'num_atoms',
       'total_electrons', 'to_weight', 'metal_atomic_mass', 'Z', 'atomic_radius_calculated',
       'van_der_waals_radius', 'electrical_resistivity', 'ionic_radii', 'group', 'ionization_energy', 'electron_affinity']

normalize(data_normalized, variables)

df_independent_variables = pd.DataFrame(data_normalized[['coresize', 'hydrosize', 'surfcharge', 'surfarea',
       'standard_enthalpy_of_formation', 'conduction_band_energy',
       'valence_band_energy', 'Mulliken_electronegativity', 'Cellline',
       'Celltype', 'Expotime', 'dosage', 'enthalpy_of_formation_of_cation',
       'polarization_ratio', 'pauling_electronegativity',
       'summation_of_electronegativity', 'ratio_of_esum_to_Noxygen', 'MW',
       'NMetal', 'NOxygen', 'ox', 'average_electroneg', 'num_atoms', 'total_electrons', 'to_weight', 'metal_atomic_mass', 'Z', 'atomic_radius_calculated',
 'van_der_waals_radius', 'electrical_resistivity', 'ionic_radii', 'group', 'ionization_energy', 'electron_affinity']])

data_normalized = data_normalized.drop(['polarization_ratio', 'ox', 'total_electrons', 'atomic_radius_calculated',
                                        'Z', 'num_atoms', 'NMetal', 'enthalpy_of_formation_of_cation', 'to_weight',
                                        'ionization_energy', 'group', 'standard_enthalpy_of_formation', 'ratio_of_esum_to_Noxygen',
                                        'summation_of_electronegativity', 'ionic_radii'], axis=1)

df_independent_variables = df_independent_variables.drop(['polarization_ratio', 'ox', 'total_electrons', 'atomic_radius_calculated',
                                        'Z', 'num_atoms', 'NMetal', 'enthalpy_of_formation_of_cation', 'to_weight',
                                        'ionization_energy', 'group', 'standard_enthalpy_of_formation', 'ratio_of_esum_to_Noxygen',
                                        'summation_of_electronegativity', 'ionic_radii'], axis=1)

cor = data_normalized.corr(numeric_only=True)
high_cor = []
[[high_cor.append(sorted([i, j])) for i in cor.columns if (cor.loc[i,j] > 0.9 or cor.loc[i,j] < -0.9) and i != j] for j in cor.columns]
print(*sorted(high_cor), sep='\n')