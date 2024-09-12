import pandas as pd
from functions import normalize, encode
import pymatgen.core as pmg


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

data['average_electroneg'] = data['NPs'].apply(lambda x: pmg.Composition(x).average_electroneg)
data['num_atoms'] = data['NPs'].apply(lambda x: pmg.Composition(x).num_atoms)
data['total_electrons'] = data['NPs'].apply(lambda x: pmg.Composition(x).total_electrons)
data['to_weight'] = data['NPs'].apply(lambda x: pmg.Composition(x).to_weight_dict[pmg.Composition(x).elements[0].name])
data['metal_atomic_mass'] = data['NPs'].apply(lambda x: pmg.Composition(x).elements[0].atomic_mass)
data['Z'] = data['NPs'].apply(lambda x: pmg.Composition(x).elements[0].Z)
data['atomic_radius_calculated'] = data['NPs'].apply(lambda x: pmg.Composition(x).elements[0].atomic_radius_calculated)
data['van_der_waals_radius'] = data['NPs'].apply(lambda x: pmg.Composition(x).elements[0].van_der_waals_radius)
data['electrical_resistivity'] = data['NPs'].apply(lambda x: pmg.Composition(x).elements[0].electrical_resistivity)
data['ionic_radii'] = data['NPs'].apply(lambda x: pmg.Composition(x).elements[0].ionic_radii[int(pmg.Composition(x).oxi_state_guesses()[0][pmg.Composition(x).elements[0].name])])
data['group'] = data['NPs'].apply(lambda x: pmg.Composition(x).elements[0].group)
data['ionization_energy'] = data['NPs'].apply(lambda x: pmg.Composition(x).elements[0].ionization_energy)
data['electron_affinity'] = data['NPs'].apply(lambda x: pmg.Composition(x).elements[0].electron_affinity)

data_normalized = data.copy()

data_normalized['viability'] = data_normalized['viability'].map(lambda x: x if x >= 0 else 0)

encoded = []

encode(['Cellline', 'Celltype', 'class'], encoded, data, data_normalized)

#print(*encoded, sep='\n')

variables = ['coresize', 'hydrosize', 'surfcharge', 'surfarea', 'standard_enthalpy_of_formation',
             'conduction_band_energy', 'valence_band_energy', 'Mulliken_electronegativity', 'Cellline',
             'Celltype', 'Expotime', 'dosage', 'enthalpy_of_formation_of_cation', 'polarization_ratio',
             'pauling_electronegativity', 'summation_of_electronegativity', 'ratio_of_esum_to_Noxygen', 'MW',
             'NMetal', 'NOxygen', 'ox', 'viability','average_electroneg', 'num_atoms', 'total_electrons',
             'to_weight', 'metal_atomic_mass', 'Z', 'atomic_radius_calculated', 'van_der_waals_radius',
             'electrical_resistivity', 'ionic_radii', 'group', 'ionization_energy', 'electron_affinity']

normalize(data_normalized, variables)

df_independent_variables = pd.DataFrame(data_normalized[['coresize', 'hydrosize', 'surfcharge', 'surfarea',
                                                         'standard_enthalpy_of_formation', 'conduction_band_energy',
                                                         'valence_band_energy', 'Mulliken_electronegativity',
                                                         'Cellline', 'Celltype', 'Expotime', 'dosage',
                                                         'enthalpy_of_formation_of_cation', 'polarization_ratio',
                                                         'pauling_electronegativity', 'summation_of_electronegativity',
                                                         'ratio_of_esum_to_Noxygen', 'MW', 'NMetal', 'NOxygen', 'ox',
                                                         'average_electroneg', 'num_atoms', 'total_electrons',
                                                         'to_weight', 'metal_atomic_mass', 'Z',
                                                         'atomic_radius_calculated', 'van_der_waals_radius',
                                                         'electrical_resistivity', 'ionic_radii', 'group',
                                                         'ionization_energy', 'electron_affinity']])