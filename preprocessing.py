import pandas as pd
from functions import encode
import pymatgen.core as pmg
from visualization import box_plot, hist

data = pd.read_csv("nanotox_dataset.tsv", sep="\t", header='infer')

missing_values = data.isnull().sum()
print(f"Missing values in the data:\n{missing_values}")

print(f"Duplicates: {data.duplicated().sum()}")
data = data.drop_duplicates(keep='first')

data = data.rename(columns={'Hsf': 'standard_enthalpy_of_formation',
                     'Ec': 'conduction_band_energy',
                     'Ev': 'valence_band_energy',
                     'MeO': 'Mulliken_electronegativity',
                     'ratio': 'polarization_ratio',
                     'e': 'pauling_electronegativity',
                     'esum': 'summation_of_electronegativity',
                     'esumbyo': 'ratio_of_esum_to_Noxygen',
                     'enthalpy': 'enthalpy_of_formation_of_cation'})

box_plot(data.drop(['NPs', 'class', 'Cellline', 'Celltype'], axis=1), 'Data')

for col in data.drop(['NPs', 'class', 'Cellline', 'Celltype'], axis=1):
    hist(data, col)

data['average_electroneg'] = data['NPs'].apply(lambda x: pmg.Composition(x).average_electroneg)
data['num_atoms'] = data['NPs'].apply(lambda x: pmg.Composition(x).num_atoms)
data['total_electrons'] = data['NPs'].apply(lambda x: pmg.Composition(x).total_electrons)
data['to_weight'] = data['NPs'].apply(lambda x: pmg.Composition(x).to_weight_dict[pmg.Composition(x).elements[0].name])
data['metal_atomic_mass'] = data['NPs'].apply(lambda x: pmg.Composition(x).elements[0].atomic_mass)
data['Z'] = data['NPs'].apply(lambda x: pmg.Composition(x).elements[0].Z)
data['atomic_radius'] = data['NPs'].apply(lambda x: pmg.Composition(x).elements[0].atomic_radius)
data['atomic_radius_calculated'] = data['NPs'].apply(lambda x: pmg.Composition(x).elements[0].atomic_radius_calculated)
data['van_der_waals_radius'] = data['NPs'].apply(lambda x: pmg.Composition(x).elements[0].van_der_waals_radius)
data['electrical_resistivity'] = data['NPs'].apply(lambda x: pmg.Composition(x).elements[0].electrical_resistivity)
data['ionic_radii'] = data['NPs'].apply(lambda x: pmg.Composition(x).elements[0].ionic_radii[int(pmg.Composition(x).oxi_state_guesses()[0][pmg.Composition(x).elements[0].name])])
data['group'] = data['NPs'].apply(lambda x: pmg.Composition(x).elements[0].group)
data['ionization_energy'] = data['NPs'].apply(lambda x: pmg.Composition(x).elements[0].ionization_energy)
data['electron_affinity'] = data['NPs'].apply(lambda x: pmg.Composition(x).elements[0].electron_affinity)
data['boiling_point'] = data['NPs'].apply(lambda x: pmg.Composition(x).elements[0].boiling_point)
data['brinell_hardness'] = data['NPs'].apply(lambda x: pmg.Composition(x).elements[0].brinell_hardness)
data['coefficient_of_linear_thermal_expansion'] = data['NPs'].apply(lambda x: pmg.Composition(x).elements[0].coefficient_of_linear_thermal_expansion)
data['density_of_solid'] = data['NPs'].apply(lambda x: pmg.Composition(x).elements[0].density_of_solid)
data['liquid_range'] = data['NPs'].apply(lambda x: pmg.Composition(x).elements[0].liquid_range)
data['melting_point'] = data['NPs'].apply(lambda x: pmg.Composition(x).elements[0].melting_point)
print(data.columns)
data_normalized = data.copy()

data_normalized['viability'] = data_normalized['viability'].map(lambda x: x if x >= 0 else 0)

print(encode(['Cellline', 'Celltype', 'class'], data, data_normalized, encoded=None), sep='\n')

df_independent_variables = pd.DataFrame(data_normalized.drop(['NPs', 'class', 'viability'], axis=1))