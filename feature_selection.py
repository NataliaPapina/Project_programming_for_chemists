from preprocessing import data_normalized, df_independent_variables
from visualization import box_plot, heatmap

drop = ['enthalpy_of_formation_of_cation', 'polarization_ratio', 'ratio_of_esum_to_Noxygen',
        'average_electroneg', 'standard_enthalpy_of_formation', 'pauling_electronegativity',
        'num_atoms', 'to_weight', 'summation_of_electronegativity', 'metal_atomic_mass',
        'atomic_radius_calculated', 'Z', 'group', 'ionization_energy', 'Mulliken_electronegativity',
        'electrical_resistivity', 'ionic_radii', 'boiling_point', 'melting_point',
        'coefficient_of_linear_thermal_expansion', 'brinell_hardness', 'atomic_radius', 'density_of_solid',
        'liquid_range', 'ox', 'NOxygen', 'electron_affinity', 'MW', 'van_der_waals_radius', 'total_electrons', 'NMetal']


data_normalized = data_normalized.drop(drop, axis=1)
df_independent_variables = df_independent_variables.drop(drop, axis=1)


box_plot(data_normalized.drop(['NPs'], axis=1), 'after drop')
heatmap(df_independent_variables.drop(['Cellline', 'Celltype'], axis=1), text=True)