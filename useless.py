import scipy.stats as stats
from scipy.stats import shapiro
from scipy.stats import skew
from scipy.stats import lognorm
import math
from scipy.stats import kstest
from scipy.stats import norm
import pandas as pd
#data_normalized['conduction_band_energy'] = data_normalized['conduction_band_energy'].map(lambda x: np.log(x + abs(min(data['conduction_band_energy'])) + 1))
#data_normalized['dosage'] = data_normalized['dosage'].map(lambda x: np.log10(x))
#data_normalized['Expotime'] = data_normalized['Expotime'].map(lambda x: np.log2(1+(x-min(data['Expotime']))/(max(data['Expotime'])-min(data['Expotime']))))
#data_normalized['surfcharge'] = data_normalized['surfcharge'].map(lambda x: (x-data['surfcharge'].mean())/data['surfcharge'].std())
#data_normalized['standard_enthalpy_of_formation'] = data_normalized['standard_enthalpy_of_formation'].map(lambda x: np.log(x+abs(min(data['standard_enthalpy_of_formation']))+1))
#data_normalized['valence_band_energy'] = data_normalized['valence_band_energy'].map(lambda x: np.log(x+abs(min(data['valence_band_energy']))+1))

#variables = ['coresize', 'surfarea', 'hydrosize', 'Mulliken_electronegativity', 'enthalpy_of_formation_of_cation', 'polarization_ratio', 'pauling_electronegativity', 'summation_of_electronegativity', 'ratio_of_esum_to_Noxygen', 'MW', 'NMetal']

#for var in variables:
    #data_normalized[var] = data_normalized[var].map(lambda x: np.log(x))

def grubbs_test(x): # оценка выбросов
    n = len(x)
    mean_x = np.mean(x)
    sd_x = np.std(x)
    numerator = max(abs(x-mean_x))
    g_calculated = numerator/sd_x
    t_value = stats.t.ppf(1 - 0.05 / (2 * n), n - 2)
    g_critical = ((n - 1) * np.sqrt(np.square(t_value))) / (np.sqrt(n) * np.sqrt(n - 2 + np.square(t_value)))
    if g_critical > g_calculated:
        return False
    else:
        return True


def norm_distribution_test(df, var):
    if shapiro(df[var])[1] < 0.05:      # kstest(data[var], 'norm')[1] > 0.05:
        return True
    else:
        return False


from imblearn.over_sampling import SMOTE

smote = SMOTE(
    sampling_strategy='auto',  # Стратегия выборки. 'auto' означает увеличение меньшего класса до размера большинственного.
    random_state=None,         # Зерно для генератора случайных чисел.
    k_neighbors=5,             # Количество ближайших соседей для создания синтетических примеров.
    n_jobs=-1                   # Количество ядер для параллельной работы. -1 означает использование всех доступных ядер.
)

#df_independent_variables_new, class_new = smote.fit_resample(df_independent_variables, y)