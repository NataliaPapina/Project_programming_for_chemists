import scipy.stats as stats
from scipy.stats import shapiro
from scipy.stats import lognorm
import math
from scipy.stats import kstest
from scipy.stats import norm


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


def norm_distribution_test(data, var):
    if shapiro(data[var])[1] < 0.05 and kstest(data[var], 'norm')[1] > 0.05:
        return True
    else:
        return False

def norm_log(data, var):
    if max(data[var])>=1000:
        data_normalized[var] = data[var].map(lambda x: np.log10(x))
    elif 100<=max(data[var])<1000:
        data_normalized[var] = data[var].map(lambda x: np.log2(x))
    else:
        data_normalized[var] = data[var].map(lambda x: np.log(x))


def normalize(data, variables):
    for var in variables:
        if min(data[var]) <= 0:
            data_normalized[var] = data[var].map(lambda x: x + min(data[var]) + 1)
            norm_log(data, var)
        else:
            norm_log(data, var)

normalize(data, variables)