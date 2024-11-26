import numpy as np
import scipy.stats as stats
import time


def kstest1(data):
    original_result = 100000
    start = time.time()
    for x in data:
        x = x - original_result
    end = time.time()
    timee = end - start
    def ks(data):
        gaussian_params = stats.norm.fit(data)
        laplace_params = stats.laplace.fit(data)
        gaussian_ks_statistic, gaussian_ks_pvalue = stats.kstest(data, 'norm', gaussian_params)
        laplace_ks_statistic, laplace_ks_pvalue = stats.kstest(data, 'laplace', laplace_params)
        print("KS Test for Gaussian Fit - Statistic:", gaussian_ks_statistic)
        print("KS Test for Laplace Fit - Statistic:", laplace_ks_statistic)
        print("P-Value-G:", gaussian_ks_pvalue)
        print("P-Value-L:", laplace_ks_pvalue)
        if laplace_ks_statistic < gaussian_ks_statistic:
            return 0
        else:
            return 1

    ks_start = time.time()
    dect_result = ks(data)
    ks_end = time.time()
    total_time = ks_end - ks_start + timee
    print("total time for ks test:", total_time)
    return dect_result


# a simple test
with open('1.txt', 'r', encoding='utf-8') as file:
    obser_1 = [float(line.strip()) for line in file.readlines()]
data = np.array(obser_1)

if np.asarray(data).ndim > 1:
        data = np.concatenate(data)
noise_dect = kstest1(data)
print(noise_dect)

