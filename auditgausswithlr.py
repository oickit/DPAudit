from scipy.stats import norm
from mechanism.analytic_gauss import calibrateAnalyticGaussianMechanism
from mechanism.gaussian import gaussian_sigma
from statisticalmethods.Likelihood import likelihood_ratio
from scipy.optimize import newton
import numpy as np
import random
import time

"""
Auditing Gauss-based DP-DBs with LR
"""


def observations(epsilon, delta, s, label):
    sigma = calibrateAnalyticGaussianMechanism(epsilon, delta, s)  # analytic gaussian
    # sigma = gaussian_sigma(s, ep, de) # classical gauss

    if label == 0:
        q = 0  # real query result on D_0
        f = q + round(np.random.normal(0, sigma))  # noisy query result on D_0
        return q, f
    else:
        q_prime = 1  # real query result on D_1
        f_prime = q_prime + round(np.random.normal(0, sigma))  # noisy query result on D_1
        return q_prime, f_prime


def construct_ob(num, ep, de):
    s = 1  # sensitivity
    trueres_1 = []
    trueres_2 = []
    noisyres_1 = []
    noisyres_2 = []
    for _ in range(num):
        q1, o1 = observations(ep, de, s, 0)
        q2, o2 = observations(ep, de, s, 1)
        trueres_1.append(q1)
        trueres_2.append(q2)
        noisyres_1.append(o1)
        noisyres_2.append(o2)

    return trueres_1, trueres_2, noisyres_1, noisyres_2


def identificationresult(tr_1, tr_2, no_1, no_2, num):
    FN = FP = TN = TP = 0
    thres = [0] * num
    for i in range(num):
        # decision threshold
        thres[i] = (tr_1[i] + tr_2[i]) / 2

        if tr_1[i] <= tr_2[i]:
            if no_1[i] < thres[i]:
                TP += 1
            else:
                FN += 1
            if no_2[i] < thres[i]:
                FP += 1
            else:
                TN += 1
        else:  # tr_1[i] >= tr_2[i]
            if no_1[i] < thres[i]:
                FN += 1
            else:
                TP += 1
            if no_2[i] < thres[i]:
                TN += 1
            else:
                FP += 1
    return FN, FP, TN, TP


def fpr(fp, tn, n):
    [fp_lo, fp_hi] = likelihood_ratio(fp, n)
    [tn_lo, tn_hi] = likelihood_ratio(tn, n)
    fpr_lo = fp_lo / (fp_hi + tn_hi)
    fpr_hi = fp_hi / (fp_lo + tn_lo)
    return fpr_lo, fpr_hi


def fnr(tp, fn, n):
    [tp_lo, tp_hi] = likelihood_ratio(tp, n)
    [fn_lo, fn_hi] = likelihood_ratio(fn, n)
    fnr_lo = fn_lo / (fn_hi + tp_hi)
    fnr_hi = fn_hi / (fn_lo + tp_lo)
    return fnr_lo, fnr_hi


def transmutoeps(delta, mu):
    def equation(epsilon, delta, mu):
        return delta - (norm.cdf(-epsilon / mu + mu / 2) - np.exp(epsilon) * norm.cdf(-epsilon / mu - mu / 2))

    epsilon_initial = 0.0  # initialization
    if mu <= 0:  # unable to provide an effective lower bound
        return epsilon_initial
    epsilon = newton(equation, epsilon_initial, args=(delta, mu))
    return epsilon


if __name__ == "__main__":
    ep = 0.8
    de = 0.0000001
    num = 200000
    print("epsilon:", ep)
    start_time = time.time()
    tr_1, tr_2, no_1, no_2 = construct_ob(num, ep, de)
    end = time.time()

    FN, FP, TN, TP = identificationresult(tr_1, tr_2, no_1, no_2, num)
    # print(FN, FP, TN, TP)

    FPR_lo, FPR_hi = fpr(FP, TN, num)
    FNR_lo, FNR_hi = fnr(TP, FN, num)

    mu_lo = norm.ppf(1 - FPR_hi) - norm.ppf(FNR_hi)
    mu_hi = norm.ppf(1 - FPR_lo) - norm.ppf(FNR_lo)

    epsilon_lo = transmutoeps(de, mu_lo)
    epsilon_hi = transmutoeps(de, mu_hi)
    interval = epsilon_hi - epsilon_lo

    end_time = time.time()
    audit_time = end_time - start_time

    print("epsilon_lo:", epsilon_lo)
    print("epsilon_hi:", epsilon_hi)
    print("interval:", interval)
    print("auditing_time:", audit_time)