from mechanism.laplace import laplace_b
from statisticalmethods.Likelihood import likelihood_ratio
import random
import time
import numpy as np

"""
Auditing Laplace-based DP-DBs with LR
"""


def observations(epsilon, delta, s, label):

    b = laplace_b(s, epsilon)

    if label == 0:
        # count query
        q = 0  # real query result on D_0
        f = q + round(np.random.laplace(scale=b))  # noisy query result on D_0
        return q, f
    else:
        q_prime = 1  # real query result on D_1
        f_prime = q_prime + round(np.random.laplace(scale=b))  # noisy query result on D_1
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


def transtoep(alpha, beta, delta):  # compute epsilon

    if 1 / 2 <= alpha < 1 and 0 < beta <= 1 / 2:
        b = 1 / np.log((1 - alpha) / beta)
    elif 1 / 2 <= alpha < 1 and 1 / 2 < beta < 1:
        b = 1 / np.log(4 * (1 - alpha) * (1 - beta))
    elif 0 < alpha < 1 / 2 and 0 < beta <= 1 / 2:
        b = 1 / np.log(1 / (4 * alpha * beta))
    elif 0 < alpha < 1 / 2 and 1 / 2 < beta < 1:
        b = 1 / np.log((1 - beta) / alpha)
    else:
        print("error")

    # print(b)

    point_1 = np.exp(-1 / b) / 2
    point_2 = 1 / 2
    alpha_1 = np.linspace(0, point_1, 1000000)
    alpha_2 = np.linspace(point_1, point_2, 1000000)
    alpha_3 = np.linspace(point_2, 1, 1000000)

    # trade-off function
    f_1 = 1 - np.exp(1 / b) * alpha_1
    f_2 = np.exp(-1 / b) / (4 * alpha_2)
    f_3 = np.exp(-1 / b) * (1 - alpha_3)

    # convert f to epsilon
    epsilon_11 = np.log((1 - delta - f_1) / alpha_1)
    epsilon_12 = np.log((1 - delta - f_2) / alpha_2)
    epsilon_13 = np.log((1 - delta - f_3) / alpha_3)
    epsilon_21 = np.log(1 - delta - alpha_1) - np.log(f_1)
    epsilon_22 = np.log(1 - delta - alpha_2) - np.log(f_2)
    epsilon_23 = np.log(1 - delta - alpha_3) - np.log(f_3)

    # filter valid values
    epsilon_11 = epsilon_11[epsilon_11 < 10]
    epsilon_12 = epsilon_12[epsilon_12 < 10]
    epsilon_13 = epsilon_13[epsilon_13 < 10]
    epsilon_21 = epsilon_21[epsilon_21 < 10]
    epsilon_22 = epsilon_22[epsilon_22 < 10]
    epsilon_23 = epsilon_23[epsilon_23 < 10]
    epsilon_1 = max(np.max(epsilon_11), np.max(epsilon_12), np.max(epsilon_13))
    epsilon_2 = max(np.max(epsilon_21), np.max(epsilon_22), np.max(epsilon_23))
    return max(epsilon_1, epsilon_2)


if __name__ == "__main__":
    ep = 0.8
    de = 0
    num = 2000000
    print("epsilon:", ep)
    start_time = time.time()
    tr_1, tr_2, no_1, no_2 = construct_ob(num, ep, de)

    FN, FP, TN, TP = identificationresult(tr_1, tr_2, no_1, no_2, num)
    print(FN, FP, TN, TP)

    FPR_lo, FPR_hi = fpr(FP, TN, num)
    FNR_lo, FNR_hi = fnr(TP, FN, num)

    epsilon_lo = transtoep(FPR_hi, FNR_hi, de)
    epsilon_hi = transtoep(FPR_lo, FNR_lo, de)
    interval = epsilon_hi - epsilon_lo

    end_time = time.time()
    audit_time = end_time - start_time
    print("epsilon_lo:", epsilon_lo)
    print("epsilon_hi:", epsilon_hi)
    print("interval:", interval)
    print("auditing_time:", audit_time)