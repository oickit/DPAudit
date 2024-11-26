from mechanism.laplace import laplace_b
from statisticalmethods.ClopperPearson import clopper_pearson
from scipy.integrate import dblquad
from scipy.optimize import root_scalar
from scipy.stats import beta
import numpy as np
import time

"""
Auditing Laplace-based DP-DBs with BayE
"""


def observations(epsilon, delta, s, label):

    b = laplace_b(s, epsilon)

    # incorrect noisy scale
    # b = laplace_b(s, epsilon) * (5/6)

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


def bayesianmethod(FN, FP, TN, TP, ep_lo, ep_hi, gamma=0.05, tol=1e-5):
    def ep_to_lambda(ep):
        alpha = 1 / 2
        f1 = 0
        f2 = 1 - np.exp(ep) * alpha
        f3 = np.exp(-ep) * (1 - alpha)
        f = max(f1, f2, f3)
        lam = np.log(1 / (2 * f))  # f(alpha) = e^(-lam)/2
        return lam

    def joint_density(fn, fp, tn, tp, x, y):
        fpr_density = beta(fn + 0.5, tp + 0.5).pdf(y)
        fnr_density = beta(fp + 0.5, tn + 0.5).pdf(x)
        return fpr_density * fnr_density

    def y1_lower(lam, x):  # x < e^(-lam)/2
        return 1 - np.exp(lam) * x

    def y1_upper(lam, x):  # x < e^(-lam)/2
        return np.exp(lam) * (1 - x)

    def y2_lower(lam, x):  # e^(-lam)/2 <= x < 1/2
        return np.exp(-lam) / (4 * x)

    def y2_upper(lam, x):  # e^(-lam)/2 <= x < 1/2
        return 1 - np.exp(-lam) / (4 - 4 * x)

    def y3_lower(lam, x):  # x > 1/2
        return np.exp(-lam) * (1 - x)

    def y3_upper(lam, x):  # x > 1/2
        return 1 - np.exp(-lam) * x

    def pp(lam):
        po = np.exp(-lam) / 2
        if po <= 0.1:
            t1 = dblquad(lambda y, x: joint_density(fn=FN, fp=FP, tn=TN, tp=TP, x=x, y=y),
                         0, po, lambda x: y1_lower(lam=lam, x=x), lambda x: y1_upper(lam=lam, x=x))[0]
            t2 = dblquad(lambda y, x: joint_density(fn=FN, fp=FP, tn=TN, tp=TP, x=x, y=y),
                         po, 0.1, lambda x: y2_lower(lam=lam, x=x), lambda x: y2_upper(lam=lam, x=x))[0]
            t3 = 0
            step = 0.05
            for x in np.arange(0.1, 0.5, step):
                t3 += dblquad(lambda y, x: joint_density(fn=FN, fp=FP, tn=TN, tp=TP, x=x, y=y),
                              x, x + step, lambda x: y2_lower(lam=lam, x=x), lambda x: y2_upper(lam=lam, x=x))[0]
            ta = t1 + t2 + t3
        elif 0.1 < po <= 0.2:
            t1 = dblquad(lambda y, x: joint_density(fn=FN, fp=FP, tn=TN, tp=TP, x=x, y=y),
                         0, po, lambda x: y1_lower(lam=lam, x=x), lambda x: y1_upper(lam=lam, x=x))[0]
            t2 = dblquad(lambda y, x: joint_density(fn=FN, fp=FP, tn=TN, tp=TP, x=x, y=y),
                         po, 0.2, lambda x: y2_lower(lam=lam, x=x), lambda x: y2_upper(lam=lam, x=x))[0]
            t3 = dblquad(lambda y, x: joint_density(fn=FN, fp=FP, tn=TN, tp=TP, x=x, y=y),
                         0.2, 0.3, lambda x: y2_lower(lam=lam, x=x), lambda x: y2_upper(lam=lam, x=x))[0]
            t4 = dblquad(lambda y, x: joint_density(fn=FN, fp=FP, tn=TN, tp=TP, x=x, y=y),
                         0.3, 0.4, lambda x: y2_lower(lam=lam, x=x), lambda x: y2_upper(lam=lam, x=x))[0]
            t5 = dblquad(lambda y, x: joint_density(fn=FN, fp=FP, tn=TN, tp=TP, x=x, y=y),
                         0.4, 0.5, lambda x: y2_lower(lam=lam, x=x), lambda x: y2_upper(lam=lam, x=x))[0]
            ta = t1 + t2 + t3 + t4 + t5
        elif 0.2 < po <= 0.3:
            t1 = dblquad(lambda y, x: joint_density(fn=FN, fp=FP, tn=TN, tp=TP, x=x, y=y),
                         0, po, lambda x: y1_lower(lam=lam, x=x), lambda x: y1_upper(lam=lam, x=x))[0]
            t2 = dblquad(lambda y, x: joint_density(fn=FN, fp=FP, tn=TN, tp=TP, x=x, y=y),
                         po, 0.3, lambda x: y2_lower(lam=lam, x=x), lambda x: y2_upper(lam=lam, x=x))[0]
            t3 = dblquad(lambda y, x: joint_density(fn=FN, fp=FP, tn=TN, tp=TP, x=x, y=y),
                         0.3, 0.4, lambda x: y2_lower(lam=lam, x=x), lambda x: y2_upper(lam=lam, x=x))[0]
            t4 = dblquad(lambda y, x: joint_density(fn=FN, fp=FP, tn=TN, tp=TP, x=x, y=y),
                         0.4, 0.5, lambda x: y2_lower(lam=lam, x=x), lambda x: y2_upper(lam=lam, x=x))[0]
            ta = t1 + t2 + t3 + t4
        elif 0.3 < po <= 0.4:
            t1 = dblquad(lambda y, x: joint_density(fn=FN, fp=FP, tn=TN, tp=TP, x=x, y=y),
                         0.1, po, lambda x: y1_lower(lam=lam, x=x), lambda x: y1_upper(lam=lam, x=x))[0]
            t2 = dblquad(lambda y, x: joint_density(fn=FN, fp=FP, tn=TN, tp=TP, x=x, y=y),
                         po, 0.4, lambda x: y2_lower(lam=lam, x=x), lambda x: y2_upper(lam=lam, x=x))[0]
            t3 = dblquad(lambda y, x: joint_density(fn=FN, fp=FP, tn=TN, tp=TP, x=x, y=y),
                         0.4, 0.45, lambda x: y2_lower(lam=lam, x=x), lambda x: y2_upper(lam=lam, x=x))[0]
            t4 = dblquad(lambda y, x: joint_density(fn=FN, fp=FP, tn=TN, tp=TP, x=x, y=y),
                         0.45, 0.5, lambda x: y2_lower(lam=lam, x=x), lambda x: y2_upper(lam=lam, x=x))[0]
            ta = t1 + t2 + t3 + t4
        elif 0.4 < po <= 0.5:
            t1 = dblquad(lambda y, x: joint_density(fn=FN, fp=FP, tn=TN, tp=TP, x=x, y=y),
                         0.1, po, lambda x: y1_lower(lam=lam, x=x), lambda x: y1_upper(lam=lam, x=x))[0]
            t2 = dblquad(lambda y, x: joint_density(fn=FN, fp=FP, tn=TN, tp=TP, x=x, y=y),
                         po, 0.5, lambda x: y2_lower(lam=lam, x=x), lambda x: y2_upper(lam=lam, x=x))[0]
            ta = t1 + t2
        else:
            print("ERROR")
        tb = dblquad(lambda y, x: joint_density(fn=FN, fp=FP, tn=TN, tp=TP, x=x, y=y),
                     0.5, 1, lambda x: y3_lower(lam=lam, x=x), lambda x: y3_upper(lam=lam, x=x))[0]
        return ta + tb

    def pp_lo(lam):
        return pp(lam) - gamma / 2

    def pp_hi(lam):
        return pp(lam) - 1 + gamma / 2

    lam_lo = ep_to_lambda(ep_lo)
    lam_hi = ep_to_lambda(ep_hi)

    estimated_lam_lo = root_scalar(pp_lo, bracket=[lam_lo, lam_hi], xtol=tol).root
    estimated_lam_hi = root_scalar(pp_hi, bracket=[lam_lo, lam_hi], xtol=tol).root

    return estimated_lam_lo, estimated_lam_hi


def alphabeta_toep(fn, fp, tn, tp, n, de):
    def esti_ep(alpha, beta, delta):
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

    [fp_lo, fp_hi] = clopper_pearson(fp, n, alpha=0.01)
    [tn_lo, tn_hi] = clopper_pearson(tn, n, alpha=0.01)
    fpr_lo = fp_lo / (fp_hi + tn_hi)
    fpr_hi = fp_hi / (fp_lo + tn_lo)

    [tp_lo, tp_hi] = clopper_pearson(tp, n, alpha=0.01)
    [fn_lo, fn_hi] = clopper_pearson(fn, n, alpha=0.01)
    fnr_lo = fn_lo / (fn_hi + tp_hi)
    fnr_hi = fn_hi / (fn_lo + tp_lo)

    ep_lo = esti_ep(fpr_hi, fnr_hi, de)
    ep_hi = esti_ep(fpr_lo, fnr_lo, de)

    # print("ep_lo,ep_hi", ep_lo,ep_hi)

    return ep_lo, ep_hi


def lam_toep(lam, delta):
    point_1 = np.exp(-lam) / 2
    point_2 = 1 / 2
    alpha_1 = np.linspace(0, point_1, 1000000)
    alpha_2 = np.linspace(point_1, point_2, 1000000)
    alpha_3 = np.linspace(point_2, 1, 1000000)

    # trade-off function
    f_1 = 1 - np.exp(lam) * alpha_1
    f_2 = np.exp(-lam) / (4 * alpha_2)
    f_3 = np.exp(-lam) * (1 - alpha_3)

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
    ep = 1.4
    de = 0
    num = 20000
    print(ep)
    start_time = time.time()
    tr_1, tr_2, no_1, no_2 = construct_ob(num, ep, de)

    FN, FP, TN, TP = identificationresult(tr_1, tr_2, no_1, no_2, num)
    print(FN, FP, TN, TP)

    ep_lo, ep_hi = alphabeta_toep(FN, FP, TN, TP, num, de)
    # print("ep_lo, ep_hi:", ep_lo, ep_hi)

    lambda_lo, lambda_hi = bayesianmethod(FN, FP, TN, TP, ep_lo, ep_hi)
    epsilon_lo = lam_toep(lambda_lo, 0)
    epsilon_hi = lam_toep(lambda_hi, 0)
    interval = epsilon_hi - epsilon_lo

    end_time = time.time()
    audit_time = end_time - start_time

    print("epsilon_lo:", epsilon_lo)
    print("epsilon_hi:", epsilon_hi)
    print("interval:", interval)
    print("auditing_time:", audit_time)