from mechanism.analytic_gauss import calibrateAnalyticGaussianMechanism
from mechanism.gaussian import gaussian_sigma
from statisticalmethods.ClopperPearson import clopper_pearson
from scipy.integrate import dblquad
from scipy.optimize import root_scalar
from scipy.stats import norm, beta
from scipy.optimize import newton
import numpy as np
import time

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

"""
Auditing Gauss-based DP-DBs with BayE
"""


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


def bayesianmethod(fn, fp, tn, tp, n, gamma=0.05, tol=1e-5):
    def joint_density(fn, fp, tn, tp, x, y):
        fpr_density = beta(fn+0.5, tp+0.5).pdf(y)
        fnr_density = beta(fp+0.5, tn+0.5).pdf(x)
        return fpr_density * fnr_density

    def y_lower(mu, x):
        return norm.cdf(norm.ppf(1 - x) - mu)

    def y_upper(mu, x):
        return 1 - norm.cdf(norm.ppf(x) - mu)

    def pp(mu):
        t1 = 0
        step = 0.05
        for x in np.arange(0, 0.5, step):
            t1 += dblquad(lambda y, x: joint_density(fn=fn, fp=fp, tn=tn, tp=tp, x=x, y=y),
                          x, x + step, lambda x: y_lower(mu=mu, x=x), lambda x: y_upper(mu=mu, x=x))[0]
        t2 = dblquad(lambda y, x: joint_density(fn=fn, fp=fp, tn=tn, tp=tp, x=x, y=y),
                     0.5, 1, lambda x: y_lower(mu=mu, x=x), lambda x: y_upper(mu=mu, x=x))[0]
        return t1 + t2

    def pp_lo(mu):
        return pp(mu) - gamma/2

    def pp_hi(mu):
        return pp(mu) - 1 + gamma/2


    [fp_lo, fp_hi] = clopper_pearson(fp, n, alpha = 0.01)
    [tn_lo, tn_hi] = clopper_pearson(tn, n, alpha = 0.01)
    fpr_lo = fp_lo / (fp_hi + tn_hi)
    fpr_hi = fp_hi / (fp_lo + tn_lo)

    [tp_lo, tp_hi] = clopper_pearson(tp, n, alpha = 0.01)
    [fn_lo, fn_hi] = clopper_pearson(fn, n, alpha = 0.01)
    fnr_lo = fn_lo / (fn_hi + tp_hi)
    fnr_hi = fn_hi / (fn_lo + tp_lo)

    mu_lo = norm.ppf(1 - fpr_hi) - norm.ppf(fnr_hi)
    mu_hi = norm.ppf(1 - fpr_lo) - norm.ppf(fnr_lo)

    estimated_mu_lo = root_scalar(pp_lo, bracket=[mu_lo, mu_hi], xtol=tol).root
    estimated_mu_hi = root_scalar(pp_hi, bracket=[mu_lo, mu_hi], xtol=tol).root

    return estimated_mu_lo, estimated_mu_hi


def transmutoeps(delta, mu):
    def equation(epsilon, delta, mu):
        return delta - (norm.cdf(-epsilon / mu + mu / 2) - np.exp(epsilon) * norm.cdf(-epsilon / mu - mu / 2))

    epsilon_initial = 0.0  # initialization
    if mu <= 0:  # unable to provide an effective lower bound
        return epsilon_initial
    epsilon = newton(equation, epsilon_initial, args = (delta, mu))
    return epsilon


if __name__ == "__main__":
    ep = 2.0
    de = 0.0000001
    num = 20000
    print(ep)
    start_time = time.time()
    tr_1, tr_2, no_1, no_2 = construct_ob(num, ep, de)

    FN, FP, TN, TP = identificationresult(tr_1, tr_2, no_1, no_2, num)
    print(FN, FP, TN, TP)

    mu_lo, mu_hi = bayesianmethod(FN, FP, TN, TP, num)

    epsilon_lo = transmutoeps(de, mu_lo)
    epsilon_hi = transmutoeps(de, mu_hi)
    interval = epsilon_hi - epsilon_lo

    end_time = time.time()
    audit_time = end_time - start_time

    print("epsilon_lo:", epsilon_lo)
    print("epsilon_hi:", epsilon_hi)
    print("interval:", interval)
    print("auditing_time:", audit_time)
