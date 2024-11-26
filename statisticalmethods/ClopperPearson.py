import scipy.stats
import math
import random


def clopper_pearson(x, n, alpha = 0.05):
    """Estimate the confidence interval for a sampled Bernoulli random
    variable.
    `x` is the number of successes and `n` is the number trials (x <=
    n). `alpha` is the confidence level (i.e., the true probability is
    inside the confidence interval with probability 1-alpha). The
    function returns a `(low, high)` pair of numbers indicating the
    interval on the probability.
    """
    b = scipy.stats.beta.ppf
    lo = b(alpha / 2, x, n - x + 1)
    hi = b(1 - alpha / 2, x + 1, n - x)
    return 0.0 if math.isnan(lo) else lo, 1.0 if math.isnan(hi) else hi


# a simple test
if __name__ == '__main__':
    total = 10000
    successes = 1000
    lo, hi = clopper_pearson(successes, total)
    print('95% confidence interval: {:.4f}-{:.4f}'.format(lo, hi))