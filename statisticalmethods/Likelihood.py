import scipy.stats as stats
import numpy as np
from scipy.optimize import minimize_scalar


def likelihood_ratio(successes, trials, alpha=0.05):
    p_hat = successes / trials  # MLE of p
    lr_threshold = stats.chi2.ppf(1 - alpha, df=1)  # equals to stats.norm.ppf(1 - alpha/2)**2
    print(lr_threshold)

    # Define the likelihood function
    def log_likelihood(p):
        return successes * np.log(p) + (trials - successes) * np.log(1 - p)

    # Define the function to find the lower or upper bound
    def bound_func(p):
        return -2 * (log_likelihood(p) - log_likelihood(p_hat)) - lr_threshold

    # Use a numerical solver to find the bounds
    lower_bound = minimize_scalar(lambda p: abs(bound_func(p)), bounds=(0, p_hat), method='bounded').x
    upper_bound = minimize_scalar(lambda p: abs(bound_func(p)), bounds=(p_hat, 1), method='bounded').x

    return lower_bound, upper_bound


# a simple test
if __name__ == '__main__':
    trials = 1000
    successes = 450

    ci_lower, ci_upper = likelihood_ratio(successes, trials)
    print(f"Likelihood Ratio Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")