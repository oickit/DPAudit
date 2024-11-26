import numpy as np

def laplace_mechanism(q, s, epsilon):
    """
    Arguments:
    q : real query result
    s : sensitivity
    epsilon : target epsilon (epsilon > 0)
    """
    b = s / epsilon
    noise = np.random.laplace(scale=b)
    f = q + noise
    return f

def laplace_b(s, epsilon):
    b = s / epsilon
    return b