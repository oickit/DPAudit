import numpy as np
import math

def gaussian_mechanism(q, s, epsilon, delta):
    """
    Arguments:
    q : real query result
    s : sensitivity
    epsilon : target epsilon (epsilon > 0)
    delta : target delta (0 < delta < 1)
    """
    variance = (2 * s**2 * math.log(1.25 / delta)) / epsilon**2
    std_dev = np.sqrt(variance)
    noise = np.random.normal(0, std_dev, 1)
    f = q + noise
    return f


def gaussian_sigma(s, epsilon, delta):
    variance = (2 * s**2 * math.log(1.25 / delta)) / epsilon**2
    std_dev = np.sqrt(variance)
    return std_dev