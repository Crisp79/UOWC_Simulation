import numpy as np
from scipy.special import gamma


def sample_gg(params, n_samples):
    """
    Generates Normalized Generalized Gamma (GG) samples.
    To replace this with another distribution (e.g., Gamma-Gamma),
    create a similar function and point to it in Segment 4.
    """
    a = params["a"]
    d = params["d"]
    p = params["p"]

    # 1. Generate raw GG samples using Gamma transformation
    # X ~ GG(a,d,p) <=> X = a * G^(1/p) where G ~ Gamma(d/p, 1)
    shape_k = d / p
    g_samples = np.random.gamma(shape=shape_k, scale=1.0, size=n_samples)
    h_raw = a * np.power(g_samples, 1 / p)

    # 2. Theoretical Normalization (Crucial for correct SNR scaling)
    # Normalize so E[h^2] = 1
    moment_2 = (a**2) * gamma((d + 2) / p) / gamma(d / p)
    h_norm = h_raw / np.sqrt(moment_2)

    return h_norm


def sample_pointing_uowc(rho2, A_eq, n_samples):
    """
    Generates Pointing Error loss samples.
    """
    # Inverse CDF method: x = A_eq * U^(1/rho^2)
    u = np.random.uniform(0, 1, n_samples)
    h_p = A_eq * np.power(u, 1 / rho2)
    return h_p


def sample_egg(params, num_samples):
    """
    Samples from the Exponential-Generalized Gamma (EGG) distribution.

    Parameters:
    num_samples  : int, number of samples to generate
    omega        : float, mixture coefficient (weight of the exponential part)
    lambda_param : float, scale parameter for the exponential part
    a            : float, scale parameter for the GG part
    d_over_p     : float, shape parameter (d/p) for the GG part
    p            : float, shape parameter for the GG part
    """
    omega = params["omega_1"]
    lambda_param = params["lambda_1"]
    a = params["a_1"]
    d_over_p = params["d_over_p_1"]
    p = params["p_1"]

    # 1. Flip a biased coin for each sample to decide which distribution to use
    u = np.random.uniform(0, 1, num_samples)

    # Masks to route samples to either Exponential or GG
    is_exp = u < omega
    is_gg = ~is_exp

    # Count how many of each we need
    num_exp = np.sum(is_exp)
    num_gg = np.sum(is_gg)

    # Create the output array
    samples = np.empty(num_samples)

    # 2. Sample the Exponential part
    if num_exp > 0:
        # np.random.exponential takes the scale parameter directly
        samples[is_exp] = np.random.exponential(scale=lambda_param, size=num_exp)

    # 3. Sample the Generalized Gamma part
    if num_gg > 0:
        # GG can be sampled by transforming standard Gamma samples
        # x = a * (Gamma(k=d/p, theta=1))^(1/p)
        gamma_samples = np.random.gamma(shape=d_over_p, scale=1.0, size=num_gg)
        samples[is_gg] = a * (gamma_samples ** (1 / p))

    return samples


def sample_ew(params, num_samples):
    """
    Samples points from the Exponentiated Weibull distribution.

    Args:
        params_json (str): JSON string containing alpha, beta, and eta.
        num_samples (int): Number of samples to generate.

    Returns:
        list: A list of floats sampled from the distribution.
    """
    # Parse parameters from JSON
    alpha = params["alpha"]
    beta = params["beta"]
    eta = params["eta"]

    # Generate uniform random variables
    u = np.random.uniform(0, 1, num_samples)

    # Apply the Inverse Transform Sampling formula
    # x = eta * (-ln(1 - u^(1/alpha)))^(1/beta)
    samples = eta * ((-np.log(1 - u ** (1 / alpha))) ** (1 / beta))

    return samples


# Example Usage:
# json_input = '{"alpha": 2.0, "beta": 1.5, "eta": 10.0}'
# data = sample_exponentiated_weibull(json_input, 1000)
# 

def sample_gamma_gamma(params, num_samples):
    """
    Samples points from the Gamma-Gamma distribution using 
    the product of two independent Gamma distributions.
    
    Args:
        params_json (str): JSON string containing alpha and beta.
        num_samples (int): Number of samples to generate.
        
    Returns:
        list: A list of floats sampled from the distribution.
    """
    # Parse parameters from JSON
    alpha = params['alpha']
    beta = params['beta']
    
    # Generate two independent Gamma sets
    # numpy.random.gamma(shape, scale, size)
    gamma_large = np.random.gamma(alpha, 1/alpha, num_samples)
    gamma_small = np.random.gamma(beta, 1/beta, num_samples)
    
    # The Gamma-Gamma sample is the point-wise product
    samples = gamma_large * gamma_small
    
    return samples
