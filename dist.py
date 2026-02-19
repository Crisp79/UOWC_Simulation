import numpy as np
from scipy.special import gamma

def get_gg_turbulence(params, n_samples):
    """
    Generates Normalized Generalized Gamma (GG) samples.
    To replace this with another distribution (e.g., Gamma-Gamma), 
    create a similar function and point to it in Segment 4.
    """
    a = params['a']
    d = params['d']
    p = params['p']
    
    # 1. Generate raw GG samples using Gamma transformation
    # X ~ GG(a,d,p) <=> X = a * G^(1/p) where G ~ Gamma(d/p, 1)
    shape_k = d / p
    g_samples = np.random.gamma(shape=shape_k, scale=1.0, size=n_samples)
    h_raw = a * np.power(g_samples, 1/p)
    
    # 2. Theoretical Normalization (Crucial for correct SNR scaling)
    # Normalize so E[h^2] = 1
    moment_2 = (a**2) * gamma((d + 2) / p) / gamma(d / p)
    h_norm = h_raw / np.sqrt(moment_2)
    
    return h_norm

def get_pointing_error(rho2, A_eq, n_samples):
    """
    Generates Pointing Error loss samples.
    """
    # Inverse CDF method: x = A_eq * U^(1/rho^2)
    u = np.random.uniform(0, 1, n_samples)
    h_p = A_eq * np.power(u, 1/rho2)
    return h_p

# --- Future Extension: Example of how you would add Gamma-Gamma ---
def get_gamma_gamma_turbulence(params, n_samples):
    # alpha = params['alpha']
    # beta = params['beta']
    # ... logic ...
    # return h_norm
    pass