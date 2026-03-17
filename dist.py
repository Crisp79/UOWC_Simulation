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


def sample_pointing(rho2, A_eq, n_samples):
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
    alpha = params["alpha"]
    beta = params["beta"]

    # Generate two independent Gamma sets
    # numpy.random.gamma(shape, scale, size)
    gamma_large = np.random.gamma(alpha, 1 / alpha, num_samples)
    gamma_small = np.random.gamma(beta, 1 / beta, num_samples)

    # The Gamma-Gamma sample is the point-wise product
    samples = gamma_large * gamma_small

    return samples


def compute_malaga_params(alphaM, betaM, OmegaM, rho_los):
    """
    Compute Malaga mixture weights and Gamma scale parameters.

    Malaga parameterisation (from paper [38], Jurado-Navas 2011):
      sigma_s2 = total scatter power = 1.0 (normalized)
      gM = 2 * sigma_s2 * (1 - rho_los)     [scatter power NOT in LOS]
      OmegaM_prime = OmegaM + 2*rho_los*sigma_s2 + 2*sqrt(...)
                   ~ OmegaM (simplified for rho_los < 1)

    Mixture weights (eq. from Ansari 2016, matches paper's eq.2):
      w_m1 = C(betaM-1, m1-1) * Gamma(alphaM+m1) / (Gamma(alphaM)*Gamma(m1+1))
             * (OmegaM/(gM*betaM+OmegaM))^m1
             * (gM*betaM/(gM*betaM+OmegaM))^alphaM
             / (gM*betaM+OmegaM)^(m1) * ...  [normalised to sum to 1]

    Gamma scale for each term:
      theta_m1 = (gM*betaM + OmegaM) / (alphaM * betaM)
    """
    from scipy.special import comb

    sigma_s2 = 1.0  # normalized total scatter power
    gM = 2.0 * sigma_s2 * (1.0 - rho_los)  # scatter power off LOS
    xi = gM * betaM / (gM * betaM + OmegaM)  # shorthand

    # Compute unnormalized weights for m1 = 1 ... betaM
    weights = []
    for m1 in range(1, betaM + 1):
        binom_coeff = comb(betaM - 1, m1 - 1, exact=True)
        w = (
            binom_coeff
            * gamma(alphaM + m1)
            / (gamma(alphaM) * gamma(m1 + 1))
            * ((OmegaM / (gM * betaM + OmegaM)) ** m1)
            * (xi**alphaM)
        )
        weights.append(w)

    weights = np.array(weights, dtype=float)
    weights /= weights.sum()  # normalize to valid probability weights

    # Gamma shape and scale for each mixture component
    # Shape = alphaM + m1,  Scale = (gM*betaM + OmegaM) / (alphaM * betaM)
    theta = (gM * betaM + OmegaM) / (alphaM * betaM)

    return weights, theta, gM


def sample_malaga(cfg, n_samples):
    """
    Sample Malaga turbulence using mixture-of-Gammas method.
    Paper eq.(2) — terrestrial atmospheric turbulence.

    Steps:
    1. Compute mixture weights w_m1 for m1=1..betaM
    2. For each sample: pick component m1 ~ Categorical(w), then draw Gamma
    3. Normalize so E[h^2] = 1
    """
    alphaM = cfg["alphaM"]
    betaM = cfg["betaM"]
    OmegaM = cfg["omegaM"]
    rho_los = cfg["rho_loss"]

    weights, theta, gM = compute_malaga_params(alphaM, betaM, OmegaM, rho_los)

    # Draw component indices according to mixture weights
    components = np.random.choice(betaM, size=n_samples, p=weights)

    # Draw Gamma samples for each component
    h_raw = np.zeros(n_samples)
    for m1_idx in range(betaM):
        mask = components == m1_idx
        m1 = m1_idx + 1  # m1 runs from 1 to betaM
        shape_param = alphaM + m1
        count = mask.sum()
        if count > 0:
            h_raw[mask] = np.random.gamma(shape=shape_param, scale=theta, size=count)

    # Normalize: E[h^2] = Var(h) + (E[h])^2
    # For Malaga: E[h] = OmegaM + gM (= total mean power = 1 normalized)
    # We compute empirically for safety
    mean_h2 = np.mean(h_raw**2)
    if mean_h2 > 0:
        h_norm = h_raw / np.sqrt(mean_h2)
    else:
        h_norm = h_raw

    return h_norm


# -------------------------------------------------------
# 2C. TOWC: Fog-induced fading — paper eq.(6)
#
# f(x) = (z^k / Gamma(k)) * (log(1/x))^(k-1) * x^(z-1),  0 < x <= 1
#
# where z = 4.343 / (beta_f * d_T)
#
# This is a Gamma distribution in the variable Y = -log(x):
#   Y = log(1/x) ~ Gamma(k, 1/z)
#   => x = exp(-Y)
#
# Sampling: Y ~ Gamma(k, 1/z), then h_f = exp(-Y)
# -------------------------------------------------------
def sample_fog(cfg, n_samples):
    """
    Sample fog-induced fading — paper eq.(6).

    z = 4.343 / (beta_f * d_T)
    Y = log(1/x) ~ Gamma(k, 1/z)
    h_f = exp(-Y)

    Paper Section V: light fog: k=2.32, beta_f=13.12, d_T=0.4km
    """
    k = cfg["fog_k"]
    beta_f = cfg["fog_beta"]
    d_T = cfg["d_T"]

    z = 4.343 / (beta_f * d_T)  # paper eq.(6) definition of z

    # Y ~ Gamma(k, scale=1/z)
    Y = np.random.gamma(shape=k, scale=1.0 / z, size=n_samples)
    h_f = np.exp(-Y)  # h_f = exp(-Y) = exp(log x) = x

    # Note: fog fading is NOT normalized to E[h^2]=1 because it represents
    # a real path loss. E[h_f] = (z/(z+1))^k (mean < 1 = loss).
    # This correctly degrades SNR as in the paper.
    return h_f


# =============================================================================
# SEGMENT 3: SIMULATION ENGINE
# =============================================================================


# def calculate_outage_probability(h_channel, snr_db_range, threshold=1.0):
#     """
#     Single-hop outage probability — paper eq.(20):
#       P_out = P(gamma < gamma_th) = P(gamma_bar * h^2 < gamma_th)
#     """
#     h_sq = h_channel**2
#     outage = []
#     for snr_db in snr_db_range:
#         snr_lin = 10 ** (snr_db / 10.0)
#         thresh = threshold / snr_lin
#         outage.append(np.mean(h_sq < thresh))
#     return np.array(outage)


# def calculate_af_outage(h_towc, h_uowc, snr_db_range, threshold=1.0, C=1.0):
#     """
#     Fixed-gain AF end-to-end outage — paper eq.(29):
#       gamma_e2e = (gamma_T * gamma_U) / (gamma_U + C)

#     Paper uses C as a constant related to relay gain.
#     Outage: P(gamma_e2e < gamma_th)
#     """
#     h_T_sq = h_towc**2
#     h_U_sq = h_uowc**2
#     outage = []
#     for snr_db in snr_db_range:
#         snr_lin = 10 ** (snr_db / 10.0)
#         g_T = snr_lin * h_T_sq
#         g_U = snr_lin * h_U_sq
#         g_e2e = (g_T * g_U) / (g_U + C)  # paper eq.(29)
#         outage.append(np.mean(g_e2e < threshold))
#     return np.array(outage)


# def calculate_df_outage(h_towc, h_uowc, snr_db_range, threshold=1.0):
# """
# DF outage (for comparison — not the paper's scheme):
#   P_out = 1 - (1-P1)(1-P2)
# """
# p1 = calculate_outage_probability(h_towc, snr_db_range, threshold)
# p2 = calculate_outage_probability(h_uowc, snr_db_range, threshold)
# return 1.0 - (1.0 - p1) * (1.0 - p2), p1, p2
