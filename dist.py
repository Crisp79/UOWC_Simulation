import numpy as np
from scipy.special import comb, gamma
from scipy.special import gammaln

def sample_gg(params, n_samples):
    """Generate normalized Generalized Gamma distributed samples."""
    # Extract GG distribution parameters
    a = params["a"]
    d = params["d"]
    p = params["p"]

    # Generalized Gamma is defined as: X = a * G^(1/p) where G ~ Gamma(d/p, 1)
    # First compute the shape parameter for the underlying Gamma distribution
    shape_k = d / p
    g_samples = np.random.gamma(shape=shape_k, scale=1.0, size=n_samples)
    h_raw = a * np.power(g_samples, 1 / p)

    # Normalize so that E[h^2] = 1 (unit mean square for correct SNR scaling)
    # Theoretical second moment: E[h^2] = a^2 * Gamma((d+2)/p) / Gamma(d/p)
    moment_2 = (a**2) * gamma((d + 2) / p) / gamma(d / p)
    h_norm = h_raw / np.sqrt(moment_2)

    return h_norm


def sample_pointing(rho2, A_eq, n_samples):
    """Generate pointing error loss samples using inverse CDF method."""
    # Generate uniform random variables [0, 1]
    u = np.random.uniform(0, 1, n_samples)

    # Apply inverse CDF of pointing error distribution
    # CDF inversion: h_p = A_eq * U^(1/rho^2)
    # This ensures proper statistical distribution of pointing losses
    h_p = A_eq * np.power(u, 1 / rho2)
    h_p = h_p / np.sqrt(np.mean(h_p**2))
    return h_p


def sample_egg(params, num_samples):
    """Generate Exponential-Generalized Gamma mixture distribution samples."""
    # Extract mixture parameters
    omega = params["omega_1"]  # Mixture weight for exponential component
    lambda_param = params["lambda_1"]  # Scale parameter for exponential
    a = params["a_1"]  # Scale parameter for GG
    d_over_p = params["d_over_p_1"]  # Shape parameter for GG
    p = params["p_1"]  # Shape parameter for GG

    # Generate uniform random variables to select mixture component
    u = np.random.uniform(0, 1, num_samples)

    # Create boolean masks for routing samples to appropriate distribution
    is_exp = u < omega  # Select exponential with probability omega
    is_gg = ~is_exp  # Remaining samples go to GG

    # Count how many samples needed from each distribution
    num_exp = np.sum(is_exp)
    num_gg = np.sum(is_gg)

    # Initialize output array
    samples = np.empty(num_samples)

    # Sample exponential component if needed
    if num_exp > 0:
        samples[is_exp] = np.random.exponential(scale=lambda_param, size=num_exp)

    # Sample GG component if needed
    if num_gg > 0:
        gamma_samples = np.random.gamma(shape=d_over_p, scale=1.0, size=num_gg)
        samples[is_gg] = a * (gamma_samples ** (1 / p))

    return samples


def sample_ew(params, num_samples):
    """Generate Exponentiated Weibull distributed samples."""
    # Extract distribution parameters
    alpha = params["alpha"]  # Exponentiation parameter
    beta = params["beta"]  # Weibull shape parameter
    eta = params["eta"]  # Scale parameter

    # Generate uniform random variables [0, 1]
    u = np.random.uniform(0, 1, num_samples)

    # Apply inverse transform sampling for Exponentiated Weibull
    # Formula: X = eta * (-ln(1 - U^(1/alpha)))^(1/beta)
    # This maps uniform [0,1] to EW distribution
    samples = eta * ((-np.log(1 - u ** (1 / alpha))) ** (1 / beta))

    return samples


def sample_gamma_gamma(params, num_samples):
    """Generate Gamma-Gamma distributed samples as product of two independent Gamma variables."""
    # Extract distribution parameters
    alpha = params["alpha"]  # First Gamma shape parameter
    beta = params["beta"]  # Second Gamma shape parameter

    # Gamma-Gamma distribution = product of two independent Gamma RVs
    # G1 ~ Gamma(alpha, 1/alpha) and G2 ~ Gamma(beta, 1/beta)
    # This models strong turbulence in optical channels
    gamma_large = np.random.gamma(alpha, 1 / alpha, num_samples)
    gamma_small = np.random.gamma(beta, 1 / beta, num_samples)

    # Multiply the two Gamma variables element-wise
    samples = gamma_large * gamma_small

    return samples


def compute_malaga_params(alphaM, betaM, OmegaM, rho_los):
    """Compute Malaga mixture weights and Gamma scale parameters."""
    # Malaga distribution is a mixture of Gamma distributions
    # Set normalized total scatter power
    sigma_s2 = 1.0

    # Scatter power not in LOS component: g_M = 2 * sigma_s2 * (1 - rho_los)
    gM = 2.0 * sigma_s2 * (1.0 - rho_los)

    # Shorthand for mixture parameter calculation
    xi = gM * betaM / (gM * betaM + OmegaM)

    # Compute mixture weights for each component m1 = 1 to betaM
    weights = []
    for m1 in range(1, betaM + 1):
        # Binomial coefficient: C(betaM-1, m1-1)
        binom_coeff = comb(betaM - 1, m1 - 1, exact=True)

        # Weight calculation from Malaga distribution theory
        w = (
            binom_coeff
            * gamma(alphaM + m1)
            / (gamma(alphaM) * gamma(m1 + 1))
            * ((OmegaM / (gM * betaM + OmegaM)) ** m1)
            * (xi**alphaM)
        )
        weights.append(w)

    # Normalize weights to form valid probability distribution
    weights = np.array(weights, dtype=float)
    weights /= weights.sum()

    # Scale parameter for all Gamma mixture components
    theta = (gM * betaM + OmegaM) / (alphaM * betaM)
    
    print("weights sum:", weights.sum())
    print("min weight:", weights.min())
    print("max weight:", weights.max())

    return weights, theta, gM


def sample_malaga(cfg, n_samples):
    """
    Málaga turbulence sampler (stable + paper-consistent)

    Uses mixture-of-Gamma approximation aligned with Eq.(5)
    """

    alphaM = cfg["alphaM"]
    betaM = cfg["betaM"]
    OmegaM = cfg["omegaM"]
    rho_los = cfg["rho_loss"]

    # --- Step 1: Derived parameters ---
    gM = rho_los
    Omega = OmegaM

    m_vals = np.arange(1, betaM + 1)

    # --- Step 2: Compute mixture weights b_m (log-domain) ---
    # From Málaga model structure (stable form)

    log_b = (
        gammaln(alphaM + m_vals)
        - gammaln(m_vals)
        - gammaln(alphaM)
        + m_vals * np.log(gM + 1e-300)
        - m_vals * np.log(gM + Omega + 1e-300)
    )

    # Stabilize
    log_b -= np.max(log_b)
    weights = np.exp(log_b)

    # Normalize
    weights /= np.sum(weights)

    # --- Step 3: Sample mixture components ---
    components = np.random.choice(betaM, size=n_samples, p=weights)

    # --- Step 4: Gamma sampling ---
    h_raw = np.zeros(n_samples)

    theta = (gM + Omega) / alphaM  # scale parameter

    for m1_idx in range(betaM):
        mask = components == m1_idx
        count = mask.sum()

        if count > 0:
            m1 = m1_idx + 1
            shape = alphaM + m1

            h_raw[mask] = np.random.gamma(
                shape=shape,
                scale=theta,
                size=count
            )

    # --- Step 5: Normalize power ---
    mean_h2 = np.mean(h_raw**2)

    if mean_h2 > 0 and np.isfinite(mean_h2):
        h_norm = h_raw / np.sqrt(mean_h2)
    else:
        print("[WARNING] Málaga normalization failed")
        h_norm = h_raw

    return h_norm



def sample_fog(cfg, n_samples):
    """Generate fog-induced fading samples using exponential transformation of Gamma."""
    # Extract fog parameters
    k = cfg["fog_k"]  # Gamma shape parameter
    beta_f = cfg["fog_beta"]  # Fog absorption coefficient
    d_T = cfg["d_T"]  # Distance in km

    # Calculate z parameter from fog model
    # z = 4.343 / (beta_f * d_T) defines the rate of fading
    z = 4.343 / (beta_f * d_T)

    # Fog fading distribution uses log-transformation of Gamma
    # Y = log(1/x) ~ Gamma(k, 1/z)
    Y = np.random.gamma(shape=k, scale=1.0 / z, size=n_samples)

    # Apply exponential transformation: h_f = exp(-Y)
    # This maps Gamma to fog fading distribution with proper attenuation
    h_f = np.exp(-Y)
    return h_f
    
def sample_multilayer_channel(model, params, n_layers, n_samples):
    """
    Generate cascaded multi-layer channel:
    h_c = ∏ h_i for i = 1..N
    """

    # Initialize channel as ones (multiplicative identity)
    h_total = np.ones(n_samples)

    for _ in range(n_layers):
        if model == "gg":
            h_layer = sample_gg(params, n_samples)

        elif model == "egg":
            h_layer = sample_egg(params, n_samples)

        elif model == "ew":
            h_layer = sample_ew(params, n_samples)

        elif model == "gamma_gamma":
            h_layer = sample_gamma_gamma(params, n_samples)

        else:
            raise ValueError("Unknown model")

        # Multiply layer contribution
        h_total *= h_layer
    
    mean_h2 = np.mean(h_total**2)
    if mean_h2 > 0:
        h_total = h_total / np.sqrt(mean_h2)
    return h_total
