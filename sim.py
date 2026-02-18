import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

# =============================================================================
# SEGMENT 1: CONFIGURATION & PARAMETERS
# =============================================================================
CONFIG = {
    'n_samples': int(1e7),      # Number of Monte Carlo samples (Increase for smoother tails)
    'snr_db_range': np.arange(0, 121, 5),
    'threshold_snr': 1.0,       # Outage threshold (usually 1.0 or 0dB)
    
    # UOWC Physical Parameters (Table IV, Set 1)
    'uowc': {
        'a': 0.6302,
        'd': 1.1780,
        'p': 0.8444,
        'rho2': 1.0,            # Pointing error parameter
        'A_eq': 1.0             # Aperture equivalent (normalized)
    },
    
    # Placeholder for future TOWC parameters
    'towc': {
        'alpha': 2.0,           # Example param
        'beta': 2.0
    }
}

# =============================================================================
# SEGMENT 2: CHANNEL MODEL DISTRIBUTIONS (Interchangeable)
# =============================================================================

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

# =============================================================================
# SEGMENT 3: SIMULATION ENGINE (Monte Carlo)
# =============================================================================

def calculate_outage_probability(h_channel, snr_db_range, threshold=1.0):
    """
    Generic function to calculate outage for ANY channel array h_channel.
    """
    n_samples = len(h_channel)
    outage_probs = []
    
    # Pre-calculate channel power to speed up loop
    h_sq = h_channel**2
    
    for snr_db in snr_db_range:
        avg_snr_lin = 10**(snr_db / 10.0)
        
        # Outage Condition: Instantaneous SNR < Threshold
        # avg_snr * h^2 < threshold  =>  h^2 < threshold / avg_snr
        thresh_effective = threshold / avg_snr_lin
        
        # Vectorized counting
        count = np.sum(h_sq < thresh_effective)
        outage_probs.append(count / n_samples)
        
    return outage_probs

# =============================================================================
# SEGMENT 4: MAIN EXECUTION & TOWC INTEGRATION POINT
# =============================================================================

def main():
    print(f"Running Monte Carlo Simulation with {CONFIG['n_samples']:.0e} samples...")

    # --- Step A: Generate UOWC Link ---
    # 1. Get Turbulence (You can swap 'get_gg_turbulence' here)
    h_u_turb = get_gg_turbulence(CONFIG['uowc'], CONFIG['n_samples'])
    
    # 2. Get Pointing Errors
    h_u_point = get_pointing_error(CONFIG['uowc']['rho2'], CONFIG['uowc']['A_eq'], CONFIG['n_samples'])
    
    # 3. Combine UOWC Channel
    h_uowc = h_u_turb * h_u_point
    
    # --- Step B: Generate TOWC Link (Placeholder) ---
    # To add TOWC, uncomment and implement below:
    # h_t_turb = get_gamma_gamma_turbulence(CONFIG['towc'], CONFIG['n_samples'])
    # h_t_point = get_pointing_error(...)
    # h_towc = h_t_turb * h_t_point
    
    # --- Step C: End-to-End Channel ---
    # If using Relay (Decode-and-Forward), you analyze links separately.
    # If using Amplify-and-Forward (AF), you multiply them:
    # h_total = h_uowc * h_towc 
    
    # For now, we simulate UOWC only (Fig 2a)
    h_final = h_uowc 

    # --- Step D: Calculate Performance ---
    outage = calculate_outage_probability(h_final, CONFIG['snr_db_range'], CONFIG['threshold_snr'])

    # --- Step E: Plotting ---
    plt.figure(figsize=(10, 6))
    plt.semilogy(CONFIG['snr_db_range'], outage, 'bo-', label='UOWC (GG + Pointing)', markeredgecolor='b', markerfacecolor='none')
    
    plt.title(f"Outage Probability (Monte Carlo, N={CONFIG['n_samples']:.0e})")
    plt.xlabel("Average SNR (dB)")
    plt.ylabel("Outage Probability")
    plt.grid(True, which="both", linestyle='--', alpha=0.5)
    plt.legend()
    plt.ylim(1e-6, 1)
    plt.xlim(0, 120)
    plt.show()

if __name__ == "__main__":
    main()