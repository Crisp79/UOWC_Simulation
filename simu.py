import numpy as np
from scipy.special import erfc


def calculate_outage_probability(h_channel, snr_db_range, threshold=1.0):
    """
    Generic function to calculate outage for ANY channel array h_channel.
    """
    n_samples = len(h_channel)
    outage_probs = []

    # Pre-calculate channel power to speed up loop
    h_sq = h_channel**2

    for snr_db in snr_db_range:
        avg_snr_lin = 10 ** (snr_db / 10.0)

        # Outage Condition: Instantaneous SNR < Threshold
        # avg_snr * h^2 < threshold  =>  h^2 < threshold / avg_snr
        thresh_effective = threshold / avg_snr_lin

        # Vectorized counting
        count = np.sum(h_sq < thresh_effective)
        outage_probs.append(count / n_samples)

    return outage_probs


def calculate_average_ber(h_channel, snr_db_range):
    """
    Calculates Average BER using Monte Carlo integration.
    Assumes IM/DD with OOK (common in underwater optical).
    Formula: BER_inst = 0.5 * erfc(0.5 * sqrt(SNR_inst))
    (Check your paper for the exact modulation factor 'k')
    """
    n_samples = len(h_channel)
    ber_results = []

    # Pre-calculate squared channel
    h_sq = h_channel**2

    for snr_db in snr_db_range:
        # Convert Average SNR to Linear
        avg_snr_lin = 10 ** (snr_db / 10.0)

        # Calculate Instantaneous SNR for all samples
        # gamma = avg_snr * h^2
        inst_snr = avg_snr_lin * h_sq

        # Calculate BER for EACH sample
        # Using approximation for OOK: Pe = Q(sqrt(gamma/2))
        # Q(x) = 0.5 * erfc(x / sqrt(2))
        # So Pe = 0.5 * erfc(sqrt(gamma/2) / sqrt(2)) = 0.5 * erfc(sqrt(gamma)/2)

        # Note: If paper uses BPSK, it might be 0.5 * erfc(sqrt(gamma))
        # Let's assume standard IM/DD OOK:
        ber_instantaneous = 0.5 * erfc(np.sqrt(inst_snr) / 2.0)

        # Average over all samples
        avg_ber = np.mean(ber_instantaneous)
        ber_results.append(avg_ber)

    return ber_results

def calculate_ergodic_capacity(h_channel, snr_db_range):
    """
    Calculates Ergodic Capacity using Monte Carlo integration.
    Formula: C = E[ log2(1 + SNR_inst) ]
    
    Units: Bits per second per Hertz (bps/Hz)
    """
    n_samples = len(h_channel)
    capacity_results = []

    # Pre-calculate squared channel
    h_sq = h_channel**2

    for snr_db in snr_db_range:
        # Convert Average SNR to Linear
        avg_snr_lin = 10 ** (snr_db / 10.0)

        # Calculate Instantaneous SNR: gamma = avg_snr * h^2
        inst_snr = avg_snr_lin * h_sq

        # Calculate Instantaneous Capacity for all samples
        # C = log2(1 + gamma)
        # Using np.log1p(x) is more numerically stable for small x than log(1+x)
        cap_instantaneous = np.log2(1 + inst_snr)

        # Average over all samples (Monte Carlo integration)
        avg_capacity = np.mean(cap_instantaneous)
        capacity_results.append(avg_capacity)

    return capacity_results