import math


def calculate_snr_imdd(params):
    p_tx_dbm = params["p_tx"]
    distance_m = params["dist"]
    alpha_m_inv = params["alpha"]
    noise_psd_a2_ghz = params["sigma_t"]

    # 1. Convert dBm to Watts
    p_tx_watts = 10 ** ((p_tx_dbm - 30) / 10)

    # 2. Calculate Received Optical Power (Beer-Lambert Law)
    # P_rx = P_tx * exp(-alpha * L)
    p_rx_watts = p_tx_watts * math.exp(-alpha_m_inv * distance_m)

    # 3. Total Noise Variance (Sigma^2 = PSD * B)
    # Since PSD is in A^2/GHz and B is in GHz, units cancel correctly
    noise_variance = noise_psd_a2_ghz

    # 4. Average SNR (Electrical)
    # SNR = I_p^2 / sigma^2
    snr_linear = (p_rx_watts**2) / noise_variance

    # 5. Convert to dB
    snr_db = 10 * math.log10(snr_linear)

    return snr_db
