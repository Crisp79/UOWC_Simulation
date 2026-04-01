import matplotlib.pyplot as plt
import numpy as np


# =========================
# 1. BASIC UTILITIES
# =========================
def generate_bits(N):
    return np.random.randint(0, 2, N)


def ook_mod(bits, A=1.0):
    return A * bits


def ook_detect(y, threshold):
    return (y > threshold).astype(int)


def compute_ber(bits, bits_hat):
    return np.mean(bits != bits_hat)


# =========================
# 2. EXACT CHANNEL MODELS
# =========================


def generalized_gamma(N, a, d, p):
    y = np.random.gamma(shape=d / p, scale=1.0, size=N)
    return a * (y ** (1 / p))


def egg_fading(N, omega=0.1953, lam=0.5273, a=0.7921, d=1.0721, p=30.3214):

    u = np.random.rand(N)

    exp_part = np.random.exponential(scale=lam, size=N)
    gg_part = generalized_gamma(N, a, d, p)

    return np.where(u < omega, exp_part, gg_part)


def pointing_error(N, rho2=2.215, A_eq=0.786):
    u = np.random.uniform(0, 1, N)

    h_p = A_eq * (u ** (1 / rho2))

    return h_p / np.sqrt(np.mean(h_p**2))


def total_channel(N, omega=0.1953, lam=0.5273, a=0.7921, d=1.0721, p=30.3214):

    h = egg_fading(N, omega=omega, lam=lam, a=a, d=d, p=p) * pointing_error(N)

    # Normalize so E[h^2] = 1 (VERY IMPORTANT for correct SNR scaling)
    return h / np.sqrt(np.mean(h**2))


# =========================
# 3. MULTI-APERTURE (SC)
# =========================
def multi_aperture_channel(
    N_samples, N_apertures, omega=0.1953, lam=0.5273, a=0.7921, d=1.0721, p=30.3214
):

    return np.array(
        [
            total_channel(N_samples, omega=omega, lam=lam, a=a, d=d, p=p)
            for _ in range(N_apertures)
        ]
    )


def selection_combining(h):

    return np.max(h, axis=0)


# =========================
# 4. OPTICAL NOISE MODEL
# =========================
def awgn_optical(signal, snr_db):

    snr_linear = 10 ** (snr_db / 10)

    # IMPORTANT: Optical SNR scaling (noise variance = 1/SNR)
    noise_var = 1 / snr_linear

    noise = np.random.normal(0, np.sqrt(noise_var), size=signal.shape)
    return signal + noise


# =========================
# 5. BER SIMULATION (OPTICAL)
# =========================
def simulate_ber(
    N, snr_db, N_apertures=1, omega=0.1953, lam=0.5273, a=0.7921, d=1.0721, p=30.3214
):

    # Generate information bits
    bits = generate_bits(N)

    # OOK modulation: bits -> {0, 1} amplitude
    x = ook_mod(bits, A=1.0)

    # Generate fading channel (single or multiple apertures with selection combining)
    if N_apertures == 1:
        h = total_channel(N, omega=omega, lam=lam, a=a, d=d, p=p)
    else:
        h_all = multi_aperture_channel(
            N, N_apertures, omega=omega, lam=lam, a=a, d=d, p=p
        )
        h = selection_combining(h_all)

    # Channel output: multiply signal by fading gain
    y_faded = h * x

    # Add AWGN noise
    y = awgn_optical(y_faded, snr_db)

    # Threshold detection (optimal threshold is 0.5 * h for OOK)
    bits_hat = (y > (0.5 * h)).astype(int)

    # Compute and return BER
    return compute_ber(bits, bits_hat)


# =========================
# 6. MAIN EXPERIMENT
# =========================
if __name__ == "__main__":
    snrs = np.arange(0, 61, 3)

    N = 10_000_000  # HIGH accuracy for paper-quality results

    # Default EGG parameters
    omega_default = 0.2109
    lam_default = 0.4603
    a_default = 0.152
    d_default = 1.1501
    p_default = 41.3258

    ber_N1 = []
    ber_N5 = []

    print("Running BER simulation (paper-level)...\n")

    for snr in snrs:
        print(f"SNR = {snr} dB")

        # Simulate BER with default EGG parameters
        ber_N1.append(
            simulate_ber(
                N,
                snr,
                N_apertures=1,
                omega=omega_default,
                lam=lam_default,
                a=a_default,
                d=d_default,
                p=p_default,
            )
        )

        ber_N5.append(
            simulate_ber(
                N,
                snr,
                N_apertures=5,
                omega=omega_default,
                lam=lam_default,
                a=a_default,
                d=d_default,
                p=p_default,
            )
        )

    ber_N1 = np.array(ber_N1)
    ber_N5 = np.array(ber_N5)

    # =========================
    # 7. PLOT
    # =========================
    plt.figure(figsize=(8, 6))

    plt.semilogy(snrs, ber_N1, "o-", label="N=1 (no diversity)")
    plt.semilogy(snrs, ber_N5, "^-", label="N=5 (selection combining)")

    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title("BER vs SNR (EGG + Pointing Error, Selection Combining, IM/DD)")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    plt.show()
