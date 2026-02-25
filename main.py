import matplotlib.pyplot as plt
from params import CONFIG
from dist import get_gg_turbulence, get_pointing_error, sample_egg
from simu import calculate_outage_probability, calculate_average_ber
from avg_snr import calculate_snr_imdd


def main():
    print(f"Running Monte Carlo Simulation with {CONFIG['n_samples']:.0e} samples...")

    # --- Step A: Generate UOWC Link ---
    # 1. Get Turbulence (You can swap 'get_gg_turbulence' here)
    h_u_turb = get_gg_turbulence(CONFIG["uowc"]["gg"], CONFIG["n_samples"])
    h_u_turb_egg = sample_egg(CONFIG["uowc"]["egg"], CONFIG["n_samples"])

    # 2. Get Pointing Errors
    h_u_point = get_pointing_error(
        CONFIG["uowc"]["rho2"], CONFIG["uowc"]["A_eq"], CONFIG["n_samples"]
    )

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
    outage = calculate_outage_probability(
        h_final, CONFIG["snr_db_range"], CONFIG["threshold_snr"]
    )

    ber_curve = calculate_average_ber(h_final, CONFIG["snr_db_range"])

    avg_snr = calculate_snr_imdd(CONFIG["config"])
    print(avg_snr)

    # --- Step E: Plotting ---
    plt.figure(figsize=(10, 6))
    plt.semilogy(
        CONFIG["snr_db_range"],
        outage,
        "bo-",
        label="UOWC (GG + Pointing)",
        markeredgecolor="b",
        markerfacecolor="none",
    )
    plt.axvline(x=avg_snr, alpha=0.3)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.title(f"Outage Probability (Monte Carlo, N={CONFIG['n_samples']:.0e})")
    plt.xlabel("Average SNR (dB)")
    plt.ylabel("Outage Probability")
    plt.legend()
    plt.ylim(1e-6, 1)
    plt.xlim(0, 110)
    plt.show()

    # #plt.figure(figsize=(10, 6))
    # plt.semilogy(CONFIG['snr_db_range'], ber_curve, 'r-s', label='Average BER')
    # plt.grid(True, which="both", linestyle='--', alpha=0.5)
    # plt.xlabel('Average SNR (dB)')
    # plt.ylabel('Bit Error Rate')
    # plt.legend()
    # plt.ylim(1e-6, 1)
    # plt.xlim(0, 110)
    # plt.show()


if __name__ == "__main__":
    main()

