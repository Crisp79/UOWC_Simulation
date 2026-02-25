import matplotlib.pyplot as plt

from avg_snr import calculate_snr_imdd
from dist import get_egg_turbulance, get_gg_turbulence, get_pointing_error
from params import CONFIG
from simu import calculate_average_ber, calculate_outage_probability


def main():
    print(f"Running Monte Carlo Simulation with {CONFIG['n_samples']:.0e} samples...")

    # --- Step A: Generate UOWC Link ---
    # 1. Get Turbulence (You can swap 'get_gg_turbulence' here)
    h_u_turb_gg = get_gg_turbulence(CONFIG["uowc"]["gg"], CONFIG["n_samples"])
    h_u_turb_egg = get_egg_turbulance(CONFIG["uowc"]["egg"], CONFIG["n_samples"])

    # 2. Get Pointing Errors
    h_u_point = get_pointing_error(
        CONFIG["uowc"]["rho2"], CONFIG["uowc"]["A_eq"], CONFIG["n_samples"]
    )

    # 3. Combine UOWC Channel
    h_uowc_gg = h_u_turb_gg * h_u_point
    h_uowc_egg = h_u_turb_egg * h_u_point

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
    h_final = h_uowc_gg
    h_final_egg = h_uowc_egg

    # --- Step D: Calculate Performance ---
    outage = calculate_outage_probability(
        h_final, CONFIG["snr_db_range"], CONFIG["threshold_snr"]
    )
    outage_egg = calculate_outage_probability(
        h_final_egg, CONFIG["snr_db_range"], CONFIG["threshold_snr"]
    )

    ber_curve = calculate_average_ber(h_final, CONFIG["snr_db_range"])
    # Optionally, calculate BER for EGG as well if needed
    # ber_curve_egg = calculate_average_ber(h_final_egg, CONFIG["snr_db_range"])

    avg_snr = calculate_snr_imdd(CONFIG["config"])
    print(avg_snr)

    # --- Step E: Plotting ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].semilogy(
        CONFIG["snr_db_range"],
        outage,
        "bo-",
        label="UOWC (GG + Pointing)",
        markeredgecolor="b",
        markerfacecolor="none",
    )
    axes[0].semilogy(
        CONFIG["snr_db_range"],
        outage_egg,
        "rs-",
        label="UOWC (EGG + Pointing)",
        markeredgecolor="r",
        markerfacecolor="none",
    )
    axes[0].axvline(x=avg_snr, alpha=0.3)
    axes[0].grid(True, which="both", linestyle="--", alpha=0.5)
    axes[0].set_title(f"Outage Probability (Monte Carlo, N={CONFIG['n_samples']:.0e})")
    axes[0].set_xlabel("Average SNR (dB)")
    axes[0].set_ylabel("Outage Probability")
    axes[0].set_ylim(1e-6, 1)
    axes[0].set_xlim(0, 110)
    axes[0].legend()

    axes[1].semilogy(CONFIG["snr_db_range"], ber_curve, "r-s", label="Average BER")
    axes[1].grid(True, which="both", linestyle="--", alpha=0.5)
    axes[1].set_title(f" Bit Error Rate (Monte Carlo, N={CONFIG['n_samples']:.0e})")
    axes[1].set_xlabel("Average SNR (dB)")
    axes[1].set_ylabel("Bit Error Rate")
    axes[1].set_ylim(1e-6, 1)
    axes[1].set_xlim(0, 110)
    axes[1].legend()
    plt.show()


if __name__ == "__main__":
    main()
