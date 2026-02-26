import matplotlib.pyplot as plt

from avg_snr import calculate_snr_imdd
from dist import (
    sample_egg,
    sample_ew,
    sample_gg,
    sample_gamma_gamma,
    sample_pointing_uowc,
)
from params import CONFIG
from simu import calculate_average_ber, calculate_outage_probability


def main():
    print(f"Running Monte Carlo Simulation with {CONFIG['n_samples']:.0e} samples...")

    # --- Step A: Generate UOWC Link ---
    # 1. Get Turbulence (You can swap 'get_gg_turbulence' here)
    h_u_turb_gg = sample_gg(CONFIG["uowc"]["gg"], CONFIG["n_samples"])
    h_u_turb_egg = sample_egg(CONFIG["uowc"]["egg"], CONFIG["n_samples"])
    h_u_turb_ew = sample_ew(CONFIG["uowc"]["ew"], CONFIG["n_samples"])
    h_u_turb_gamma_gamma = sample_gamma_gamma(CONFIG["uowc"]["gamma_gamma"], CONFIG["n_samples"])

    # 2. Get Pointing Errors
    h_u_point = sample_pointing_uowc(
        CONFIG["uowc"]["rho2"], CONFIG["uowc"]["A_eq"], CONFIG["n_samples"]
    )

    # 3. Combine UOWC Channel
    h_uowc_gg = h_u_turb_gg * h_u_point
    h_uowc_egg = h_u_turb_egg * h_u_point
    h_uowc_ew = h_u_turb_ew * h_u_point
    h_uowc_gamma_gamma = h_u_turb_gamma_gamma * h_u_point

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
    h_final_gg = h_uowc_gg
    h_final_egg = h_uowc_egg
    h_final_ew = h_uowc_ew
    h_final_gamma_gamma = h_uowc_gamma_gamma

    # --- Step D: Calculate Performance ---
    outage_gg = calculate_outage_probability(
        h_final_gg, CONFIG["snr_db_range"], CONFIG["threshold_snr"]
    )
    outage_egg = calculate_outage_probability(
        h_final_egg, CONFIG["snr_db_range"], CONFIG["threshold_snr"]
    )
    outage_ew = calculate_outage_probability(
        h_final_ew, CONFIG["snr_db_range"], CONFIG["threshold_snr"]
    )
    outage_gamma_gamma = calculate_outage_probability(
        h_final_gamma_gamma, CONFIG["snr_db_range"], CONFIG["threshold_snr"]
    )

    ber_curve_gg = calculate_average_ber(h_final_gg, CONFIG["snr_db_range"])
    # Optionally, calculate BER for EGG as well if needed
    ber_curve_egg = calculate_average_ber(h_final_egg, CONFIG["snr_db_range"])
    ber_curve_ew = calculate_average_ber(h_final_ew, CONFIG["snr_db_range"])
    ber_curve_gamma_gamma = calculate_average_ber(h_final_gamma_gamma, CONFIG["snr_db_range"])

    avg_snr = calculate_snr_imdd(CONFIG["config"])
    print(avg_snr)

    # --- Step E: Plotting ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].semilogy(
        CONFIG["snr_db_range"],
        outage_gg,
        "rs-",
        label="UOWC (GG + Pointing)",
        markeredgecolor="r",
        markerfacecolor="none",
    )
    axes[0].semilogy(
        CONFIG["snr_db_range"],
        outage_egg,
        "bo-",
        label="UOWC (EGG + Pointing)",
        markeredgecolor="b",
        markerfacecolor="none",
    )
    axes[0].semilogy(
        CONFIG["snr_db_range"],
        outage_ew,
        "go-",
        label="UOWC (EW + Pointing)",
        markeredgecolor="g",
        markerfacecolor="none",
    )
    axes[0].semilogy(
        CONFIG["snr_db_range"],
        outage_gamma_gamma,
        "ys-",
        label="UOWC (Gamma Gamma + Pointing)",
        markeredgecolor="y",
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

    axes[1].semilogy(CONFIG["snr_db_range"], ber_curve_gg, "r-s", label="Average BER - GG")
    axes[1].semilogy(
        CONFIG["snr_db_range"], ber_curve_egg, "b-s", label="Average BER - EGG"
    )
    axes[1].semilogy(
        CONFIG["snr_db_range"], ber_curve_ew, "g-s", label="Average BER - EW"
    )
    axes[1].semilogy(CONFIG["snr_db_range"], ber_curve_gamma_gamma, "y-s", label="Average BER - Gamma Gamma")
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
