import matplotlib.pyplot as plt
import numpy as np

from avg_snr import calculate_snr_imdd
from dist import (
    sample_egg,
    sample_ew,
    sample_fog,
    sample_gamma_gamma,
    sample_gg,
    sample_malaga,
    sample_pointing,
)
from params import CONFIG
from simu import (
    calculate_average_ber,
    calculate_ergodic_capacity,
    calculate_outage_probability,
)


def main():
    print(f"Running Monte Carlo Simulation with {CONFIG['n_samples']:.0e} samples...")

    # --- Step A: Generate UOWC Link ---
    # 1. Get Turbulence (You can swap 'get_gg_turbulence' here)
    h_u_turb_gg = sample_gg(CONFIG["uowc"]["gg"], CONFIG["n_samples"])
    h_u_turb_egg = sample_egg(CONFIG["uowc"]["egg"], CONFIG["n_samples"])
    h_u_turb_ew = sample_ew(CONFIG["uowc"]["ew"], CONFIG["n_samples"])
    h_u_turb_gamma_gamma = sample_gamma_gamma(
        CONFIG["uowc"]["gamma_gamma"], CONFIG["n_samples"]
    )

    # 2. Get Pointing Errors
    h_u_point = sample_pointing(
        CONFIG["uowc"]["rho2"], CONFIG["uowc"]["A_eq"], CONFIG["n_samples"]
    )

    # 3. Combine UOWC Channel
    h_uowc_gg = h_u_turb_gg * h_u_point
    h_uowc_egg = h_u_turb_egg * h_u_point
    h_uowc_ew = h_u_turb_ew * h_u_point
    h_uowc_gamma_gamma = h_u_turb_gamma_gamma * h_u_point
    h_uowc_gg /= np.sqrt(np.mean(h_uowc_gg**2))

    # --- Step B: Generate TOWC Link (Placeholder) ---
    h_t_malaga = sample_malaga(CONFIG["towc"]["malaga"], CONFIG["n_samples"])
    h_t_fog = sample_fog(CONFIG["towc"]["fog"], CONFIG["n_samples"])
    h_t_point = sample_pointing(
        CONFIG["towc"]["point"]["rho2"],
        CONFIG["towc"]["point"]["A_eq"],
        CONFIG["n_samples"],
    )

    h_towc = h_t_malaga * h_t_point * h_t_fog
    h_towc /= np.sqrt(np.mean(h_towc**2))
    outage_towc = calculate_outage_probability(
        h_towc, CONFIG["snr_db_range"], CONFIG["threshold_snr"]
    )

    # For now, we simulate UOWC only (Fig 2a)
    h_final_gg = h_uowc_gg
    h_final_egg = h_uowc_egg
    h_final_ew = h_uowc_ew
    h_final_gamma_gamma = h_uowc_gamma_gamma

    print(np.mean(h_uowc_gg**2))  # should be ~1
    print(np.mean(h_towc**2))  # likely NOT 1
    print(np.mean(h_final_gg**2))  # definitely NOT 1 (problem)

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

    outage_gg_towc_df = [1 - (1 - pu) * (1 - pt) for pu, pt in zip(outage_gg, outage_towc)]
    outage_egg_towc_df = [1 - (1 - pu) * (1 - pt) for pu, pt in zip(outage_egg, outage_towc)]
    outage_ew_towc_df = [1 - (1 - pu) * (1 - pt) for pu, pt in zip(outage_ew, outage_towc)]
    outage_gam_gam_towc_df = [1 - (1 - pu) * (1 - pt) for pu, pt in zip(outage_gamma_gamma, outage_towc)]

    print("UOWC outage:", outage_gg_towc_df[:5])
    print("TOWC outage:", outage_towc[:5])

    ber_curve_gg = calculate_average_ber(h_final_gg, CONFIG["snr_db_range"])
    ber_curve_egg = calculate_average_ber(h_final_egg, CONFIG["snr_db_range"])
    ber_curve_ew = calculate_average_ber(h_final_ew, CONFIG["snr_db_range"])
    ber_curve_gamma_gamma = calculate_average_ber(
        h_final_gamma_gamma, CONFIG["snr_db_range"]
    )

    ber_u = calculate_average_ber(h_uowc_gg, CONFIG["snr_db_range"])
    ber_t = calculate_average_ber(h_towc, CONFIG["snr_db_range"])

    er_cap_gg = calculate_ergodic_capacity(h_final_gg, CONFIG["snr_db_range"])
    er_cap_egg = calculate_ergodic_capacity(h_final_egg, CONFIG["snr_db_range"])
    er_cap_ew = calculate_ergodic_capacity(h_final_ew, CONFIG["snr_db_range"])
    er_cap_gamma_gamma = calculate_ergodic_capacity(h_final_gamma_gamma, CONFIG["snr_db_range"])

    avg_snr = calculate_snr_imdd(CONFIG["config"])
    print(avg_snr)

    # --- Step E: Plotting ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0,0].semilogy(
        CONFIG["snr_db_range"],
        outage_gg,
        "rs-",
        label="UOWC (GG + Pointing)",
        markeredgecolor="r",
        markerfacecolor="none",
    )
    
    axes[0,0].semilogy(
        CONFIG["snr_db_range"],
        outage_egg,
        "bo-",
        label="UOWC (EGG + Pointing)",
        markeredgecolor="b",
        markerfacecolor="none",
    )
    axes[0,0].semilogy(
        CONFIG["snr_db_range"],
        outage_ew,
        "go-",
        label="UOWC (EW + Pointing)",
        markeredgecolor="g",
        markerfacecolor="none",
    )
    axes[0,0].semilogy(
        CONFIG["snr_db_range"],
        outage_gamma_gamma,
        "ys-",
        label="UOWC (Gamma Gamma + Pointing)",
        markeredgecolor="y",
        markerfacecolor="none",
    )
    axes[0,0].axvline(x=avg_snr, alpha=0.3)
    axes[0,0].grid(True, which="both", linestyle="--", alpha=0.5)
    axes[0,0].set_title(f"Outage Probability (Monte Carlo, N={CONFIG['n_samples']:.0e})")
    axes[0,0].set_xlabel("Average SNR (dB)")
    axes[0,0].set_ylabel("Outage Probability")
    axes[0,0].set_ylim(1e-6, 1)
    axes[0,0].legend()

    axes[0,1].semilogy(
        CONFIG["snr_db_range"], ber_curve_gg, "r-s", label="Average BER - GG"
    )
    axes[0,1].semilogy(
        CONFIG["snr_db_range"], ber_curve_egg, "b-s", label="Average BER - EGG"
    )
    axes[0,1].semilogy(
        CONFIG["snr_db_range"], ber_curve_ew, "g-s", label="Average BER - EW"
    )
    axes[0,1].semilogy(
        CONFIG["snr_db_range"],
        ber_curve_gamma_gamma,
        "y-s",
        label="Average BER - Gamma Gamma",
    )
    axes[0,1].grid(True, which="both", linestyle="--", alpha=0.5)
    axes[0,1].set_title(f" Bit Error Rate (Monte Carlo, N={CONFIG['n_samples']:.0e})")
    axes[0,1].set_xlabel("Average SNR (dB)")
    axes[0,1].set_ylabel("Bit Error Rate")
    axes[0,1].set_ylim(1e-6, 1)
    axes[0,1].set_xlim(0, 110)
    axes[0,1].legend() 

    axes[1,0].plot(
        CONFIG["snr_db_range"], er_cap_gg, "r-s", label="Ergodic Capacity - GG"
    )
    axes[1,0].plot(
        CONFIG["snr_db_range"], er_cap_egg, "b-s", label="Ergodic Capacity - EGG"
    )
    axes[1,0].plot(
        CONFIG["snr_db_range"], er_cap_ew, "g-s", label="Ergodic Capacity - EW"
    )
    axes[1,0].plot(
        CONFIG["snr_db_range"], er_cap_gamma_gamma, "y-s", label="Ergodic Capacity - Gamma Gamma"
    )
    axes[1,0].grid(True, which="both", linestyle="--", alpha=0.5)
    axes[1,0].set_title(f" Ergodic Capacity (Monte Carlo, N={CONFIG['n_samples']:.0e})")
    axes[1,0].set_xlabel("Average SNR (dB)")
    axes[1,0].set_ylabel("Ergodic Capacity")
    axes[1,0].set_xlim(0, 110)
    axes[1,0].legend()
    
    axes[1,1].semilogy(
        CONFIG["snr_db_range"],
        outage_gg_towc_df,
        "rs-",
        label="UOWC (GG + Pointing + TOWC (DF))",
        markeredgecolor="r",
        markerfacecolor="none",
    )
    axes[1,1].semilogy(
        CONFIG["snr_db_range"],
        outage_egg_towc_df,
        "rs-",
        label="UOWC (EGG + Pointing + TOWC (DF))",
        markeredgecolor="g",
        markerfacecolor="none",
    )
    axes[1,1].semilogy(
        CONFIG["snr_db_range"],
        outage_ew_towc_df,
        "rs-",
        label="UOWC (EW + Pointing + TOWC (DF))",
        markeredgecolor="b",
        markerfacecolor="none",
    )
    axes[1,1].semilogy(
        CONFIG["snr_db_range"],
        outage_gam_gam_towc_df,
        "rs-",
        label="UOWC (Gamma Gamma + Pointing + TOWC (DF))",
        markeredgecolor="y",
        markerfacecolor="none",
    )
    axes[1,1].grid(True, which="both", linestyle="--", alpha=0.5)
    axes[1,1].set_title(f" UOWC + TOWC Outage Probability (Monte Carlo, N={CONFIG['n_samples']:.0e})")
    axes[1,1].set_xlabel("Average SNR (dB)")
    axes[1,1].set_ylabel("Outage Probability")
    axes[1,1].set_xlim(0, 110)
    axes[1,1].legend()
    plt.show()

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

if __name__ == "__main__":
    main()
