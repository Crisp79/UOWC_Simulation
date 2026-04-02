import matplotlib.pyplot as plt
import numpy as np

from avg_snr import calculate_snr_imdd
from dist import (
    sample_fog,
    sample_malaga,
    sample_pointing,
)
from params import CONFIG
from simu import (
    calculate_outage_probability,
)

def towc():
    h_t_malaga = sample_malaga(CONFIG["towc"]["malaga"], CONFIG["n_samples"])
    h_t_fog = sample_fog(CONFIG["towc"]["fog"], CONFIG["n_samples"])
    h_t_point = sample_pointing(
        CONFIG["towc"]["point"]["rho2"],
        CONFIG["towc"]["point"]["A_eq"],
        CONFIG["n_samples"],
    )
    
    h_towc_rtp = h_t_malaga * h_t_point * h_t_fog
    h_towc_rtp /= np.sqrt(np.mean(h_towc_rtp**2))
    h_towc_rp = h_t_point * h_t_fog
    h_towc_rp /= np.sqrt(np.mean(h_towc_rp**2))
    h_towc_rt = h_t_malaga * h_t_fog
    h_towc_rt /= np.sqrt(np.mean(h_towc_rt**2))
    
    
    outage_towc_rtp = calculate_outage_probability(
        h_towc_rtp, CONFIG["snr_db_range"], CONFIG["threshold_snr"]
    )
    outage_towc_rp = calculate_outage_probability(
        h_towc_rp, CONFIG["snr_db_range"], CONFIG["threshold_snr"]
    )
    outage_towc_rt = calculate_outage_probability(
        h_towc_rt, CONFIG["snr_db_range"], CONFIG["threshold_snr"]
    )
    
    plt.figure(figsize=(8, 6))
    plt.semilogy(
        CONFIG["snr_db_range"],
        outage_towc_rtp,
        "rs-",
        label="RTP",
        markeredgecolor="r",
        markerfacecolor="none",
    )
    plt.semilogy(
        CONFIG["snr_db_range"],
        outage_towc_rp,
        "gs-",
        label="RP",
        markeredgecolor="g",
        markerfacecolor="none",
    )
    plt.semilogy(
        CONFIG["snr_db_range"],
        outage_towc_rt,
        "bs-",
        label="RT",
        markeredgecolor="b",
        markerfacecolor="none",
    )
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.title(f"TOWC Outage Probability (Monte Carlo, N={CONFIG['n_samples']:.0e})")
    plt.xlabel("Average SNR (dB)")
    plt.ylabel("Outage Probability")
    plt.xlim(0, 110)
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    towc()