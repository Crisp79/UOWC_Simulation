import json
import numpy as np


def load_config(file_path="params.json"):
    with open(file_path, "r") as f:
        config = json.load(f)

    # Reconstruct the NumPy array for SNR range
    config["snr_db_range"] = np.arange(
        config["snr_db_start"], config["snr_db_stop"], config["snr_db_step"]
    )

    return config


# Load it once for the module
CONFIG = load_config()

