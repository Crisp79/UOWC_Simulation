"""Simulation execution and caching module for the UOWC Streamlit app."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Tuple

import numpy as np
import streamlit as st

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
from simu import (
    calculate_average_ber,
    calculate_ergodic_capacity,
    calculate_outage_probability,
)


def serialize_instances_for_cache(
    instances: List[Tuple[str, int, Dict[str, Any]]],
) -> str:
    """Convert model instances to a stable JSON string for hashing/caching."""
    payload = [
        {"model": model, "index": idx, "params": params}
        for model, idx, params in instances
    ]
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def make_cache_signature(
    instances: List[Tuple[str, int, Dict[str, Any]]],
    rho2: float,
    a_eq: float,
    towc_params: Dict[str, Any],
    cfg_config: Dict[str, Any],
    n_samples: int,
    snr_db_range: List[float],
    threshold_snr: float,
) -> str:
    """Build a stable signature that uniquely identifies a simulation run."""
    payload = {
        "instances": json.loads(serialize_instances_for_cache(instances)),
        "rho2": float(rho2),
        "a_eq": float(a_eq),
        "towc": towc_params,
        "cfg_config": cfg_config,
        "n_samples": int(n_samples),
        "snr_db_range": [float(x) for x in snr_db_range],
        "threshold_snr": float(threshold_snr),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@st.cache_data(show_spinner=False)
def run_simulation_cached(
    cache_signature: str,
    instances_json: str,
    rho2: float,
    a_eq: float,
    towc_json: str,
    cfg_config_json: str,
    n_samples: int,
    snr_db_range_list: List[float],
    threshold_snr: float,
) -> Dict[str, Any]:
    """
    Run the full Monte Carlo simulation and cache the result.

    The cache_signature is not used directly in the calculation; it exists to
    force cache invalidation when the full simulation configuration changes.
    """
    _ = cache_signature

    instances = json.loads(instances_json)
    towc_params = json.loads(towc_json)
    cfg_config = json.loads(cfg_config_json)

    snr_db_range = np.array(snr_db_range_list)

    # Common pointing for UOWC instances
    h_u_point = sample_pointing(rho2, a_eq, n_samples)

    results: List[Dict[str, Any]] = []
    for inst in instances:
        model = inst["model"]
        idx = inst["index"]
        params = inst["params"]

        if model == "GG":
            h_turb = sample_gg(params, n_samples)
        elif model == "EGG":
            h_turb = sample_egg(params, n_samples)
        elif model == "EW":
            h_turb = sample_ew(params, n_samples)
        elif model == "Gamma-Gamma":
            h_turb = sample_gamma_gamma(params, n_samples)
        else:
            continue

        h = h_turb * h_u_point
        h = h / np.sqrt(np.mean(h**2))

        outage = calculate_outage_probability(h, snr_db_range, threshold_snr)
        ber = calculate_average_ber(h, snr_db_range)
        cap = calculate_ergodic_capacity(h, snr_db_range)

        results.append(
            {
                "label": f"{model}-{idx}",
                "model": model,
                "outage": outage,
                "ber": ber,
                "cap": cap,
            }
        )

    # TOWC link
    h_t_malaga = sample_malaga(towc_params["malaga"], n_samples)
    h_t_fog = sample_fog(towc_params["fog"], n_samples)
    h_t_point = sample_pointing(
        towc_params["point"]["rho2"],
        towc_params["point"]["A_eq"],
        n_samples,
    )
    h_towc = h_t_malaga * h_t_point * h_t_fog
    h_towc = h_towc / np.sqrt(np.mean(h_towc**2))
    outage_towc = calculate_outage_probability(h_towc, snr_db_range, threshold_snr)

    avg_snr = calculate_snr_imdd(cfg_config)

    return {
        "results": results,
        "outage_towc": outage_towc,
        "avg_snr": avg_snr,
    }
