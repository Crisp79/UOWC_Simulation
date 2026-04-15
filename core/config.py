"""Configuration helpers for the UOWC Streamlit app."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PARAMS_PATH = ROOT_DIR / "params.json"


def load_defaults(params_path: str | Path = DEFAULT_PARAMS_PATH) -> Dict[str, Any]:
    """Load configuration from params.json, or return minimal defaults if missing."""
    params_path = Path(params_path)

    try:
        with params_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "n_samples": 100000,
            "snr_db_start": -5,
            "snr_db_stop": 116,
            "snr_db_step": 10,
            "threshold_snr": 1.0,
            "config": {
                "p_tx": 10.0,
                "dist": 30.0,
                "alpha": 0.0056,
                "sigma_t": 1e-14,
            },
            "uowc": {
                "gg": {"a": 0.6302, "d": 1.178, "p": 0.8444},
                "egg": {
                    "omega_1": 0.4589,
                    "lambda_1": 0.3449,
                    "d_over_p_1": 1.0421,
                    "a_1": 1.5768,
                    "p_1": 35.9424,
                },
                "ew": {"alpha": 2.5, "beta": 0.7, "eta": 0.5},
                "gamma_gamma": {"alpha": 5.0, "beta": 1.18},
                "rho2": 1.0,
                "A_eq": 1.0,
            },
            "towc": {
                "malaga": {
                    "alphaM": 4.0,
                    "betaM": 3.0,
                    "omegaM": 1.0,
                    "rho_loss": 1.0,
                },
                "fog": {"fog_k": 2.32, "fog_beta": 13.32, "d_T": 400.0},
                "point": {"rho2": 6.0, "A_eq": 6.0},
            },
        }


def ensure_cfg_structure(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure all expected configuration keys exist."""
    cfg.setdefault("n_samples", 100000)
    cfg.setdefault("snr_db_start", -5)
    cfg.setdefault("snr_db_stop", 116)
    cfg.setdefault("snr_db_step", 10)
    cfg.setdefault("threshold_snr", 1.0)

    cfg.setdefault("config", {})
    cfg["config"].setdefault("p_tx", 10.0)
    cfg["config"].setdefault("dist", 30.0)
    cfg["config"].setdefault("alpha", 0.0056)
    cfg["config"].setdefault("sigma_t", 1e-14)

    cfg.setdefault("uowc", {})
    cfg["uowc"].setdefault("gg", {"a": 0.6302, "d": 1.178, "p": 0.8444})
    cfg["uowc"].setdefault(
        "egg",
        {
            "omega_1": 0.4589,
            "lambda_1": 0.3449,
            "d_over_p_1": 1.0421,
            "a_1": 1.5768,
            "p_1": 35.9424,
        },
    )
    cfg["uowc"].setdefault("ew", {"alpha": 2.5, "beta": 0.7, "eta": 0.5})
    cfg["uowc"].setdefault("gamma_gamma", {"alpha": 5.0, "beta": 1.18})
    cfg["uowc"].setdefault("rho2", 1.0)
    cfg["uowc"].setdefault("A_eq", 1.0)

    cfg.setdefault("towc", {})
    cfg["towc"].setdefault(
        "malaga",
        {"alphaM": 4.0, "betaM": 3.0, "omegaM": 1.0, "rho_loss": 1.0},
    )
    cfg["towc"].setdefault("fog", {"fog_k": 2.32, "fog_beta": 13.32, "d_T": 400.0})
    cfg["towc"].setdefault("point", {"rho2": 6.0, "A_eq": 6.0})

    return cfg


def get_default_snr_range(cfg: Dict[str, Any]) -> list[int]:
    """Return the default SNR range as a list."""
    start = int(cfg.get("snr_db_start", -5))
    stop = int(cfg.get("snr_db_stop", 116))
    step = int(cfg.get("snr_db_step", 10))
    return list(range(start, stop, step))


def build_simulation_metadata(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return a compact metadata dictionary useful for caching/signatures."""
    return {
        "n_samples": int(cfg.get("n_samples", 100000)),
        "snr_db_start": int(cfg.get("snr_db_start", -5)),
        "snr_db_stop": int(cfg.get("snr_db_stop", 116)),
        "snr_db_step": int(cfg.get("snr_db_step", 10)),
        "threshold_snr": float(cfg.get("threshold_snr", 1.0)),
        "config": {
            "p_tx": float(cfg.get("config", {}).get("p_tx", 10.0)),
            "dist": float(cfg.get("config", {}).get("dist", 30.0)),
            "alpha": float(cfg.get("config", {}).get("alpha", 0.0056)),
            "sigma_t": float(cfg.get("config", {}).get("sigma_t", 1e-14)),
        },
    }
