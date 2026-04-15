import json
import time

import numpy as np
import streamlit as st

from core.config import ensure_cfg_structure, load_defaults
from plots.dashboard import build_dashboard_figures
from simulation.runner import make_cache_signature, run_simulation_cached

st.set_page_config(
    page_title="OWC Simulation Dashboard",
    page_icon="📡",
    layout="wide",
)


def _safe_float(value, default):
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value, default):
    try:
        return int(value)
    except Exception:
        return int(default)


def _sync_instance_list(cfg, key, defaults, count):
    current = cfg["uowc"].setdefault(key, [])
    count = max(0, int(count))

    while len(current) < count:
        if current:
            current.append(current[-1].copy())
        else:
            current.append(defaults.copy())

    if len(current) > count:
        del current[count:]


def _build_instance_inputs(cfg):
    st.header("UOWC model instances")

    model_specs = [
        (
            "GG",
            "gg_instances",
            {"a": 0.6302, "d": 1.178, "p": 0.8444},
        ),
        (
            "EGG",
            "egg_instances",
            {
                "omega_1": 0.4589,
                "lambda_1": 0.3449,
                "d_over_p_1": 1.0421,
                "a_1": 1.5768,
                "p_1": 35.9424,
            },
        ),
        (
            "EW",
            "ew_instances",
            {"alpha": 2.5, "beta": 0.7, "eta": 0.5},
        ),
        (
            "Gamma-Gamma",
            "gamma_gamma_instances",
            {"alpha": 5.0, "beta": 1.18},
        ),
    ]

    for model_name, cfg_key, defaults in model_specs:
        count_key = f"{cfg_key}_count"
        count = st.number_input(
            f"{model_name} instances",
            min_value=0,
            max_value=20,
            step=1,
            value=_safe_int(st.session_state.get(count_key, 1), 1),
            key=count_key,
        )
        _sync_instance_list(cfg, cfg_key, defaults, count)

        for idx in range(int(count)):
            inst = cfg["uowc"][cfg_key][idx]
            with st.expander(f"{model_name} instance {idx + 1}", expanded=False):
                if model_name == "GG":
                    inst["a"] = st.number_input(
                        f"gg[{idx + 1}]: a",
                        value=_safe_float(inst.get("a", 0.6302), 0.6302),
                        format="%.6f",
                        key=f"{cfg_key}_{idx}_a",
                    )
                    inst["d"] = st.number_input(
                        f"gg[{idx + 1}]: d",
                        value=_safe_float(inst.get("d", 1.178), 1.178),
                        format="%.6f",
                        key=f"{cfg_key}_{idx}_d",
                    )
                    inst["p"] = st.number_input(
                        f"gg[{idx + 1}]: p",
                        value=_safe_float(inst.get("p", 0.8444), 0.8444),
                        format="%.6f",
                        key=f"{cfg_key}_{idx}_p",
                    )
                elif model_name == "EGG":
                    inst["omega_1"] = st.number_input(
                        f"egg[{idx + 1}]: omega_1",
                        value=_safe_float(inst.get("omega_1", 0.4589), 0.4589),
                        format="%.6f",
                        key=f"{cfg_key}_{idx}_omega_1",
                    )
                    inst["lambda_1"] = st.number_input(
                        f"egg[{idx + 1}]: lambda_1",
                        value=_safe_float(inst.get("lambda_1", 0.3449), 0.3449),
                        format="%.6f",
                        key=f"{cfg_key}_{idx}_lambda_1",
                    )
                    inst["d_over_p_1"] = st.number_input(
                        f"egg[{idx + 1}]: d_over_p_1",
                        value=_safe_float(inst.get("d_over_p_1", 1.0421), 1.0421),
                        format="%.6f",
                        key=f"{cfg_key}_{idx}_d_over_p_1",
                    )
                    inst["a_1"] = st.number_input(
                        f"egg[{idx + 1}]: a_1",
                        value=_safe_float(inst.get("a_1", 1.5768), 1.5768),
                        format="%.6f",
                        key=f"{cfg_key}_{idx}_a_1",
                    )
                    inst["p_1"] = st.number_input(
                        f"egg[{idx + 1}]: p_1",
                        value=_safe_float(inst.get("p_1", 35.9424), 35.9424),
                        format="%.6f",
                        key=f"{cfg_key}_{idx}_p_1",
                    )
                elif model_name == "EW":
                    inst["alpha"] = st.number_input(
                        f"ew[{idx + 1}]: alpha",
                        value=_safe_float(inst.get("alpha", 2.5), 2.5),
                        key=f"{cfg_key}_{idx}_alpha",
                    )
                    inst["beta"] = st.number_input(
                        f"ew[{idx + 1}]: beta",
                        value=_safe_float(inst.get("beta", 0.7), 0.7),
                        key=f"{cfg_key}_{idx}_beta",
                    )
                    inst["eta"] = st.number_input(
                        f"ew[{idx + 1}]: eta",
                        value=_safe_float(inst.get("eta", 0.5), 0.5),
                        key=f"{cfg_key}_{idx}_eta",
                    )
                else:
                    inst["alpha"] = st.number_input(
                        f"gam[{idx + 1}]: alpha",
                        value=_safe_float(inst.get("alpha", 5.0), 5.0),
                        key=f"{cfg_key}_{idx}_alpha",
                    )
                    inst["beta"] = st.number_input(
                        f"gam[{idx + 1}]: beta",
                        value=_safe_float(inst.get("beta", 1.18), 1.18),
                        key=f"{cfg_key}_{idx}_beta",
                    )


def main():
    st.title("UOWC Simulation Dashboard")

    if "config" not in st.session_state:
        st.session_state.config = load_defaults()

    cfg = ensure_cfg_structure(st.session_state.config)

    with st.sidebar:
        st.header("Global parameters")

        cfg["n_samples"] = int(
            st.number_input(
                "Samples (n_samples)",
                min_value=1,
                step=1000,
                value=_safe_int(cfg.get("n_samples", 100000), 100000),
            )
        )
        cfg["snr_db_start"] = int(
            st.number_input(
                "SNR Start (dB)",
                value=_safe_int(cfg.get("snr_db_start", -5), -5),
            )
        )
        cfg["snr_db_stop"] = int(
            st.number_input(
                "SNR Stop (dB)",
                value=_safe_int(cfg.get("snr_db_stop", 116), 116),
            )
        )
        cfg["snr_db_step"] = int(
            st.number_input(
                "SNR Step (dB)",
                min_value=1,
                value=_safe_int(cfg.get("snr_db_step", 10), 10),
            )
        )
        cfg["threshold_snr"] = float(
            st.number_input(
                "Threshold SNR",
                value=_safe_float(cfg.get("threshold_snr", 1.0), 1.0),
            )
        )

        st.markdown("---")
        st.header("Link config")
        cfg["config"]["p_tx"] = float(
            st.number_input(
                "Transmit Power (p_tx)",
                value=_safe_float(cfg["config"].get("p_tx", 10.0), 10.0),
            )
        )
        cfg["config"]["dist"] = float(
            st.number_input(
                "Distance (dist)",
                value=_safe_float(cfg["config"].get("dist", 30.0), 30.0),
            )
        )
        cfg["config"]["alpha"] = float(
            st.number_input(
                "Alpha (alpha)",
                value=_safe_float(cfg["config"].get("alpha", 0.0056), 0.0056),
                format="%.6f",
            )
        )
        cfg["config"]["sigma_t"] = float(
            st.number_input(
                "Sigma T (sigma_t)",
                value=_safe_float(cfg["config"].get("sigma_t", 1e-14), 1e-14),
                format="%.1e",
            )
        )

        st.markdown("---")
        _build_instance_inputs(cfg)

        st.markdown("---")
        st.header("Pointing / common UOWC")
        cfg["uowc"]["rho2"] = float(
            st.number_input(
                "uowc: rho2",
                value=_safe_float(cfg["uowc"].get("rho2", 1.0), 1.0),
            )
        )
        cfg["uowc"]["A_eq"] = float(
            st.number_input(
                "uowc: A_eq",
                value=_safe_float(cfg["uowc"].get("A_eq", 1.0), 1.0),
            )
        )

        st.markdown("---")
        st.header("TOWC")
        cfg["towc"]["malaga"]["alphaM"] = float(
            st.number_input(
                "malaga: alphaM",
                value=_safe_float(cfg["towc"]["malaga"].get("alphaM", 4.0), 4.0),
            )
        )
        cfg["towc"]["malaga"]["betaM"] = float(
            st.number_input(
                "malaga: betaM",
                value=_safe_float(cfg["towc"]["malaga"].get("betaM", 3.0), 3.0),
            )
        )
        cfg["towc"]["malaga"]["omegaM"] = float(
            st.number_input(
                "malaga: omegaM",
                value=_safe_float(cfg["towc"]["malaga"].get("omegaM", 1.0), 1.0),
            )
        )
        cfg["towc"]["malaga"]["rho_loss"] = float(
            st.number_input(
                "malaga: rho_loss",
                value=_safe_float(cfg["towc"]["malaga"].get("rho_loss", 1.0), 1.0),
            )
        )

        cfg["towc"]["fog"]["fog_k"] = float(
            st.number_input(
                "fog: fog_k",
                value=_safe_float(cfg["towc"]["fog"].get("fog_k", 2.32), 2.32),
            )
        )
        cfg["towc"]["fog"]["fog_beta"] = float(
            st.number_input(
                "fog: fog_beta",
                value=_safe_float(cfg["towc"]["fog"].get("fog_beta", 13.32), 13.32),
            )
        )
        cfg["towc"]["fog"]["d_T"] = float(
            st.number_input(
                "fog: d_T",
                value=_safe_float(cfg["towc"]["fog"].get("d_T", 400.0), 400.0),
            )
        )

        cfg["towc"]["point"]["rho2"] = float(
            st.number_input(
                "towc point: rho2",
                value=_safe_float(cfg["towc"]["point"].get("rho2", 6.0), 6.0),
            )
        )
        cfg["towc"]["point"]["A_eq"] = float(
            st.number_input(
                "towc point: A_eq",
                value=_safe_float(cfg["towc"]["point"].get("A_eq", 6.0), 6.0),
            )
        )

        st.markdown("---")
        run_sim = st.button("Run simulation", type="primary", use_container_width=True)

    st.write(
        "Use the sidebar to configure multiple model instances, then run the simulation."
    )

    if not run_sim:
        st.info("Configure instances and parameters in the sidebar, then run.")
        return

    sim_start_time = time.perf_counter()

    snr_db_range = np.arange(
        cfg["snr_db_start"], cfg["snr_db_stop"], cfg["snr_db_step"]
    )
    threshold_snr = cfg["threshold_snr"]

    model_instances = []
    for model in ["GG", "EGG", "EW", "Gamma-Gamma"]:
        count = int(st.session_state.get(f"{model}_instances_count", 0))
        for idx in range(count):
            params = {}
            if model == "GG":
                params = {
                    "a": float(
                        st.session_state.get(
                            f"gg_instances_{idx}_a", cfg["uowc"]["gg"]["a"]
                        )
                    ),
                    "d": float(
                        st.session_state.get(
                            f"gg_instances_{idx}_d", cfg["uowc"]["gg"]["d"]
                        )
                    ),
                    "p": float(
                        st.session_state.get(
                            f"gg_instances_{idx}_p", cfg["uowc"]["gg"]["p"]
                        )
                    ),
                }
            elif model == "EGG":
                params = {
                    "omega_1": float(
                        st.session_state.get(
                            f"egg_instances_{idx}_omega_1",
                            cfg["uowc"]["egg"]["omega_1"],
                        )
                    ),
                    "lambda_1": float(
                        st.session_state.get(
                            f"egg_instances_{idx}_lambda_1",
                            cfg["uowc"]["egg"]["lambda_1"],
                        )
                    ),
                    "d_over_p_1": float(
                        st.session_state.get(
                            f"egg_instances_{idx}_d_over_p_1",
                            cfg["uowc"]["egg"]["d_over_p_1"],
                        )
                    ),
                    "a_1": float(
                        st.session_state.get(
                            f"egg_instances_{idx}_a_1", cfg["uowc"]["egg"]["a_1"]
                        )
                    ),
                    "p_1": float(
                        st.session_state.get(
                            f"egg_instances_{idx}_p_1", cfg["uowc"]["egg"]["p_1"]
                        )
                    ),
                }
            elif model == "EW":
                params = {
                    "alpha": float(
                        st.session_state.get(
                            f"ew_instances_{idx}_alpha", cfg["uowc"]["ew"]["alpha"]
                        )
                    ),
                    "beta": float(
                        st.session_state.get(
                            f"ew_instances_{idx}_beta", cfg["uowc"]["ew"]["beta"]
                        )
                    ),
                    "eta": float(
                        st.session_state.get(
                            f"ew_instances_{idx}_eta", cfg["uowc"]["ew"]["eta"]
                        )
                    ),
                }
            else:
                params = {
                    "alpha": float(
                        st.session_state.get(
                            f"gamma_gamma_instances_{idx}_alpha",
                            cfg["uowc"]["gamma_gamma"]["alpha"],
                        )
                    ),
                    "beta": float(
                        st.session_state.get(
                            f"gamma_gamma_instances_{idx}_beta",
                            cfg["uowc"]["gamma_gamma"]["beta"],
                        )
                    ),
                }
            model_instances.append((model, idx + 1, params))

    if not model_instances:
        st.warning("No model instances defined — please add at least one instance.")
        return

    snr_db_range_list = [float(x) for x in snr_db_range.tolist()]

    cache_signature = make_cache_signature(
        model_instances,
        cfg["uowc"]["rho2"],
        cfg["uowc"]["A_eq"],
        cfg["towc"],
        cfg["config"],
        cfg["n_samples"],
        snr_db_range_list,
        threshold_snr,
    )

    cached = run_simulation_cached(
        cache_signature,
        json.dumps(
            [
                {"model": model, "index": idx, "params": params}
                for model, idx, params in model_instances
            ],
            sort_keys=True,
        ),
        float(cfg["uowc"]["rho2"]),
        float(cfg["uowc"]["A_eq"]),
        json.dumps(cfg["towc"], sort_keys=True),
        json.dumps(cfg["config"], sort_keys=True),
        int(cfg["n_samples"]),
        snr_db_range_list,
        float(threshold_snr),
    )

    figures = build_dashboard_figures(
        snr_db_range=snr_db_range_list,
        results=cached["results"],
        outage_towc=cached["outage_towc"],
        avg_snr=cached["avg_snr"],
        n_samples=int(cfg["n_samples"]),
    )

    st.success("Simulation complete — building plots.")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(figures["outage"], use_container_width=True)
        st.plotly_chart(figures["capacity"], use_container_width=True)
    with col2:
        st.plotly_chart(figures["ber"], use_container_width=True)
        st.plotly_chart(figures["combined"], use_container_width=True)

    sim_elapsed = time.perf_counter() - sim_start_time
    st.caption(f"Simulation runtime: {sim_elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
