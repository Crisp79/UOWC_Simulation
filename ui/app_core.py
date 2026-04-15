import hashlib
import json
import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# --- existing modules (assumed present in the repo) ---
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

# --- Setup ---
st.set_page_config(page_title="OWC Simulation Dashboard", page_icon="📡", layout="wide")


def load_defaults():
    """Load params.json if present, else return minimal defaults."""
    try:
        with open("params.json", "r") as f:
            cfg = json.load(f)
            return cfg
    except FileNotFoundError:
        # Minimal structure if params.json missing
        return {
            "n_samples": 100000,
            "snr_db_start": -5,
            "snr_db_stop": 116,
            "snr_db_step": 10,
            "threshold_snr": 1.0,
            "config": {"p_tx": 10.0, "dist": 30.0, "alpha": 0.0056, "sigma_t": 1e-14},
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
                "malaga": {"alphaM": 4.0, "betaM": 3.0, "omegaM": 1.0, "rho_loss": 1.0},
                "fog": {"fog_k": 2.32, "fog_beta": 13.32, "d_T": 400.0},
                "point": {"rho2": 6.0, "A_eq": 6.0},
            },
        }


def ensure_cfg_structure(cfg):
    """Ensure keys exist and are in expected types/shape."""
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
        "malaga", {"alphaM": 4.0, "betaM": 3.0, "omegaM": 1.0, "rho_loss": 1.0}
    )
    cfg["towc"].setdefault("fog", {"fog_k": 2.32, "fog_beta": 13.32, "d_T": 400.0})
    cfg["towc"].setdefault("point", {"rho2": 6.0, "A_eq": 6.0})


def _serialize_instances_for_cache(instances):
    # Convert instance list to a JSON-serializable, stable representation for caching
    # Instances: list of tuples (model, index, params) where params is a dict
    return json.dumps(
        [
            {"model": model, "index": idx, "params": params}
            for (model, idx, params) in instances
        ],
        sort_keys=True,
    )


def _make_cache_signature(
    instances,
    rho2,
    a_eq,
    towc_params,
    cfg_config,
    n_samples,
    snr_db_range,
    threshold_snr,
):
    payload = {
        "instances": json.loads(_serialize_instances_for_cache(instances)),
        "rho2": rho2,
        "a_eq": a_eq,
        "towc": towc_params,
        "cfg_config": cfg_config,
        "n_samples": n_samples,
        "snr_db_range": list(map(float, snr_db_range)),
        "threshold_snr": threshold_snr,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@st.cache_data(show_spinner=False)
def _run_simulation_cached(
    cache_signature,
    instances_json,
    rho2,
    a_eq,
    towc_json,
    cfg_config_json,
    n_samples,
    snr_db_range_list,
    threshold_snr,
):
    """
    Cached simulation runner.

    The first argument is a stable signature computed from the full simulation
    configuration. Streamlit uses it to reuse results across reruns.
    """
    _ = cache_signature

    instances = json.loads(instances_json)
    towc_params = json.loads(towc_json)
    cfg_config = json.loads(cfg_config_json)

    snr_db_range = np.array(snr_db_range_list)

    h_u_point = sample_pointing(rho2, a_eq, n_samples)

    results = []
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

    h_t_malaga = sample_malaga(towc_params["malaga"], n_samples)
    h_t_fog = sample_fog(towc_params["fog"], n_samples)
    h_t_point = sample_pointing(
        towc_params["point"]["rho2"], towc_params["point"]["A_eq"], n_samples
    )
    h_towc = h_t_malaga * h_t_point * h_t_fog
    h_towc = h_towc / np.sqrt(np.mean(h_towc**2))
    outage_towc = calculate_outage_probability(h_towc, snr_db_range, threshold_snr)

    avg_snr = calculate_snr_imdd(cfg_config)

    return {"results": results, "outage_towc": outage_towc, "avg_snr": avg_snr}


def main():
    st.title("UOWC Simulation — multiple instances per model")

    # Load defaults into session_state once
    if "config" not in st.session_state:
        st.session_state.config = load_defaults()
    cfg = st.session_state.config
    ensure_cfg_structure(cfg)

    # Sidebar: global / link config
    with st.sidebar:
        st.header("Global parameters")
        cfg["n_samples"] = int(
            st.number_input(
                "Samples (n_samples)",
                value=int(cfg.get("n_samples", 100000)),
                step=1000,
                min_value=1,
            )
        )
        cfg["snr_db_start"] = int(
            st.number_input("SNR Start (dB)", value=int(cfg.get("snr_db_start", -5)))
        )
        cfg["snr_db_stop"] = int(
            st.number_input("SNR Stop (dB)", value=int(cfg.get("snr_db_stop", 116)))
        )
        cfg["snr_db_step"] = int(
            st.number_input(
                "SNR Step (dB)", value=int(cfg.get("snr_db_step", 10)), min_value=1
            )
        )
        cfg["threshold_snr"] = float(
            st.number_input("Threshold SNR", value=float(cfg.get("threshold_snr", 1.0)))
        )

        st.markdown("---")
        st.header("Link config")
        cfg["config"]["p_tx"] = float(
            st.number_input(
                "Transmit Power (p_tx)", value=float(cfg["config"].get("p_tx", 10.0))
            )
        )
        cfg["config"]["dist"] = float(
            st.number_input(
                "Distance (dist)", value=float(cfg["config"].get("dist", 30.0))
            )
        )
        cfg["config"]["alpha"] = float(
            st.number_input(
                "Alpha (alpha)",
                value=float(cfg["config"].get("alpha", 0.0056)),
                format="%.6f",
            )
        )
        cfg["config"]["sigma_t"] = float(
            st.number_input(
                "Sigma T (sigma_t)",
                value=float(cfg["config"].get("sigma_t", 1e-14)),
                format="%.1e",
            )
        )

        st.markdown("---")
        st.header("UOWC model instances")

        # For each model, let user specify how many instances they want
        # and then show parameter inputs for each instance.
        model_list = ["GG", "EGG", "EW", "Gamma-Gamma"]
        instance_counts = {}
        for model in model_list:
            default_count_key = f"{model}_count"
            # default to 1 instance if previously unseen
            default_count = st.session_state.get(default_count_key, 1)
            count = st.number_input(
                f"{model} instances",
                value=int(default_count),
                min_value=0,
                max_value=20,
                step=1,
                key=default_count_key,
            )
            instance_counts[model] = int(count)

            # show per-instance parameter groups
            base_default = cfg["uowc"].get(
                model.lower().replace("-", "_"), cfg["uowc"].get("gg", {})
            )

            # For each instance, create inputs with unique keys so Streamlit can remember them
            for i in range(instance_counts[model]):
                exp_label = f"{model} instance {i + 1}"
                with st.expander(exp_label, expanded=False):
                    # Use model-specific inputs
                    if model == "GG":
                        a = st.number_input(
                            f"gg[{i}].a",
                            value=float(
                                st.session_state.get(
                                    f"gg_{i}_a", base_default.get("a", 0.6302)
                                )
                            ),
                            format="%.6f",
                            key=f"gg_{i}_a",
                        )
                        d = st.number_input(
                            f"gg[{i}].d",
                            value=float(
                                st.session_state.get(
                                    f"gg_{i}_d", base_default.get("d", 1.178)
                                )
                            ),
                            format="%.6f",
                            key=f"gg_{i}_d",
                        )
                        p = st.number_input(
                            f"gg[{i}].p",
                            value=float(
                                st.session_state.get(
                                    f"gg_{i}_p", base_default.get("p", 0.8444)
                                )
                            ),
                            format="%.6f",
                            key=f"gg_{i}_p",
                        )
                    elif model == "EGG":
                        omega_1 = st.number_input(
                            f"egg[{i}].omega_1",
                            value=float(
                                st.session_state.get(
                                    f"egg_{i}_omega_1",
                                    base_default.get("omega_1", 0.4589),
                                )
                            ),
                            format="%.6f",
                            key=f"egg_{i}_omega_1",
                        )
                        lambda_1 = st.number_input(
                            f"egg[{i}].lambda_1",
                            value=float(
                                st.session_state.get(
                                    f"egg_{i}_lambda_1",
                                    base_default.get("lambda_1", 0.3449),
                                )
                            ),
                            format="%.6f",
                            key=f"egg_{i}_lambda_1",
                        )
                        d_over_p_1 = st.number_input(
                            f"egg[{i}].d_over_p_1",
                            value=float(
                                st.session_state.get(
                                    f"egg_{i}_d_over_p_1",
                                    base_default.get("d_over_p_1", 1.0421),
                                )
                            ),
                            format="%.6f",
                            key=f"egg_{i}_d_over_p_1",
                        )
                        a_1 = st.number_input(
                            f"egg[{i}].a_1",
                            value=float(
                                st.session_state.get(
                                    f"egg_{i}_a_1", base_default.get("a_1", 1.5768)
                                )
                            ),
                            format="%.6f",
                            key=f"egg_{i}_a_1",
                        )
                        p_1 = st.number_input(
                            f"egg[{i}].p_1",
                            value=float(
                                st.session_state.get(
                                    f"egg_{i}_p_1", base_default.get("p_1", 35.9424)
                                )
                            ),
                            format="%.6f",
                            key=f"egg_{i}_p_1",
                        )
                    elif model == "EW":
                        alpha = st.number_input(
                            f"ew[{i}].alpha",
                            value=float(
                                st.session_state.get(
                                    f"ew_{i}_alpha", base_default.get("alpha", 2.5)
                                )
                            ),
                            key=f"ew_{i}_alpha",
                        )
                        beta = st.number_input(
                            f"ew[{i}].beta",
                            value=float(
                                st.session_state.get(
                                    f"ew_{i}_beta", base_default.get("beta", 0.7)
                                )
                            ),
                            key=f"ew_{i}_beta",
                        )
                        eta = st.number_input(
                            f"ew[{i}].eta",
                            value=float(
                                st.session_state.get(
                                    f"ew_{i}_eta", base_default.get("eta", 0.5)
                                )
                            ),
                            key=f"ew_{i}_eta",
                        )
                    elif model == "Gamma-Gamma":
                        alpha = st.number_input(
                            f"gam[{i}].alpha",
                            value=float(
                                st.session_state.get(
                                    f"gam_{i}_alpha", base_default.get("alpha", 5.0)
                                )
                            ),
                            key=f"gam_{i}_alpha",
                        )
                        beta = st.number_input(
                            f"gam[{i}].beta",
                            value=float(
                                st.session_state.get(
                                    f"gam_{i}_beta", base_default.get("beta", 1.18)
                                )
                            ),
                            key=f"gam_{i}_beta",
                        )

        st.markdown("---")
        st.header("Pointing / other (UOWC common)")
        cfg["uowc"]["rho2"] = float(
            st.number_input("uowc: rho2", value=float(cfg["uowc"].get("rho2", 1.0)))
        )
        cfg["uowc"]["A_eq"] = float(
            st.number_input("uowc: A_eq", value=float(cfg["uowc"].get("A_eq", 1.0)))
        )

        st.markdown("---")
        st.header("TOWC (kept available for combined plots)")
        st.subheader("Malaga")
        cfg["towc"]["malaga"]["alphaM"] = float(
            st.number_input(
                "malaga: alphaM", value=float(cfg["towc"]["malaga"].get("alphaM", 4.0))
            )
        )
        cfg["towc"]["malaga"]["betaM"] = float(
            st.number_input(
                "malaga: betaM", value=float(cfg["towc"]["malaga"].get("betaM", 3.0))
            )
        )
        cfg["towc"]["malaga"]["omegaM"] = float(
            st.number_input(
                "malaga: omegaM", value=float(cfg["towc"]["malaga"].get("omegaM", 1.0))
            )
        )
        cfg["towc"]["malaga"]["rho_loss"] = float(
            st.number_input(
                "malaga: rho_loss",
                value=float(cfg["towc"]["malaga"].get("rho_loss", 1.0)),
            )
        )

        st.subheader("Fog")
        cfg["towc"]["fog"]["fog_k"] = float(
            st.number_input(
                "fog: fog_k", value=float(cfg["towc"]["fog"].get("fog_k", 2.32))
            )
        )
        cfg["towc"]["fog"]["fog_beta"] = float(
            st.number_input(
                "fog: fog_beta", value=float(cfg["towc"]["fog"].get("fog_beta", 13.32))
            )
        )
        cfg["towc"]["fog"]["d_T"] = float(
            st.number_input(
                "fog: d_T", value=float(cfg["towc"]["fog"].get("d_T", 400.0))
            )
        )

        st.subheader("Point")
        cfg["towc"]["point"]["rho2"] = float(
            st.number_input(
                "towc point: rho2", value=float(cfg["towc"]["point"].get("rho2", 6.0))
            )
        )
        cfg["towc"]["point"]["A_eq"] = float(
            st.number_input(
                "towc point: A_eq", value=float(cfg["towc"]["point"].get("A_eq", 6.0))
            )
        )

        # Final action button
        st.markdown("---")
        run_sim = st.button("Run simulation", type="primary", use_container_width=True)

    # MAIN area
    st.write(
        "Use the sidebar to configure instances and parameters, then Run simulation to compute and plot."
    )

    if run_sim:
        sim_start_time = time.perf_counter()

        snr_db_range = np.arange(
            cfg["snr_db_start"], cfg["snr_db_stop"], cfg["snr_db_step"]
        )
        threshold_snr = cfg["threshold_snr"]

        model_instances = []
        for model in ["GG", "EGG", "EW", "Gamma-Gamma"]:
            count = int(st.session_state.get(f"{model}_count", 0))
            for i in range(count):
                params = {}
                if model == "GG":
                    params["a"] = float(
                        st.session_state.get(f"gg_{i}_a", cfg["uowc"]["gg"]["a"])
                    )
                    params["d"] = float(
                        st.session_state.get(f"gg_{i}_d", cfg["uowc"]["gg"]["d"])
                    )
                    params["p"] = float(
                        st.session_state.get(f"gg_{i}_p", cfg["uowc"]["gg"]["p"])
                    )
                elif model == "EGG":
                    params["omega_1"] = float(
                        st.session_state.get(
                            f"egg_{i}_omega_1", cfg["uowc"]["egg"]["omega_1"]
                        )
                    )
                    params["lambda_1"] = float(
                        st.session_state.get(
                            f"egg_{i}_lambda_1", cfg["uowc"]["egg"]["lambda_1"]
                        )
                    )
                    params["d_over_p_1"] = float(
                        st.session_state.get(
                            f"egg_{i}_d_over_p_1", cfg["uowc"]["egg"]["d_over_p_1"]
                        )
                    )
                    params["a_1"] = float(
                        st.session_state.get(f"egg_{i}_a_1", cfg["uowc"]["egg"]["a_1"])
                    )
                    params["p_1"] = float(
                        st.session_state.get(f"egg_{i}_p_1", cfg["uowc"]["egg"]["p_1"])
                    )
                elif model == "EW":
                    params["alpha"] = float(
                        st.session_state.get(
                            f"ew_{i}_alpha", cfg["uowc"]["ew"]["alpha"]
                        )
                    )
                    params["beta"] = float(
                        st.session_state.get(f"ew_{i}_beta", cfg["uowc"]["ew"]["beta"])
                    )
                    params["eta"] = float(
                        st.session_state.get(f"ew_{i}_eta", cfg["uowc"]["ew"]["eta"])
                    )
                elif model == "Gamma-Gamma":
                    params["alpha"] = float(
                        st.session_state.get(
                            f"gam_{i}_alpha", cfg["uowc"]["gamma_gamma"]["alpha"]
                        )
                    )
                    params["beta"] = float(
                        st.session_state.get(
                            f"gam_{i}_beta", cfg["uowc"]["gamma_gamma"]["beta"]
                        )
                    )
                model_instances.append((model, i + 1, params))

        if len(model_instances) == 0:
            st.warning(
                "No model instances defined — please create at least one instance in the sidebar."
            )
            return

        cache_signature = _make_cache_signature(
            model_instances,
            cfg["uowc"]["rho2"],
            cfg["uowc"]["A_eq"],
            cfg["towc"],
            cfg["config"],
            cfg["n_samples"],
            snr_db_range,
            threshold_snr,
        )

        cached = _run_simulation_cached(
            cache_signature,
            _serialize_instances_for_cache(model_instances),
            float(cfg["uowc"]["rho2"]),
            float(cfg["uowc"]["A_eq"]),
            json.dumps(cfg["towc"], sort_keys=True),
            json.dumps(cfg["config"], sort_keys=True),
            int(cfg["n_samples"]),
            snr_db_range.tolist(),
            float(threshold_snr),
        )

        results = cached["results"]
        outage_towc = cached["outage_towc"]
        avg_snr = cached["avg_snr"]

        st.success("Simulation complete — building plots.")

        # Plotly: create figures and add series for each instance
        color_cycle = [
            "red",
            "blue",
            "green",
            "goldenrod",
            "crimson",
            "purple",
            "orange",
            "teal",
        ]
        marker_cycle = ["square", "circle", "triangle-up", "diamond", "cross", "x"]

        # Outage probability plot (UOWC-only for each instance and dashed combined with TOWC)
        fig_outage = go.Figure()
        for i, res in enumerate(results):
            color = color_cycle[i % len(color_cycle)]
            marker = marker_cycle[i % len(marker_cycle)]
            fig_outage.add_trace(
                go.Scatter(
                    x=snr_db_range,
                    y=res["outage"],
                    name=f"{res['label']} (UOWC)",
                    mode="lines+markers",
                    line=dict(color=color),
                    marker=dict(
                        symbol=marker,
                        size=7,
                        line=dict(color=color, width=2),
                        color="rgba(0,0,0,0)",
                    ),
                    hovertemplate="<b>%{fullData.name}</b><br>SNR: %{x:.1f} dB<br>Outage: %{y:.2e}<extra></extra>",
                )
            )
            # Combined DF with TOWC
            combined = [
                1 - (1 - pu) * (1 - pt) for pu, pt in zip(res["outage"], outage_towc)
            ]
            fig_outage.add_trace(
                go.Scatter(
                    x=snr_db_range,
                    y=combined,
                    name=f"{res['label']} + TOWC (DF)",
                    mode="lines",
                    line=dict(color=color, dash="dash"),
                    hovertemplate="<b>%{fullData.name}</b><br>SNR: %{x:.1f} dB<br>Outage: %{y:.2e}<extra></extra>",
                )
            )
        fig_outage.add_vline(
            x=avg_snr,
            line_dash="dash",
            line_color="grey",
            opacity=0.4,
            annotation_text=f"Avg SNR = {avg_snr:.1f} dB",
            annotation_position="top right",
        )
        fig_outage.update_layout(
            title=f"Outage Probability (N={cfg['n_samples']:.0e})",
            xaxis_title="Average SNR (dB)",
            yaxis_title="Outage Probability",
            yaxis_type="log",
            hovermode="x unified",
            height=500,
        )

        # BER plot
        fig_ber = go.Figure()
        for i, res in enumerate(results):
            color = color_cycle[i % len(color_cycle)]
            marker = marker_cycle[i % len(marker_cycle)]
            fig_ber.add_trace(
                go.Scatter(
                    x=snr_db_range,
                    y=res["ber"],
                    name=f"{res['label']} BER",
                    mode="lines+markers",
                    line=dict(color=color),
                    marker=dict(
                        symbol=marker,
                        size=7,
                        line=dict(color=color, width=2),
                        color="rgba(0,0,0,0)",
                    ),
                    hovertemplate="<b>%{fullData.name}</b><br>SNR: %{x:.1f} dB<br>BER: %{y:.2e}<extra></extra>",
                )
            )
        fig_ber.update_layout(
            title=f"Bit Error Rate (N={cfg['n_samples']:.0e})",
            xaxis_title="Average SNR (dB)",
            yaxis_title="Bit Error Rate",
            yaxis_type="log",
            hovermode="x unified",
            height=500,
        )

        # Ergodic capacity plot
        fig_cap = go.Figure()
        for i, res in enumerate(results):
            color = color_cycle[i % len(color_cycle)]
            marker = marker_cycle[i % len(marker_cycle)]
            fig_cap.add_trace(
                go.Scatter(
                    x=snr_db_range,
                    y=res["cap"],
                    name=f"{res['label']} Capacity",
                    mode="lines+markers",
                    line=dict(color=color),
                    marker=dict(
                        symbol=marker,
                        size=7,
                        line=dict(color=color, width=2),
                        color="rgba(0,0,0,0)",
                    ),
                    hovertemplate="<b>%{fullData.name}</b><br>SNR: %{x:.1f} dB<br>Capacity: %{y:.4f}<extra></extra>",
                )
            )
        fig_cap.update_layout(
            title=f"Ergodic Capacity (N={cfg['n_samples']:.0e})",
            xaxis_title="Average SNR (dB)",
            yaxis_title="Ergodic Capacity",
            hovermode="x unified",
            height=500,
        )

        # Combined UOWC+TOWC outage (show only combined curves, one per instance)
        fig_combined = go.Figure()
        for i, res in enumerate(results):
            color = color_cycle[i % len(color_cycle)]
            marker = marker_cycle[i % len(marker_cycle)]
            combined = [
                1 - (1 - pu) * (1 - pt) for pu, pt in zip(res["outage"], outage_towc)
            ]
            fig_combined.add_trace(
                go.Scatter(
                    x=snr_db_range,
                    y=combined,
                    name=f"{res['label']} + TOWC (DF)",
                    mode="lines+markers",
                    line=dict(color=color),
                    marker=dict(
                        symbol=marker,
                        size=7,
                        line=dict(color=color, width=2),
                        color="rgba(0,0,0,0)",
                    ),
                    hovertemplate="<b>%{fullData.name}</b><br>SNR: %{x:.1f} dB<br>Outage: %{y:.2e}<extra></extra>",
                )
            )
        fig_combined.update_layout(
            title=f"UOWC + TOWC Outage Probability (N={cfg['n_samples']:.0e})",
            xaxis_title="Average SNR (dB)",
            yaxis_title="Outage Probability",
            yaxis_type="log",
            hovermode="x unified",
            height=500,
        )

        # Render charts in two columns
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_outage, use_container_width=True)
            st.plotly_chart(fig_cap, use_container_width=True)
        with col2:
            st.plotly_chart(fig_ber, use_container_width=True)
            st.plotly_chart(fig_combined, use_container_width=True)

        sim_elapsed = time.perf_counter() - sim_start_time
        st.caption(f"Simulation runtime: {sim_elapsed:.2f} seconds")

    else:
        st.info(
            "Configure instances and parameters in the sidebar, then Run simulation."
        )


if __name__ == "__main__":
    main()
