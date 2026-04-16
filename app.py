import hashlib
import json
import time

import numpy as np
import plotly.graph_objects as go
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

st.set_page_config(page_title="OWC Simulation Dashboard", page_icon="📡", layout="wide")


def load_defaults():
    """Load params.json if present, else return minimal defaults."""
    try:
        with open("params.json", "r", encoding="utf-8") as f:
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
        "malaga",
        {"alphaM": 4.0, "betaM": 3.0, "omegaM": 1.0, "rho_loss": 1.0},
    )
    cfg["towc"].setdefault("fog", {"fog_k": 2.32, "fog_beta": 13.32, "d_T": 400.0})
    cfg["towc"].setdefault("point", {"rho2": 6.0, "A_eq": 6.0})

    return cfg


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


def _serialize_instances_for_cache(instances):
    return json.dumps(
        [
            {"model": model, "index": idx, "params": params}
            for (model, idx, params) in instances
        ],
        sort_keys=True,
        separators=(",", ":"),
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
        ("GG", "gg_instances", {"a": 0.6302, "d": 1.178, "p": 0.8444}),
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
        ("EW", "ew_instances", {"alpha": 2.5, "beta": 0.7, "eta": 0.5}),
        ("Gamma-Gamma", "gamma_gamma_instances", {"alpha": 5.0, "beta": 1.18}),
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

        if count > 0 and count_key not in st.session_state:
            st.session_state[count_key] = int(count)
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
    for model_name, cfg_key, _defaults in [
        ("GG", "gg_instances", cfg["uowc"]["gg"]),
        ("EGG", "egg_instances", cfg["uowc"]["egg"]),
        ("EW", "ew_instances", cfg["uowc"]["ew"]),
        ("Gamma-Gamma", "gamma_gamma_instances", cfg["uowc"]["gamma_gamma"]),
    ]:
        instances = cfg["uowc"].get(cfg_key, [])
        for idx, inst in enumerate(instances):
            if model_name == "GG":
                params = {
                    "a": float(inst.get("a", cfg["uowc"]["gg"]["a"])),
                    "d": float(inst.get("d", cfg["uowc"]["gg"]["d"])),
                    "p": float(inst.get("p", cfg["uowc"]["gg"]["p"])),
                }
            elif model_name == "EGG":
                params = {
                    "omega_1": float(
                        inst.get("omega_1", cfg["uowc"]["egg"]["omega_1"])
                    ),
                    "lambda_1": float(
                        inst.get("lambda_1", cfg["uowc"]["egg"]["lambda_1"])
                    ),
                    "d_over_p_1": float(
                        inst.get("d_over_p_1", cfg["uowc"]["egg"]["d_over_p_1"])
                    ),
                    "a_1": float(inst.get("a_1", cfg["uowc"]["egg"]["a_1"])),
                    "p_1": float(inst.get("p_1", cfg["uowc"]["egg"]["p_1"])),
                }
            elif model_name == "EW":
                params = {
                    "alpha": float(inst.get("alpha", cfg["uowc"]["ew"]["alpha"])),
                    "beta": float(inst.get("beta", cfg["uowc"]["ew"]["beta"])),
                    "eta": float(inst.get("eta", cfg["uowc"]["ew"]["eta"])),
                }
            else:
                params = {
                    "alpha": float(
                        inst.get("alpha", cfg["uowc"]["gamma_gamma"]["alpha"])
                    ),
                    "beta": float(inst.get("beta", cfg["uowc"]["gamma_gamma"]["beta"])),
                }
            model_instances.append((model_name, idx + 1, params))

    if not model_instances:
        st.warning("No model instances defined — please add at least one instance.")
        st.stop()

    snr_db_range_list = [float(x) for x in snr_db_range.tolist()]

    cache_signature = _make_cache_signature(
        model_instances,
        cfg["uowc"]["rho2"],
        cfg["uowc"]["A_eq"],
        cfg["towc"],
        cfg["config"],
        cfg["n_samples"],
        snr_db_range_list,
        threshold_snr,
    )

    cached = _run_simulation_cached(
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

    results = cached["results"]
    outage_towc = cached["outage_towc"]
    avg_snr = cached["avg_snr"]

    st.success("Simulation complete — building plots.")

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

    fig_outage = go.Figure()
    for i, res in enumerate(results):
        color = color_cycle[i % len(color_cycle)]
        marker = marker_cycle[i % len(marker_cycle)]
        fig_outage.add_trace(
            go.Scatter(
                x=snr_db_range_list,
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
        combined = [
            1 - (1 - pu) * (1 - pt) for pu, pt in zip(res["outage"], outage_towc)
        ]
        fig_outage.add_trace(
            go.Scatter(
                x=snr_db_range_list,
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

    fig_ber = go.Figure()
    for i, res in enumerate(results):
        color = color_cycle[i % len(color_cycle)]
        marker = marker_cycle[i % len(marker_cycle)]
        fig_ber.add_trace(
            go.Scatter(
                x=snr_db_range_list,
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

    fig_cap = go.Figure()
    for i, res in enumerate(results):
        color = color_cycle[i % len(color_cycle)]
        marker = marker_cycle[i % len(marker_cycle)]
        fig_cap.add_trace(
            go.Scatter(
                x=snr_db_range_list,
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

    fig_combined = go.Figure()
    for i, res in enumerate(results):
        color = color_cycle[i % len(color_cycle)]
        marker = marker_cycle[i % len(marker_cycle)]
        combined = [
            1 - (1 - pu) * (1 - pt) for pu, pt in zip(res["outage"], outage_towc)
        ]
        fig_combined.add_trace(
            go.Scatter(
                x=snr_db_range_list,
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

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_outage, use_container_width=True)
        st.plotly_chart(fig_cap, use_container_width=True)
    with col2:
        st.plotly_chart(fig_ber, use_container_width=True)
        st.plotly_chart(fig_combined, use_container_width=True)

    sim_elapsed = time.perf_counter() - sim_start_time
    st.caption(f"Simulation runtime: {sim_elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
