import streamlit as st
import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. IMPORT YOUR EXISTING MODULES ---
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
    """Loads the default parameters from your json file."""
    try:
        with open('params.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("params.json not found! Please ensure it is in the same directory.")
        return {}

def main():
    st.title("UOWC Simulation")
    st.markdown("Run to generate graphs")
    
    if 'config' not in st.session_state:
        st.session_state.config = load_defaults()
    
    cfg = st.session_state.config
    if not cfg:
        return

    # ==========================================
    # SIDEBAR: PARAMETER INPUTS
    # ==========================================
    with st.sidebar:
        st.header("Parameters")
        
        with st.expander("Global Parameters", expanded=True):
            cfg['n_samples'] = st.number_input("Samples (n_samples)", value=cfg.get('n_samples', 1000000), step=10000)
            cfg['snr_db_start'] = st.number_input("SNR Start (dB)", value=cfg.get('snr_db_start', -5))
            cfg['snr_db_stop'] = st.number_input("SNR Stop (dB)", value=cfg.get('snr_db_stop', 116))
            cfg['snr_db_step'] = st.number_input("SNR Step (dB)", value=cfg.get('snr_db_step', 10))
            cfg['threshold_snr'] = st.number_input("Threshold SNR", value=cfg.get('threshold_snr', 1.0))

        with st.expander("Link Config"):
            cfg['config']['p_tx'] = st.number_input("Transmit Power (p_tx)", value=cfg['config'].get('p_tx', 10.0))
            cfg['config']['dist'] = st.number_input("Distance (dist)", value=cfg['config'].get('dist', 30.0))
            cfg['config']['alpha'] = st.number_input("Alpha (alpha)", value=cfg['config'].get('alpha', 0.0056), format="%.4f")
            cfg['config']['sigma_t'] = st.number_input("Sigma T", value=cfg['config'].get('sigma_t', 1e-14), format="%.1e")

        with st.expander("UOWC Parameters"):
            st.markdown("**GG Model**")
            cfg['uowc']['gg']['a'] = st.number_input("gg: a", value=cfg['uowc']['gg'].get('a', 0.6302), format="%.4f")
            cfg['uowc']['gg']['d'] = st.number_input("gg: d", value=cfg['uowc']['gg'].get('d', 1.178), format="%.4f")
            cfg['uowc']['gg']['p'] = st.number_input("gg: p", value=cfg['uowc']['gg'].get('p', 0.8444), format="%.4f")
            
            st.markdown("**EGG Model**")
            cfg['uowc']['egg']['omega_1'] = st.number_input("egg: omega_1", value=cfg['uowc']['egg'].get('omega_1', 0.4589), format="%.4f")
            cfg['uowc']['egg']['lambda_1'] = st.number_input("egg: lambda_1", value=cfg['uowc']['egg'].get('lambda_1', 0.3449), format="%.4f")
            cfg['uowc']['egg']['d_over_p_1'] = st.number_input("egg: d/p_1", value=cfg['uowc']['egg'].get('d_over_p_1', 1.0421), format="%.4f")
            cfg['uowc']['egg']['a_1'] = st.number_input("egg: a_1", value=cfg['uowc']['egg'].get('a_1', 1.5768), format="%.4f")
            cfg['uowc']['egg']['p_1'] = st.number_input("egg: p_1", value=cfg['uowc']['egg'].get('p_1', 35.9424), format="%.4f")
            
            st.markdown("**EW Model**")
            cfg['uowc']['ew']['alpha'] = st.number_input("ew: alpha", value=cfg['uowc']['ew'].get('alpha', 2.5))
            cfg['uowc']['ew']['beta'] = st.number_input("ew: beta", value=cfg['uowc']['ew'].get('beta', 0.7))
            cfg['uowc']['ew']['eta'] = st.number_input("ew: eta", value=cfg['uowc']['ew'].get('eta', 0.5))
            
            st.markdown("**Gamma-Gamma Model**")
            cfg['uowc']['gamma_gamma']['alpha'] = st.number_input("gg: alpha", value=cfg['uowc']['gamma_gamma'].get('alpha', 5.0))
            cfg['uowc']['gamma_gamma']['beta'] = st.number_input("gg: beta", value=cfg['uowc']['gamma_gamma'].get('beta', 1.18))
            
            st.markdown("**Pointing / Other**")
            cfg['uowc']['rho2'] = st.number_input("uowc: rho2", value=cfg['uowc'].get('rho2', 1.0))
            cfg['uowc']['A_eq'] = st.number_input("uowc: A_eq", value=cfg['uowc'].get('A_eq', 1.0))

        with st.expander("TOWC Parameters"):
            st.markdown("**Malaga Model**")
            cfg['towc']['malaga']['alphaM'] = st.number_input("malaga: alphaM", value=cfg['towc']['malaga'].get('alphaM', 4.0))
            cfg['towc']['malaga']['betaM'] = st.number_input("malaga: betaM", value=cfg['towc']['malaga'].get('betaM', 3.0))
            cfg['towc']['malaga']['omegaM'] = st.number_input("malaga: omegaM", value=cfg['towc']['malaga'].get('omegaM', 1.0))
            cfg['towc']['malaga']['rho_loss'] = st.number_input("malaga: rho_loss", value=cfg['towc']['malaga'].get('rho_loss', 1.0))
            
            st.markdown("**Fog Model**")
            cfg['towc']['fog']['fog_k'] = st.number_input("fog: fog_k", value=cfg['towc']['fog'].get('fog_k', 2.32))
            cfg['towc']['fog']['fog_beta'] = st.number_input("fog: fog_beta", value=cfg['towc']['fog'].get('fog_beta', 13.32))
            cfg['towc']['fog']['d_T'] = st.number_input("fog: d_T", value=cfg['towc']['fog'].get('d_T', 400.0))
            
            st.markdown("**Point Settings**")
            cfg['towc']['point']['rho2'] = st.number_input("towc point: rho2", value=cfg['towc']['point'].get('rho2', 6.0))
            cfg['towc']['point']['A_eq'] = st.number_input("towc point: A_eq", value=cfg['towc']['point'].get('A_eq', 6.0))

        st.markdown("---")
        run_sim = st.button("Run", type="primary", use_container_width=True)

    # ==========================================
    # MAIN AREA: SIMULATION EXECUTION & GRAPHS
    # ==========================================
    if run_sim:
        with st.spinner(f"Running Monte Carlo Simulation with {cfg['n_samples']:.0e} samples..."):
            
            # --- Step A: Generate UOWC Link ---
            h_u_turb_gg = sample_gg(cfg["uowc"]["gg"], cfg["n_samples"])
            h_u_turb_egg = sample_egg(cfg["uowc"]["egg"], cfg["n_samples"])
            h_u_turb_ew = sample_ew(cfg["uowc"]["ew"], cfg["n_samples"])
            h_u_turb_gamma_gamma = sample_gamma_gamma(cfg["uowc"]["gamma_gamma"], cfg["n_samples"])

            h_u_point = sample_pointing(cfg["uowc"]["rho2"], cfg["uowc"]["A_eq"], cfg["n_samples"])

            h_uowc_gg = h_u_turb_gg * h_u_point
            h_uowc_egg = h_u_turb_egg * h_u_point
            h_uowc_ew = h_u_turb_ew * h_u_point
            h_uowc_gamma_gamma = h_u_turb_gamma_gamma * h_u_point
            
            h_uowc_gg /= np.sqrt(np.mean(h_uowc_gg**2))
            h_uowc_egg /= np.sqrt(np.mean(h_uowc_egg**2))
            h_uowc_ew /= np.sqrt(np.mean(h_uowc_ew**2))
            h_uowc_gamma_gamma /= np.sqrt(np.mean(h_uowc_gamma_gamma**2))

            # --- Step B: Generate TOWC Link ---
            h_t_malaga = sample_malaga(cfg["towc"]["malaga"], cfg["n_samples"])
            h_t_fog = sample_fog(cfg["towc"]["fog"], cfg["n_samples"])
            h_t_point = sample_pointing(cfg["towc"]["point"]["rho2"], cfg["towc"]["point"]["A_eq"], cfg["n_samples"])

            h_towc = h_t_malaga * h_t_point * h_t_fog
            h_towc /= np.sqrt(np.mean(h_towc**2))

            snr_db_range = np.arange(cfg["snr_db_start"], cfg["snr_db_stop"], cfg["snr_db_step"])
            threshold_snr = cfg["threshold_snr"]

            # --- Step C: Calculate Performance ---
            outage_towc = calculate_outage_probability(h_towc, snr_db_range, threshold_snr)
            outage_gg = calculate_outage_probability(h_uowc_gg, snr_db_range, threshold_snr)
            outage_egg = calculate_outage_probability(h_uowc_egg, snr_db_range, threshold_snr)
            outage_ew = calculate_outage_probability(h_uowc_ew, snr_db_range, threshold_snr)
            outage_gamma_gamma = calculate_outage_probability(h_uowc_gamma_gamma, snr_db_range, threshold_snr)

            outage_gg_towc_df = [1 - (1 - pu) * (1 - pt) for pu, pt in zip(outage_gg, outage_towc)]
            outage_egg_towc_df = [1 - (1 - pu) * (1 - pt) for pu, pt in zip(outage_egg, outage_towc)]
            outage_ew_towc_df = [1 - (1 - pu) * (1 - pt) for pu, pt in zip(outage_ew, outage_towc)]
            outage_gam_gam_towc_df = [1 - (1 - pu) * (1 - pt) for pu, pt in zip(outage_gamma_gamma, outage_towc)]

            ber_curve_gg = calculate_average_ber(h_uowc_gg, snr_db_range)
            ber_curve_egg = calculate_average_ber(h_uowc_egg, snr_db_range)
            ber_curve_ew = calculate_average_ber(h_uowc_ew, snr_db_range)
            ber_curve_gamma_gamma = calculate_average_ber(h_uowc_gamma_gamma, snr_db_range)

            er_cap_gg = calculate_ergodic_capacity(h_uowc_gg, snr_db_range)
            er_cap_egg = calculate_ergodic_capacity(h_uowc_egg, snr_db_range)
            er_cap_ew = calculate_ergodic_capacity(h_uowc_ew, snr_db_range)
            er_cap_gamma_gamma = calculate_ergodic_capacity(h_uowc_gamma_gamma, snr_db_range)

            avg_snr = calculate_snr_imdd(cfg["config"])

            st.success("Simulation Complete!")

            # ==========================================
            # PLOTLY INTERACTIVE CHARTS
            # ==========================================

            # Shared hover template for log-scale plots
            log_hover = "<b>%{fullData.name}</b><br>SNR: %{x:.1f} dB<br>Value: %{y:.2e}<extra></extra>"
            lin_hover = "<b>%{fullData.name}</b><br>SNR: %{x:.1f} dB<br>Capacity: %{y:.4f}<extra></extra>"

            marker_styles = [
                dict(symbol="square", size=7),
                dict(symbol="circle", size=7),
                dict(symbol="circle", size=7),
                dict(symbol="square", size=7),
                dict(symbol="square", size=7),
            ]
            colors = ["red", "blue", "green", "goldenrod", "crimson"]

            # ---- Plot 1: UOWC Outage Probability ----
            fig1 = go.Figure()

            datasets_outage = [
                (snr_db_range, outage_gg, "UOWC (GG + Pointing)", colors[0]),
                (snr_db_range, outage_egg, "UOWC (EGG + Pointing)", colors[1]),
                (snr_db_range, outage_ew, "UOWC (EW + Pointing)", colors[2]),
                (snr_db_range, outage_gamma_gamma, "UOWC (Gamma Gamma + Pointing)", colors[3]),
                (snr_db_range, outage_gg_towc_df, "UOWC (GG + Pointing + TOWC DF)", colors[4]),
            ]

            for i, (x, y, name, color) in enumerate(datasets_outage):
                fig1.add_trace(go.Scatter(
                    x=x, y=y, name=name,
                    mode="lines+markers",
                    line=dict(color=color),
                    marker=dict(symbol=marker_styles[i % len(marker_styles)]["symbol"],
                                size=marker_styles[i % len(marker_styles)]["size"],
                                color="rgba(0,0,0,0)",
                                line=dict(color=color, width=2)),
                    hovertemplate=log_hover,
                ))

            fig1.add_vline(x=avg_snr, line_dash="dash", line_color="grey", opacity=0.4,
                           annotation_text=f"Avg SNR = {avg_snr:.1f} dB",
                           annotation_position="top right")

            fig1.update_layout(
                title=f"Outage Probability (N={cfg['n_samples']:.0e})",
                xaxis_title="Average SNR (dB)",
                yaxis_title="Outage Probability",
                yaxis_type="log",
                yaxis=dict(range=[-6, 0]),  # 1e-6 to 1
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5),
                height=500,
            )

            # ---- Plot 2: Bit Error Rate ----
            fig2 = go.Figure()

            datasets_ber = [
                (snr_db_range, ber_curve_gg, "Average BER - GG", colors[0]),
                (snr_db_range, ber_curve_egg, "Average BER - EGG", colors[1]),
                (snr_db_range, ber_curve_ew, "Average BER - EW", colors[2]),
                (snr_db_range, ber_curve_gamma_gamma, "Average BER - Gamma Gamma", colors[3]),
            ]

            for i, (x, y, name, color) in enumerate(datasets_ber):
                fig2.add_trace(go.Scatter(
                    x=x, y=y, name=name,
                    mode="lines+markers",
                    line=dict(color=color),
                    marker=dict(symbol="square", size=7,
                                color="rgba(0,0,0,0)",
                                line=dict(color=color, width=2)),
                    hovertemplate=log_hover,
                ))

            fig2.update_layout(
                title=f"Bit Error Rate (N={cfg['n_samples']:.0e})",
                xaxis_title="Average SNR (dB)",
                yaxis_title="Bit Error Rate",
                yaxis_type="log",
                yaxis=dict(range=[-6, 0]),
                xaxis=dict(range=[0, 110]),
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
                height=500,
            )

            # ---- Plot 3: Ergodic Capacity ----
            fig3 = go.Figure()

            datasets_cap = [
                (snr_db_range, er_cap_gg, "Ergodic Capacity - GG", colors[0]),
                (snr_db_range, er_cap_egg, "Ergodic Capacity - EGG", colors[1]),
                (snr_db_range, er_cap_ew, "Ergodic Capacity - EW", colors[2]),
                (snr_db_range, er_cap_gamma_gamma, "Ergodic Capacity - Gamma Gamma", colors[3]),
            ]

            for i, (x, y, name, color) in enumerate(datasets_cap):
                fig3.add_trace(go.Scatter(
                    x=x, y=y, name=name,
                    mode="lines+markers",
                    line=dict(color=color),
                    marker=dict(symbol="square", size=7,
                                color="rgba(0,0,0,0)",
                                line=dict(color=color, width=2)),
                    hovertemplate=lin_hover,
                ))

            fig3.update_layout(
                title=f"Ergodic Capacity (N={cfg['n_samples']:.0e})",
                xaxis_title="Average SNR (dB)",
                yaxis_title="Ergodic Capacity",
                xaxis=dict(range=[0, 110]),
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
                height=500,
            )

            # ---- Plot 4: UOWC + TOWC Outage Probability ----
            fig4 = go.Figure()

            datasets_combined = [
                (snr_db_range, outage_gg_towc_df, "UOWC (GG + Pointing + TOWC DF)", colors[0]),
                (snr_db_range, outage_egg_towc_df, "UOWC (EGG + Pointing + TOWC DF)", colors[2]),
                (snr_db_range, outage_ew_towc_df, "UOWC (EW + Pointing + TOWC DF)", colors[1]),
                (snr_db_range, outage_gam_gam_towc_df, "UOWC (Gamma Gamma + Pointing + TOWC DF)", colors[3]),
            ]

            for i, (x, y, name, color) in enumerate(datasets_combined):
                fig4.add_trace(go.Scatter(
                    x=x, y=y, name=name,
                    mode="lines+markers",
                    line=dict(color=color),
                    marker=dict(symbol="square", size=7,
                                color="rgba(0,0,0,0)",
                                line=dict(color=color, width=2)),
                    hovertemplate=log_hover,
                ))

            fig4.update_layout(
                title="UOWC + TOWC Outage Probability",
                xaxis_title="Average SNR (dB)",
                yaxis_title="Outage Probability",
                yaxis_type="log",
                xaxis=dict(range=[0, 110]),
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5),
                height=500,
            )

            # ---- Render all four charts in a 2x2 grid ----
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig1, use_container_width=True)
                st.plotly_chart(fig3, use_container_width=True)
            with col2:
                st.plotly_chart(fig2, use_container_width=True)
                st.plotly_chart(fig4, use_container_width=True)


if __name__ == "__main__":
    main()