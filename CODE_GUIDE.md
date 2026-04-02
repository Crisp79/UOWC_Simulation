# Optical Wireless Communication Simulator - Code Guide

## Overview
This project simulates underwater optical wireless communication (UOWC) channels under various turbulence and environmental conditions. It uses Monte Carlo methods to evaluate communication performance metrics (outage probability and bit error rate) across different channel models.

## Project Structure

```
Simulator/
├── main.py              # Entry point - orchestrates the full simulation
├── simu.py              # Performance metrics calculation (outage, BER)
├── dist.py              # Channel fading distribution samplers
├── avg_snr.py           # Average SNR calculation for IM/DD systems
├── params.py            # Configuration loader
├── params.json          # Simulation parameters
├── environment.yml      # Conda dependencies
└── CODE_GUIDE.md        # This file
```

## File Descriptions

### main.py
**Purpose**: Main simulation orchestrator that ties everything together.

**Key Steps**:
1. Loads configuration (SNR range, turbulence parameters, etc.)
2. Generates turbulence samples for 4 different fading models (GG, EGG, EW, Gamma-Gamma)
3. Generates pointing error loss samples
4. Combines turbulence and pointing errors for complete channel model
5. Calculates outage probability and BER for each model across SNR range
6. Plots results in two subplots for comparison

**Functions**:
- `main()`: Runs the complete simulation and displays plots

**Key Variables**:
- `h_u_turb_*`: Raw turbulence channel gains (different models)
- `h_u_point`: Pointing error effects
- `h_final_*`: Combined UOWC channel gains (after turbulence × pointing errors)
- `outage_*`: Outage probability arrays for each model
- `ber_curve_*`: BER arrays for each model

### simu.py
**Purpose**: Calculates communication performance metrics.

**Functions**:

1. **`calculate_outage_probability(h_channel, snr_db_range, threshold=1.0)`**
   - Calculates probability that instantaneous SNR falls below threshold
   - For each SNR: checks how many fading samples result in SNR < threshold
   - Returns array of outage probabilities

2. **`calculate_average_ber(h_channel, snr_db_range)`**
   - Calculates average Bit Error Rate across all fading samples
   - Uses OOK (On-Off Keying) BER formula: Pe = 0.5 * erfc(√γ/2)
   - Averages instantaneous BER over all channel realizations
   - Returns array of average BER values

**Key Concepts**:
- Instantaneous SNR = Average SNR × |h|²
- Monte Carlo integration: average BER over channel realizations

### dist.py
**Purpose**: Generates random samples from various fading distribution models.

**Turbulence Models**:

1. **`sample_gg(params, n_samples)`** - Generalized Gamma
   - Parameters: a (scale), d, p (shape parameters)
   - Used for weak to moderate turbulence
   - Normalized so E[h²] = 1

2. **`sample_egg(params, num_samples)`** - Exponential-Generalized Gamma
   - Mixture distribution: exponential + GG
   - Weight parameter (omega) controls mixture ratio
   - Provides flexibility for various turbulence conditions

3. **`sample_ew(params, num_samples)`** - Exponentiated Weibull
   - Parameters: alpha (exponentiation), beta (shape), eta (scale)
   - Inverse transform sampling method
   - Flexible model for different fading scenarios

4. **`sample_gamma_gamma(params, num_samples)`** - Gamma-Gamma
   - Product of two independent Gamma variables
   - Used for strong atmospheric turbulence
   - Particularly suitable for FSO links

**Loss Models**:

5. **`sample_pointing(rho2, A_eq, n_samples)`** - Pointing Error
   - Inverse CDF method: h = A_eq × U^(1/ρ²)
   - Represents beam misalignment losses
   - Applied to all turbulence models

6. **`sample_malaga(cfg, n_samples)`** - Malaga (Terrestrial)
   - Mixture of Gamma distributions for atmospheric turbulence
   - More realistic for terrestrial free-space optical (FSO)
   - Uses `compute_malaga_params()` helper function

7. **`sample_fog(cfg, n_samples)`** - Fog Attenuation
   - Exponential transformation of Gamma distribution
   - Models fog-induced fading (not normalized - represents real loss)
   - Parameters: k (shape), beta_f (absorption), d_T (distance)

**Normalization**:
- Most distributions normalized to E[h²] = 1 (except fog) for consistent SNR scaling
- This ensures SNR values in main.py represent true channel conditions

### avg_snr.py
**Purpose**: Calculates average SNR for IM/DD (Intensity Modulation/Direct Detection) optical systems.

**Function**: `calculate_snr_imdd(params)`

**Steps**:
1. Convert Tx power from dBm to Watts: P_W = 10^((P_dBm - 30) / 10)
2. Apply Beer-Lambert absorption law: P_rx = P_tx × exp(-α × distance)
3. Calculate SNR = (P_rx)² / noise_variance
4. Convert to dB: SNR_dB = 10 × log₁₀(SNR_linear)

**Parameters**:
- `p_tx`: Transmit power (dBm)
- `dist`: Distance (meters)
- `alpha`: Absorption coefficient (m⁻¹)
- `sigma_t`: Noise PSD (A²/GHz)

**Output**: Single SNR value in dB (displayed as vertical line in plots)

### params.py
**Purpose**: Configuration management.

**Function**: `load_config(file_path="params.json")`
- Loads JSON configuration file
- Generates SNR range array from start/stop/step parameters
- Returns dictionary with all simulation parameters

**Global Variable**: `CONFIG`
- Module-level singleton loaded at import time
- Used throughout simulation for all parameters

### params.json
**Structure**:
```json
{
  "n_samples": 10000000,              // Number of Monte Carlo realizations
  "snr_db_range": {...},              // Computed SNR array
  "threshold_snr": 1.0,               // Outage threshold (dB)
  "uowc": {
    "gg": {...},                      // Generalized Gamma parameters
    "egg": {...},                     // EGG parameters
    "ew": {...},                      // Exponentiated Weibull parameters
    "gamma_gamma": {...},             // Gamma-Gamma parameters
    "rho2": 1.0,                      // Pointing error parameter
    "A_eq": 1.0                       // Pointing error amplitude
  },
  "towc": {...},                      // Terrestrial OWC parameters (future use)
  "config": {                         // IM/DD system parameters
    "p_tx": 10.0,                     // Tx power (dBm)
    "dist": 30.0,                     // Distance (meters)
    "alpha": 0.0056,                  // Absorption coefficient
    "sigma_t": 1e-14                  // Noise PSD
  }
}
```

## Simulation Workflow

```
┌─────────────────────────────────────────────┐
│ Load Configuration (params.json)            │
│ - SNR range, turbulence params, noise      │
└────────────────┬────────────────────────────┘
                 │
┌────────────────▼────────────────────────────┐
│ Generate 10M Channel Samples               │
│ - 4 turbulence models (h_turbulence)      │
│ - Pointing errors (h_point)               │
└────────────────┬────────────────────────────┘
                 │
┌────────────────▼────────────────────────────┐
│ Combine Effects: h_channel = h_turb × h_pt │
│ Complete UOWC channel model                │
└────────────────┬────────────────────────────┘
                 │
┌────────────────▼────────────────────────────┐
│ For Each SNR Value:                        │
│ - Count outage events (SNR < threshold)   │
│ - Calculate BER for all samples           │
│ - Average BER across realizations         │
└────────────────┬────────────────────────────┘
                 │
┌────────────────▼────────────────────────────┐
│ Generate 2 Plots                           │
│ - Left: Outage Probability vs SNR         │
│ - Right: BER vs SNR                       │
└─────────────────────────────────────────────┘
```

## Key Concepts

### Monte Carlo Method
- Generate many random channel realizations (10M samples)
- For each SNR point, calculate metrics over all realizations
- As samples increase, results converge to theoretical values

### Channel Normalization (E[h²] = 1)
- Most turbulence distributions normalized so average power = 1
- Ensures SNR scaling is correct: SNR_inst = SNR_avg × |h|²
- Without this, SNR interpretation would be incorrect

### Outage Probability
- Probability that instantaneous SNR drops below threshold
- At low average SNR: high outage (many deep fades)
- At high average SNR: low outage (rarely falls below threshold)

### Bit Error Rate
- OOK formula: Pe = 0.5 × erfc(√γ / 2)
- γ varies with channel fading (multipled by |h|²)
- Average BER obtained by integrating over fading distribution

### Pointing Errors
- Beam misalignment reduces received power
- Models practical challenges in aligning transmitter/receiver
- Same pointing error loss applied to all turbulence models

## Running the Simulation

```bash
# Create conda environment
conda env create -f environment.yml
conda activate sim

# Run simulation
python main.py
```

**Output**:
- Console prints: Average SNR value in dB
- Figure displays: 2 plots comparing 4 turbulence models
- Run time: ~30-60 seconds (depends on hardware and 10M samples)

## Modifying the Simulation

### Change SNR Range
Edit `params.json`:
```json
"snr_db_start": -5,
"snr_db_stop": 116,
"snr_db_step": 10
```

### Change Turbulence Model Parameters
Edit respective parameters in `params.json` under `"uowc"`

### Add TOWC (Terrestrial) Link
Uncomment TOWC sections in `main.py` to include terrestrial FSO channel

### Use Different Modulation
Modify BER formula in `simu.py`:
- BPSK: `0.5 * erfc(sqrt(gamma))`  (instead of current OOK)
- QPSK: Different formula needed
- Adjust based on modulation scheme

### Increase Accuracy
Increase `n_samples` in `params.json` (more samples = longer runtime but better accuracy)

## Dependencies

- **NumPy**: Array operations and random sampling
- **SciPy**: Special functions (erfc, gamma, comb)
- **Matplotlib**: Plotting and visualization
- Python 3.11+

## Common Issues

**Issue**: Simulation runs very slowly
- **Solution**: Reduce `n_samples` in params.json (e.g., to 1,000,000)

**Issue**: Plots don't show
- **Solution**: Add `plt.show()` call (already present in main.py)

**Issue**: Import errors
- **Solution**: Ensure conda environment is activated: `conda activate sim`

## References

This simulator implements models from optical wireless communication literature:
- Generalized Gamma, EGG, Exponentiated Weibull for underwater turbulence
- Gamma-Gamma and Malaga for atmospheric turbulence
- Standard pointing error models for beam misalignment
- Beer-Lambert law for optical absorption
- OOK BER formulas for direct detection receivers

## Next Work

- Ergodic Capacity
- Fog in TOWC
- OOK vs BPSK
- inputs for flexibility
- frontend requirements

## Extra Ideas

- Tex to fn (maybe)
-
