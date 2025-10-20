# Coercive Field Analysis with Laser Power

**Complete analysis of coercive field from backward-subtracted hysteresis with laser power calibration**

Date: 2025-10-19

---

## Overview

This document describes the coercive field analysis for September 2025 IV measurements, incorporating laser power calibration to study the photoresponse of the device.

### Device Specifications
- **Dimensions**: 100 μm × 50 μm
- **Channel length**: L = 100 μm = 0.01 cm
- **Channel area**: A = 5×10⁻⁵ cm²
- **Beam-to-sample ratio**: 5.7 (laser spot is larger than device)

### Unit Conversions
- **Electric field**: E = V / L = V × 100 V/cm
- **Current density**: J = I / A = I × 2×10⁴ A/cm²
- **Effective power**: P_eff = P_incident / 5.7

---

## Measurement Dates

### Sept 11, 2025
- **Laser**: OFF (dark measurements)
- **Voltage ranges**: 3.0-8.0 V (E-field: 300-800 V/cm)
- **Effective power**: 0 W

### Sept 29, 2025
- **Laser**: ON at 2.5 V
- **Incident power**: 51 μW (from calibration)
- **Effective power**: 9 μW (51 μW / 5.7)
- **Voltage ranges**: 3.0-8.0 V (E-field: 300-800 V/cm)

### Sept 30, 2025
- **Laser**: ON at 5.0 V
- **Incident power**: 90 μW (from calibration)
- **Effective power**: 16 μW (90 μW / 5.7)
- **Voltage ranges**: 3.0-8.0 V (E-field: 300-800 V/cm)
- **Note**: Used Sept 29 laser calibration (no calibration file for Sept 30)

---

## Coercive Field Definition

The **coercive field** (E_c) is defined as the electric field where the maximum backward-subtracted hysteresis current density occurs.

### Backward-Subtracted Hysteresis
- Subtract the max-range (8V) return fit from all voltage ranges
- This isolates the memristive switching behavior
- Analyzed separately for positive (+) and negative (-) fields

### Calculation Method
1. Load backward-subtracted data for each voltage range
2. Compute hysteresis magnitude: |J_forward - J_backward|
3. Find E-field where this magnitude is maximum
4. Report separately for E > 0 and E < 0

---

## Key Results (Polynomial Order 3)

### Dark Measurements (Sept 11, 2025)

| V_max | E_c+ (V/cm) | E_c- (V/cm) | J_max+ (A/cm²) | J_max- (A/cm²) |
|-------|-------------|-------------|----------------|----------------|
| 3.0 V | 230 | -140 | 1.93×10⁻⁴ | 2.25×10⁻⁴ |
| 4.0 V | 260 | -130 | 4.02×10⁻⁴ | 2.56×10⁻⁴ |
| 5.0 V | 330 | -370 | 5.89×10⁻⁴ | 7.90×10⁻⁴ |
| 6.0 V | 440 | -380 | 1.06×10⁻³ | 1.11×10⁻³ |
| 7.0 V | 490 | -410 | 1.81×10⁻³ | 1.42×10⁻³ |
| 8.0 V | 530 | -390 | 1.83×10⁻³ | 1.25×10⁻³ |

### Illuminated - Low Power (Sept 29, 2025, 9 μW)

| V_max | E_c+ (V/cm) | E_c- (V/cm) | J_max+ (A/cm²) | J_max- (A/cm²) |
|-------|-------------|-------------|----------------|----------------|
| 3.0 V | 190 | -220 | 9.48×10⁻⁴ | 4.53×10⁻⁴ |
| 4.0 V | 220 | -250 | 1.57×10⁻³ | 1.25×10⁻³ |
| 5.0 V | 190 | -280 | 1.99×10⁻³ | 2.34×10⁻³ |
| 6.0 V | 450 | -340 | 3.11×10⁻³ | 3.63×10⁻³ |
| 7.0 V | 540 | -440 | 5.79×10⁻³ | 5.51×10⁻³ |
| 8.0 V | 570 | -470 | 5.94×10⁻³ | 6.46×10⁻³ |

### Illuminated - High Power (Sept 30, 2025, 16 μW)

| V_max | E_c+ (V/cm) | E_c- (V/cm) | J_max+ (A/cm²) | J_max- (A/cm²) |
|-------|-------------|-------------|----------------|----------------|
| 3.0 V | 190 | -190 | 1.10×10⁻³ | 5.89×10⁻⁴ |
| 4.0 V | 220 | -240 | 2.04×10⁻³ | 1.74×10⁻³ |
| 5.0 V | 270 | -250 | 2.86×10⁻³ | 2.72×10⁻³ |
| 6.0 V | 360 | -300 | 4.30×10⁻³ | 3.40×10⁻³ |
| 7.0 V | 520 | -400 | 6.87×10⁻³ | 4.28×10⁻³ |
| 8.0 V | 540 | -450 | 7.11×10⁻³ | 5.56×10⁻³ |

---

## Observations

### Laser Power Effects

1. **Current Density Enhancement**
   - Dark (0 μW): J_max ~ 10⁻⁴ to 10⁻³ A/cm²
   - Low power (9 μW): J_max ~ 10⁻³ to 10⁻² A/cm²
   - High power (16 μW): J_max ~ 10⁻³ to 10⁻² A/cm² (slightly higher)
   - **Up to 5-7× increase in hysteresis magnitude with illumination**

2. **Coercive Field Trends**
   - Increases with V_max (higher applied fields shift switching point)
   - Asymmetry between positive and negative fields
   - Less pronounced shift with laser power compared to current enhancement

3. **Asymmetry Analysis**
   - |E_c-| generally smaller than E_c+ for dark measurements
   - Light illumination tends to increase negative coercive field magnitude
   - Suggests photoresponse affects both polarities differently

---

## Scripts and Data Files

### Analysis Scripts

1. **`src/analysis/IV/analyze_coercive_field.py`**
   - Loads backward-subtracted hysteresis data
   - Loads laser calibration curve
   - Interpolates incident power from laser voltage
   - Computes effective power (incident / 5.7)
   - Finds coercive fields for positive and negative polarities
   - Outputs CSV with all results

2. **`src/analysis/IV/visualize_coercive_field.py`**
   - Creates 3-panel comparison plots
   - Shows coercive field vs V_max, vs power, and max J vs V_max
   - Creates power dependence plots for all polynomial orders
   - Generates summary tables

### Data Files

**Input:**
- `data/04_analysis/backward_substracted_efield_sept/{date}/backward_sub_vmax*.csv`
- `data/02_stage/raw_measurements/proc=LaserCalibration/date=2025-09-29/`

**Output:**
- `data/04_analysis/coercive_field/coercive_field_analysis.csv` - Full results
- `data/04_analysis/coercive_field/coercive_field_{date}.csv` - Per-date results
- `plots/coercive_field/coercive_field_summary_poly3.csv` - Summary table

---

## Generated Plots

All plots available in `plots/coercive_field/` (PNG and PDF formats):

1. **`coercive_field_analysis_poly{1,3,5,7}.png`**
   - 3-panel comparison for each polynomial order
   - (a) Coercive field vs V_max for all dates
   - (b) Coercive field vs effective laser power
   - (c) Max hysteresis current density vs V_max

2. **`coercive_field_power_dependence.png`**
   - 4-panel plot showing first 4 voltage ranges
   - Comparison of all polynomial orders
   - Focus on laser power effects

---

## Usage Examples

### Run Complete Analysis

```bash
# Step 1: Compute coercive fields with laser power
python src/analysis/IV/analyze_coercive_field.py \
  --dates 2025-09-11 2025-09-29 2025-09-30 \
  --backward-sub-root data/04_analysis/backward_substracted_efield_sept \
  --stage-root data/02_stage/raw_measurements \
  --output-dir data/04_analysis/coercive_field \
  --voltage-ranges 3.0 4.0 5.0 6.0 7.0 8.0 \
  --poly-orders 1 3 5 7

# Step 2: Generate visualizations
python src/analysis/IV/visualize_coercive_field.py \
  --input-file data/04_analysis/coercive_field/coercive_field_analysis.csv \
  --output-dir plots/coercive_field \
  --poly-orders 1 3 5 7
```

### Load and Analyze Results

```python
import polars as pl

# Load results
df = pl.read_csv('data/04_analysis/coercive_field/coercive_field_analysis.csv')

# Filter for polynomial order 3, specific date
df_sept29_poly3 = df.filter(
    (pl.col('date') == '2025-09-29') &
    (pl.col('poly_order') == 3)
)

# Compare dark vs illuminated at 8V
df_8v = df.filter(
    (pl.col('V_max') == 8.0) &
    (pl.col('poly_order') == 3)
).select(['date', 'E_coercive_pos', 'effective_power_W'])

print(df_8v)
```

---

## Future Directions

1. **Power-Dependent Mapping**
   - Measure at more laser power values
   - Create continuous power-dependence curves

2. **Wavelength Dependence**
   - Repeat with different laser wavelengths
   - Map absorption spectrum effects

3. **Temperature Dependence**
   - Combine with temperature sweeps
   - Study thermal vs photonic effects

4. **Switching Dynamics**
   - Frequency-dependent coercive field
   - Relaxation time measurements

---

## References

- Backward-subtracted hysteresis analysis: `BACKWARD_SUBTRACTED_ANALYSIS.md`
- Laser calibration procedure: LaserCalibration measurements in raw data
- E-field and current density units: `run_sept_iv_analysis_efield.sh`

---

**Analysis completed**: 2025-10-19  
**Analyst**: Claude Code with user guidance  
**Device**: 100 μm × 50 μm channel, nanoelectronic memristive device  
**Laser**: 455 nm blue LED, calibrated power measurement
