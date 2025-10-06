# IV Procedure - Intermediate Processing

This folder contains scripts for intermediate processing of IV (current-voltage) measurements between the staging and analysis layers.

## Quick Start

Run the complete IV preprocessing pipeline with one command:

```bash
# From project root (recommended - pure Python)
python3 process_iv.py --date 2025-09-11 --procedure IV

# Or directly from the script location
python3 src/intermediate/IV/run_iv_pipeline.py --date 2025-09-11 --procedure IV
```

### Examples

```bash
# Process all IV data for a specific date
python3 process_iv.py --date 2025-09-11 --procedure IV

# Process with hysteresis computation (forward - return traces)
python3 process_iv.py --date 2025-09-11 --procedure IV --compute-hysteresis

# Process specific chip
python3 process_iv.py --date 2025-09-11 --procedure IV --chip-number 71

# Process specific voltage range
python3 process_iv.py --date 2025-09-11 --procedure IV --v-max 8

# Custom output directory
python3 process_iv.py --date 2025-09-11 --procedure IV --output-suffix my_analysis
```

## Pipeline Steps

The pipeline automatically runs:

1. **Statistics Aggregation** (`aggregate_iv_stats.py`)
   - Detects and separates forward/return segments
   - Computes mean IV traces with standard deviations
   - Fits polynomial models (n=1,3,5,7) to return segments
   - Calculates resistance values

2. **Plot Generation** (`plot_aggregated_iv_odd.py`)
   - Creates overview plots for all voltage ranges
   - Generates detailed plots per V_max value
   - Shows polynomial fits with R² statistics
   - Plots resistance vs voltage range

3. **Hysteresis Computation** (optional, `--compute-hysteresis`)
   - Calculates hysteresis current: `I_hyst = I_forward - I_return`
   - Computes for raw data and polynomial fits (n=1,3,5,7)
   - Propagates uncertainties: `σ_hyst = sqrt(σ_fwd² + σ_ret²)`
   - Handles voltage rounding for proper alignment

## Individual Scripts

### `run_iv_pipeline.py` ⭐
Main pipeline runner - runs the complete workflow with one command.

### `iv_preprocessing_script.py`
Preprocesses IV sweep data with segment detection and filtering.

### `visualize_iv_segments.py`
Diagnostic tool to visualize IV sweep segments and verify correct detection of forward/return traces.

### `fit_return_segments.py`
Fits polynomial models to return segments of IV sweeps for resistance and characteristic extraction.

## Output Structure

```
data/04_analysis/iv_stats/{date}/
├── fit_summary.csv                    # Linear fit results
├── polynomial_fits_summary.csv        # Polynomial fit coefficients & R²
├── forward_vmax*.csv                  # Forward segment statistics
├── return_vmax*.csv                   # Return segment statistics
└── return_with_fit_vmax*.csv         # Return data with fit columns

plots/iv_stats/{date}/
├── iv_aggregated_all_return_fit.png   # Overview of all voltage ranges
├── iv_aggregated_return_fit_vmax*.png # Detailed plots per V_max
└── resistance_vs_vmax_return_fit.png  # Resistance vs voltage summary

data/04_analysis/hysteresis/{date}/     # Optional (--compute-hysteresis)
├── hysteresis_summary.csv             # Summary statistics for all V_max
└── hysteresis_vmax*.csv              # Detailed hysteresis per V_max
                                       # Columns: V, I_hyst_raw, I_hyst_poly1-7,
                                       #          I_forward, I_return, uncertainties
```

## Data Flow

```
data/02_stage/raw_measurements/proc=IV/
    ↓
src/intermediate/IV/run_iv_pipeline.py
    ↓
data/04_analysis/iv_stats/
    ↓
plots/iv_stats/
```
