# IV Analysis Module

Statistical analysis of IV (current-voltage) sweep measurements.

## Quick Start

### Option 1: Full Pipeline (Recommended)
```bash
# Complete analysis + plotting
python src/analysis/IV/run_full_pipeline.py --date 2025-09-11
```

### Option 2: Analysis Only
```bash
# Just run the analysis steps
python src/analysis/IV/run_analysis.py --date 2025-09-11
```

### Option 3: Manual Steps
```bash
# Step 1: Aggregate statistics
python src/analysis/IV/aggregate_iv_stats.py \
  --stage-root data/02_stage/raw_measurements \
  --date 2025-09-11 \
  --procedure IV \
  --output-dir data/04_analysis/iv_stats/2025-09-11 \
  --fit-backward \
  --poly-orders 1 3 5 7

# Step 2: Compute hysteresis
python src/analysis/IV/compute_hysteresis.py \
  --stats-dir data/04_analysis/iv_stats/2025-09-11 \
  --output-dir data/04_analysis/hysteresis/2025-09-11

# Step 3: Analyze peaks
python src/analysis/IV/analyze_hysteresis_peaks.py \
  --hysteresis-dir data/04_analysis/hysteresis/2025-09-11 \
  --output-dir data/04_analysis/hysteresis_peaks/2025-09-11
```

## Scripts

### Runners (⭐ Use These)
- **`run_full_pipeline.py`** - Complete workflow (analysis + plotting)
- **`run_analysis.py`** - Analysis pipeline only (3 steps)

### Core Analysis Scripts
- **`aggregate_iv_stats.py`** - Aggregate measurements, compute polynomial fits
- **`compute_hysteresis.py`** - Calculate hysteresis (forward - return)
- **`analyze_hysteresis_peaks.py`** - Find peak locations

## Pipeline Steps

### 1. Aggregate IV Statistics
**Input:** Staged parquet files (from `02_stage/`)
**Output:** Forward/return statistics with polynomial fits
**Location:** `data/04_analysis/iv_stats/{date}/`

- Separates forward/backward sweep segments
- Computes mean ± std for each voltage point
- Fits polynomials (n=1,3,5,7) to return traces
- Partitions by V_max range (1V, 2V, 3V, etc.)

**Files generated:**
```
data/04_analysis/iv_stats/{date}/
├── fit_summary.csv                  # Linear fit results
├── polynomial_fits_summary.csv      # Polynomial coefficients & R²
├── forward_vmax*.csv                # Forward segment statistics
├── return_vmax*.csv                 # Return segment statistics
└── return_with_fit_vmax*.csv       # Return data + fit columns
```

### 2. Compute Hysteresis
**Input:** Statistics from step 1
**Output:** Hysteresis curves
**Location:** `data/04_analysis/hysteresis/{date}/`

- Calculates: `I_hyst = I_forward - I_return`
- Computes for raw data and each polynomial fit
- Propagates uncertainties: `σ_hyst = sqrt(σ_fwd² + σ_ret²)`

**Files generated:**
```
data/04_analysis/hysteresis/{date}/
├── hysteresis_summary.csv     # Summary stats for all V_max
└── hysteresis_vmax*.csv       # Detailed hysteresis per V_max
                               # Columns: V, I_hyst_raw, I_hyst_poly1-7,
                               #          I_forward, I_return, uncertainties
```

### 3. Analyze Hysteresis Peaks
**Input:** Hysteresis from step 2
**Output:** Peak locations and trends
**Location:** `data/04_analysis/hysteresis_peaks/{date}/`

- Finds voltage where maximum hysteresis occurs
- Analyzes trends across V_max ranges
- Generates peak visualization plots

**Files generated:**
```
data/04_analysis/hysteresis_peaks/{date}/
├── hysteresis_peaks.csv              # Peak locations for all methods
├── peak_summary_table.csv            # Compact summary table
├── hysteresis_with_peaks.png         # Plots with peak markers
└── peak_voltage_trends.png           # Trend analysis
```

## Examples

### Analyze specific chip
```bash
python src/analysis/IV/run_analysis.py \
  --date 2025-09-11 \
  --chip-number 71
```

### Custom output location
```bash
python src/analysis/IV/run_analysis.py \
  --date 2025-09-11 \
  --output-suffix my_test
```

### Skip peak analysis
```bash
python src/analysis/IV/run_analysis.py \
  --date 2025-09-11 \
  --skip-peaks
```

### Statistics only
```bash
python src/analysis/IV/run_analysis.py \
  --date 2025-09-11 \
  --stats-only
```

## Next Steps

After running analysis, generate plots:
```bash
python src/ploting/IV/run_plotting.py \
  --stats-dir data/04_analysis/iv_stats/2025-09-11 \
  --hysteresis-dir data/04_analysis/hysteresis/2025-09-11
```

Or use the full pipeline to do everything:
```bash
python src/analysis/IV/run_full_pipeline.py --date 2025-09-11
```

## Important Notes

- **Polynomial order 3** is recommended default (good fit, not overfit)
- **V_max partitioning**: Data auto-partitioned by maximum voltage
- **Error propagation**: Uncertainties correctly propagated through calculations
- **Idempotent**: Safe to rerun (uses same `run_id` for same inputs)
