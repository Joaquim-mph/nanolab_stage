# IV Plotting Module

Visualization tools for IV analysis results.

## Quick Start

### Option 1: Automated Runner (Recommended)
```bash
# Generate all plots
python src/ploting/IV/run_plotting.py \
  --stats-dir data/04_analysis/iv_stats/2025-09-11 \
  --hysteresis-dir data/04_analysis/hysteresis/2025-09-11
```

### Option 2: From Full Pipeline
```bash
# Analysis + plotting in one command
python src/analysis/IV/run_full_pipeline.py --date 2025-09-11
```

## Scripts

### Runner (⭐ Use This)
- **`run_plotting.py`** - Generates all visualization types

### Individual Plot Scripts
- **`plot_aggregated_iv.py`** - IV traces (forward/return with fits)
- **`compare_polynomial_orders.py`** - **⭐ MAIN FIGURE** (8-subplot comparison)
- **`explore_hysteresis.py`** - Statistical exploration (5 plots)
- **`visualize_hysteresis.py`** - Detailed per-range analysis
- **`plot_hysteresis.py`** - Basic hysteresis plots
- **`visualize_segments.py`** - Diagnostic segment visualization

## Plot Types

### 1. IV Traces (`plot_aggregated_iv.py`)
**What it shows:** Forward and return sweep data with polynomial fits

**Output directory:** `plots/{date}/iv_traces/`

**Files:**
```
iv_traces/
├── iv_aggregated_all_return_fit.png       # Overview (all V_max)
├── iv_aggregated_return_fit_vmax*.png     # Detailed per V_max
└── resistance_vs_vmax_return_fit.png      # Resistance summary
```

**When to use:** Understanding raw IV characteristics, checking fit quality

---

### 2. Hysteresis Comparison (`compare_polynomial_orders.py`) ⭐
**What it shows:** Single figure with 8 subplots comparing all polynomial orders

**Output directory:** `plots/{date}/hysteresis_comparison/`

**Files:**
```
hysteresis_comparison/
├── all_ranges_all_polynomials.png           ⭐ MAIN PUBLICATION FIGURE
├── all_ranges_all_polynomials_compact.png   # Clean version (no error bars)
└── residuals_all_ranges_all_polynomials.png # Fit quality analysis
```

**Layout:** 2×4 grid, one subplot per V_max range
**Each subplot shows:**
- Gray points: Raw hysteresis data with error bars
- Red line: Polynomial n=1
- Blue line: Polynomial n=3
- Green line: Polynomial n=5
- Purple line: Polynomial n=7

**When to use:** Publication figure, comparing polynomial fits across all ranges

---

### 3. Statistical Exploration (`explore_hysteresis.py`)
**What it shows:** Comprehensive statistical analysis across all V_max ranges

**Output directory:** `plots/{date}/hysteresis_exploration/`

**Files (5 plots):**
```
hysteresis_exploration/
├── overlay_all_ranges_poly3.png       # All ranges overlaid
├── grid_comparison_poly3.png          # Side-by-side grid
├── statistics_summary.png             # 6-panel statistical analysis
├── normalized_comparison.png          # Percentage hysteresis
└── distribution_analysis.png          # Histograms, box plots, CDFs
```

**When to use:** Deep statistical analysis, understanding trends, distributions

---

### 4. Detailed Per-Range (`visualize_hysteresis.py`)
**What it shows:** Multi-panel analysis for each V_max range individually

**Output directory:** `plots/{date}/hysteresis_detailed/`

**Files:**
```
hysteresis_detailed/
├── comprehensive_hysteresis_vmax*.png   # 3-panel view per V_max
├── polynomial_comparison_vmax*.png      # Side-by-side poly orders
└── hysteresis_heatmap_poly3.png        # Heatmap across all ranges
```

**When to use:** Detailed analysis of specific voltage ranges

---

### 5. Basic Hysteresis (`plot_hysteresis.py`)
**What it shows:** Simple hysteresis plots

**Output directory:** `plots/{date}/hysteresis_basic/`

**Files:**
```
hysteresis_basic/
├── hysteresis_raw_all.png             # Raw data overview
├── hysteresis_polynomial_all.png      # Polynomial fits overview
├── hysteresis_vmax*.png              # Individual V_max plots
└── hysteresis_vs_vmax.png            # Summary vs V_max
```

**When to use:** Quick overview, simple presentations

---

## Usage Examples

### Generate all plots
```bash
python src/ploting/IV/run_plotting.py \
  --stats-dir data/04_analysis/iv_stats/2025-09-11 \
  --hysteresis-dir data/04_analysis/hysteresis/2025-09-11
```

### Custom output directory
```bash
python src/ploting/IV/run_plotting.py \
  --stats-dir data/04_analysis/iv_stats/2025-09-11 \
  --hysteresis-dir data/04_analysis/hysteresis/2025-09-11 \
  --output-dir plots/my_analysis
```

### Select specific plot types
```bash
python src/ploting/IV/run_plotting.py \
  --stats-dir data/04_analysis/iv_stats/2025-09-11 \
  --hysteresis-dir data/04_analysis/hysteresis/2025-09-11 \
  --plots comparison exploration
```

Available types: `traces`, `comparison`, `exploration`, `detailed`, `basic`, `all`

### Use different polynomial order
```bash
python src/ploting/IV/run_plotting.py \
  --stats-dir data/04_analysis/iv_stats/2025-09-11 \
  --hysteresis-dir data/04_analysis/hysteresis/2025-09-11 \
  --poly-order 5
```

### Generate compact and residual plots
```bash
python src/ploting/IV/run_plotting.py \
  --stats-dir data/04_analysis/iv_stats/2025-09-11 \
  --hysteresis-dir data/04_analysis/hysteresis/2025-09-11 \
  --compact \
  --residuals
```

## Output Structure

```
plots/{date}/
├── iv_traces/                    # Forward/return IV traces
├── hysteresis_comparison/        # ⭐ Main publication figure
├── hysteresis_exploration/       # Statistical analysis (5 plots)
├── hysteresis_detailed/          # Per-range deep dive
└── hysteresis_basic/             # Simple overview plots
```

## Individual Script Usage

### Compare Polynomial Orders (Main Figure)
```bash
python src/ploting/IV/compare_polynomial_orders.py \
  --hysteresis-dir data/04_analysis/hysteresis/2025-09-11 \
  --output-dir plots/comparison \
  --compact --residuals
```

### Explore Hysteresis Statistics
```bash
python src/ploting/IV/explore_hysteresis.py \
  --stats-dir data/04_analysis/iv_stats/2025-09-11 \
  --hysteresis-dir data/04_analysis/hysteresis/2025-09-11 \
  --output-dir plots/exploration \
  --poly-order 3
```

### Detailed Visualization
```bash
python src/ploting/IV/visualize_hysteresis.py \
  --stats-dir data/04_analysis/iv_stats/2025-09-11 \
  --hysteresis-dir data/04_analysis/hysteresis/2025-09-11 \
  --output-dir plots/detailed \
  --poly-order 3
```

### Plot IV Traces
```bash
python src/ploting/IV/plot_aggregated_iv.py \
  --stats-dir data/04_analysis/iv_stats/2025-09-11 \
  --output-dir plots/traces
```

### Basic Hysteresis Plots
```bash
python src/ploting/IV/plot_hysteresis.py \
  --hysteresis-dir data/04_analysis/hysteresis/2025-09-11 \
  --output-dir plots/basic
```

## Which Plots Should I Use?

| Need | Recommended Script | Output |
|------|-------------------|--------|
| Publication figure | `compare_polynomial_orders.py` | Single 8-subplot comparison |
| Statistical analysis | `explore_hysteresis.py` | 5 comprehensive plots |
| Understand one V_max range | `visualize_hysteresis.py` | Detailed per-range plots |
| Quick overview | `plot_hysteresis.py` | Basic plots |
| Check raw IV data | `plot_aggregated_iv.py` | Forward/return traces |
| **Do everything** | `run_plotting.py` | All of the above |

## Important Notes

- All plots are generated at **300 DPI** (publication quality)
- **Polynomial order 3** is recommended default for exploration plots
- The **comparison plot** is the main figure for publications
- Use `--compact` flag for cleaner figures (removes error bars)
- Use `--residuals` flag to check fit quality
