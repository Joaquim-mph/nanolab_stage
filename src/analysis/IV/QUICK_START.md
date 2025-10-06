# IV Analysis - Quick Start

## TL;DR

```bash
# One command for everything
python src/analysis/IV/run_full_pipeline.py --date 2025-09-11
```

This runs:
1. ✅ Analysis (statistics → hysteresis → peaks)
2. ✅ Plotting (all visualizations)

---

## Common Commands

### Full pipeline
```bash
python src/analysis/IV/run_full_pipeline.py --date 2025-09-11
```

### Analysis only
```bash
python src/analysis/IV/run_analysis.py --date 2025-09-11
```

### Plotting only
```bash
python src/ploting/IV/run_plotting.py \
  --stats-dir data/04_analysis/iv_stats/2025-09-11 \
  --hysteresis-dir data/04_analysis/hysteresis/2025-09-11
```

---

## Output Locations

After running the pipeline:

**Analysis data:**
- `data/04_analysis/iv_stats/{date}/` - Statistics & polynomial fits
- `data/04_analysis/hysteresis/{date}/` - Hysteresis curves
- `data/04_analysis/hysteresis_peaks/{date}/` - Peak analysis

**Plots:**
- `plots/{date}/hysteresis_comparison/all_ranges_all_polynomials.png` - ⭐ **MAIN FIGURE**
- `plots/{date}/hysteresis_exploration/` - Statistical analysis (5 plots)
- `plots/{date}/iv_traces/` - Forward/return IV traces
- `plots/{date}/hysteresis_detailed/` - Per-range detailed plots

---

## Customize

### Specific chip
```bash
python src/analysis/IV/run_full_pipeline.py \
  --date 2025-09-11 \
  --chip-number 71
```

### Custom output location
```bash
python src/analysis/IV/run_full_pipeline.py \
  --date 2025-09-11 \
  --output-suffix my_analysis
```

### Generate compact plots (no error bars)
```bash
python src/analysis/IV/run_full_pipeline.py \
  --date 2025-09-11 \
  --compact
```

### Select specific plots
```bash
python src/analysis/IV/run_full_pipeline.py \
  --date 2025-09-11 \
  --plots comparison exploration
```

Choices: `traces`, `comparison`, `exploration`, `detailed`, `basic`, `all`

---

## What Each Runner Does

| Script | What It Does |
|--------|--------------|
| `run_full_pipeline.py` | Everything (analysis + plotting) |
| `run_analysis.py` | Analysis only (3 steps) |
| `run_plotting.py` | Plotting only (5 plot types) |

---

## Pipeline Breakdown

### Analysis Steps (run_analysis.py)
1. **Aggregate statistics** - Separate forward/return, compute means, fit polynomials
2. **Compute hysteresis** - Calculate `I_hyst = I_forward - I_return`
3. **Analyze peaks** - Find voltage locations of maximum hysteresis

### Plot Types (run_plotting.py)
1. **IV traces** - Forward/return with polynomial fits
2. **Comparison** - ⭐ Main figure (8 subplots, all polynomial orders)
3. **Exploration** - Statistical analysis (5 plots)
4. **Detailed** - Per-range deep dive
5. **Basic** - Simple overview plots

---

## Need Help?

- `python src/analysis/IV/run_full_pipeline.py --help`
- `python src/analysis/IV/run_analysis.py --help`
- `python src/ploting/IV/run_plotting.py --help`

See also:
- `src/analysis/IV/README.md` - Analysis documentation
- `src/ploting/IV/README.md` - Plotting documentation
