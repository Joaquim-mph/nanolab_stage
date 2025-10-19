# Quick Start Guide - 4-Layer Pipeline

## TL;DR - Just Run This

```bash
# Full 4-layer pipeline (preprocessing + analysis + plotting)
python run_pipeline.py --config config/examples/4layer_pipeline.json
```

That's it! The pipeline will:
1. ✅ Validate all parameters
2. ✅ Run intermediate preprocessing (segment detection - runs once)
3. ✅ Aggregate IV statistics (reads pre-segmented data - fast!)
4. ✅ Compute hysteresis
5. ✅ Analyze peaks
6. ✅ Create publication plots

**Note:** The 4-layer architecture requires preprocessing before analysis. This is faster for repeated analysis on the same date!

---

## Common Use Cases

### 1. Full 4-Layer Pipeline (Recommended)

```bash
# Use the complete 4-layer config
python run_pipeline.py --config config/examples/4layer_pipeline.json
```

This runs:
- Intermediate preprocessing (segments voltage sweeps)
- Analysis (reads pre-segmented data, computes fits)
- Plotting (creates publication figures)

### 2. Create Your Own 4-Layer Config

```bash
# Create config once
cat > config/my_4layer.json <<EOF
{
  "staging": {
    "raw_root": "data/01_raw",
    "stage_root": "data/02_stage/raw_measurements",
    "procedures_yaml": "config/procedures.yml"
  },
  "intermediate": {
    "stage_root": "data/02_stage/raw_measurements",
    "output_root": "data/03_intermediate",
    "procedure": "IV",
    "voltage_col": "Vsd (V)",
    "workers": 8
  },
  "analysis": {
    "stage_root": "data/02_stage/raw_measurements",
    "date": "2025-10-18",
    "output_base_dir": "data/04_analysis",
    "procedure": "IV",
    "poly_orders": [1, 3, 5, 7],
    "intermediate_root": "data/03_intermediate/iv_segments"
  },
  "plotting": {
    "output_dir": "plots/2025-10-18",
    "dpi": 300,
    "compact_layout": true
  },
  "run_staging": false,
  "run_intermediate": true,
  "run_analysis": true,
  "run_plotting": true
}
EOF

# Use it many times
python run_pipeline.py --config config/my_4layer.json
```

### 3. Individual Steps (4-Layer Approach)

```bash
# Step 1: Staging (if needed)
python src/staging/stage_raw_measurements.py \
  --raw-root data/01_raw \
  --stage-root data/02_stage/raw_measurements \
  --procedures-yaml config/procedures.yml \
  --workers 8

# Step 2: Preprocessing (run once per date)
python src/intermediate/IV/iv_preprocessing_script.py \
  --config config/examples/intermediate_config.json

# Step 3: Analysis (run many times - fast!)
python src/analysis/IV/aggregate_iv_stats.py \
  --intermediate-root data/03_intermediate/iv_segments \
  --date 2025-10-18 \
  --output-base-dir data/04_analysis \
  --poly-orders 1 3 5 7
```

### 4. Reanalyze Same Date (Super Fast!)

```bash
# Once preprocessing is done, analysis is ~10x faster
# Just change analysis parameters and rerun:
python src/analysis/IV/aggregate_iv_stats.py \
  --intermediate-root data/03_intermediate/iv_segments \
  --date 2025-10-18 \
  --output-base-dir data/04_analysis/v2 \
  --poly-orders 3 5 7 9
```

---

## Output Locations

After running the 4-layer pipeline:

```
data/03_intermediate/iv_segments/         # Layer 3: Pre-segmented data (NEW!)
├── proc=IV/date=2025-10-18/run_id=*/
│   ├── segment=0/part-*.parquet          # forward_negative
│   ├── segment=1/part-*.parquet          # return_negative
│   ├── segment=2/part-*.parquet          # forward_positive
│   └── segment=3/part-*.parquet          # return_positive

data/04_analysis/                          # Layer 4: Analysis results
├── iv_stats/2025-10-18_IV/
│   ├── forward_vmax*.csv                  # Forward sweep statistics
│   ├── return_with_fit_vmax*.csv          # Return sweep + polynomial fits
│   ├── fit_summary.csv                    # Fit parameters summary
│   └── polynomial_fits_summary.csv
├── hysteresis/2025-10-18_IV/
│   ├── hysteresis_vmax*.csv               # Hysteresis curves
│   └── hysteresis_summary.csv             # Peak locations
└── hysteresis_peaks/2025-10-18_IV/
    └── peaks_analysis.csv

plots/2025-10-18_IV/
├── all_ranges_all_polynomials.png          # Main comparison figure
├── all_ranges_all_polynomials_compact.png  # Compact version
└── residuals_all_ranges_all_polynomials.png # Fit quality
```

**Key benefit:** The intermediate layer (Layer 3) is created once and reused many times for analysis!

---

## Validation Examples

The pipeline validates parameters before running:

```bash
# ❌ This will fail
python run_pipeline.py --date 2025-9-11  # Wrong format
# Error: String should match pattern '^\d{4}-\d{2}-\d{2}$'

# ❌ This will fail
python run_pipeline.py --date 2025-09-11 --poly-orders 2 4 6
# Error: Polynomial order 2 should be odd

# ✅ This will work
python run_pipeline.py --date 2025-09-11 --poly-orders 1 3 5 7
```

---

## Available Options

```bash
python run_pipeline.py --help

# Key options:
--date DATE               # YYYY-MM-DD format (required unless using --config)
--procedure PROCEDURE     # IV, IVg, etc. (default: IV)
--poly-orders N [N ...]   # Polynomial orders (default: 1 3 5 7)
--dpi N                   # Plot DPI (default: 300)
--compact                 # Use compact plot layout
--residuals               # Show fit residuals
--no-hysteresis           # Skip hysteresis computation
--no-peaks                # Skip peak analysis
```

---

## Example Configs

### Minimal 4-Layer Config

`config/minimal_4layer.json`:
```json
{
  "staging": {
    "raw_root": "data/01_raw",
    "stage_root": "data/02_stage/raw_measurements",
    "procedures_yaml": "config/procedures.yml"
  },
  "intermediate": {
    "stage_root": "data/02_stage/raw_measurements",
    "output_root": "data/03_intermediate",
    "procedure": "IV",
    "voltage_col": "Vsd (V)",
    "workers": 8
  },
  "analysis": {
    "stage_root": "data/02_stage/raw_measurements",
    "date": "2025-10-18",
    "output_base_dir": "data/04_analysis",
    "intermediate_root": "data/03_intermediate/iv_segments"
  },
  "plotting": {
    "output_dir": "plots/analysis"
  },
  "run_staging": false,
  "run_intermediate": true,
  "run_analysis": true,
  "run_plotting": true
}
```

### High-Quality Publication

`config/publication_4layer.json`:
```json
{
  "staging": {
    "raw_root": "data/01_raw",
    "stage_root": "data/02_stage/raw_measurements",
    "procedures_yaml": "config/procedures.yml"
  },
  "intermediate": {
    "stage_root": "data/02_stage/raw_measurements",
    "output_root": "data/03_intermediate",
    "procedure": "IV",
    "voltage_col": "Vsd (V)",
    "workers": 8
  },
  "analysis": {
    "stage_root": "data/02_stage/raw_measurements",
    "date": "2025-10-18",
    "output_base_dir": "data/04_analysis/publication",
    "procedure": "IV",
    "poly_orders": [1, 3, 5, 7],
    "intermediate_root": "data/03_intermediate/iv_segments"
  },
  "plotting": {
    "output_dir": "plots/publication",
    "dpi": 600,
    "format": "pdf",
    "style": "publication",
    "compact_layout": true,
    "show_residuals": true
  },
  "run_staging": false,
  "run_intermediate": true,
  "run_analysis": true,
  "run_plotting": true
}
```

**Important:** `intermediate_root` is required in all 4-layer configs!

---

## Programmatic Usage

```python
#!/usr/bin/env python3
from pathlib import Path
from models.parameters import IVAnalysisParameters, PlottingParameters, PipelineParameters
import run_pipeline

# Create parameters
analysis = IVAnalysisParameters(
    stage_root=Path("data/02_stage/raw_measurements"),
    date="2025-10-18",
    output_base_dir=Path("data/04_analysis"),
    procedure="IV",
    poly_orders=[1, 3, 5, 7],
)

plotting = PlottingParameters(
    output_dir=Path("plots/my_analysis"),
    dpi=300,
)

# Run
success = run_pipeline.run_iv_analysis(analysis)
if success:
    run_pipeline.run_plotting(plotting, analysis)
```

---

## Troubleshooting

### "intermediate_root is required"
**Problem:** Analysis requires pre-segmented data but `intermediate_root` is not set.

**Solution:**
```bash
# Option 1: Run preprocessing first
python src/intermediate/IV/iv_preprocessing_script.py \
  --config config/examples/intermediate_config.json

# Option 2: Add intermediate_root to your config
# Make sure "intermediate_root": "data/03_intermediate/iv_segments" is set
```

### "No such file or directory: data/03_intermediate/..."
**Problem:** Intermediate data doesn't exist for this date.

**Solution:**
```bash
# Run preprocessing to create intermediate data
python src/intermediate/IV/iv_preprocessing_script.py \
  --config config/examples/intermediate_config.json
```

### "Config file not found"
```bash
# Check path
ls -la config/my_config.json

# Use absolute path
python run_pipeline.py --config /full/path/to/config.json
```

### "Stage root does not exist"
```bash
# Make sure data is staged first
ls -la data/02_stage/raw_measurements/

# Or run staging
python run_pipeline.py --config config/full_pipeline.json --run-all
```

### "Validation failed"
- Check date format: must be `YYYY-MM-DD`
- Check polynomial orders: must be odd numbers (1, 3, 5, 7, etc.)
- Check DPI: must be between 72 and 1200
- Check paths: use absolute paths or paths relative to project root
- **Check intermediate_root is set** for 4-layer configs

---

## Getting Help

```bash
# General help
python run_pipeline.py --help

# Script-specific help
python src/staging/stage_raw_measurements.py --help
python src/analysis/IV/aggregate_iv_stats.py --help

# See examples
python examples/use_pydantic_config.py

# Run tests
pytest tests/test_parameters.py -v
```

---

## Next Steps

1. **Try it:**
   ```bash
   python run_pipeline.py --date 2025-10-18 --procedure IV
   ```

2. **Create a config:**
   ```bash
   cp config/examples/analysis_only.json config/my_project.json
   vim config/my_project.json  # Edit for your needs
   python run_pipeline.py --config config/my_project.json
   ```

3. **Read full docs:**
   - `4LAYER_COMPLETE.md` - Complete 4-layer architecture guide
   - `FOUR_LAYER_ARCHITECTURE.md` - Architecture design details
   - `PYDANTIC_MIGRATION.md` - Detailed parameter docs

---

## One-Liner Examples

```bash
# Full 4-layer pipeline (recommended)
python run_pipeline.py --config config/examples/4layer_pipeline.json

# Test 4-layer pipeline (small dataset)
python run_pipeline.py --config config/examples/test_4layer.json

# Preprocessing only (run once per date)
python src/intermediate/IV/iv_preprocessing_script.py \
  --config config/examples/intermediate_config.json

# Reanalyze with different polynomial orders (fast!)
python src/analysis/IV/aggregate_iv_stats.py \
  --intermediate-root data/03_intermediate/iv_segments \
  --date 2025-10-18 \
  --output-base-dir data/04_analysis/v2 \
  --poly-orders 3 5 7 9
```

---

## Performance Tips

1. **Run preprocessing once per date** (slower: ~2 minutes for 7636 runs)
2. **Rerun analysis many times** (faster: ~10 seconds)
3. **Use workers=8** for preprocessing to speed up segment detection
4. **Set force=false** in preprocessing to skip already-processed runs

That's all you need to get started with the 4-layer pipeline! 🚀
