# Pipeline Usage Guide

## Current State: Both Approaches Work!

Your pipeline can be run in **two ways**:

### ✅ Option 1: Original Method (Still Works!)

The existing scripts work exactly as before with `argparse`:

```bash
# Step 1: Stage raw data
python src/staging/stage_raw_measurements.py \
  --raw-root data/01_raw \
  --stage-root data/02_stage/raw_measurements \
  --procedures-yaml config/procedures.yml \
  --workers 8 \
  --force

# Step 2: Aggregate IV statistics
python src/analysis/IV/aggregate_iv_stats.py \
  --stage-root data/02_stage/raw_measurements \
  --date 2025-09-11 \
  --procedure IV \
  --output-dir data/04_analysis/iv_stats

# Step 3: Compute hysteresis
python src/analysis/IV/compute_hysteresis.py \
  --stats-dir data/04_analysis/iv_stats \
  --output-dir data/04_analysis/hysteresis

# Step 4: Plot results
python src/ploting/IV/compare_polynomial_orders.py \
  --hysteresis-dir data/04_analysis/hysteresis \
  --output-dir plots \
  --compact --residuals
```

**Status:** ✅ Fully functional, no changes needed

---

### ✨ Option 2: New Pydantic Method (Optional, More Professional)

Use validated configuration with the new Pydantic models:

#### Method 2A: From JSON Config

```bash
# Create/edit config file
cat > config/my_pipeline.json <<EOF
{
  "staging": {
    "raw_root": "data/01_raw",
    "stage_root": "data/02_stage/raw_measurements",
    "procedures_yaml": "config/procedures.yml",
    "workers": 8,
    "force": true
  },
  "analysis": {
    "stage_root": "data/02_stage/raw_measurements",
    "date": "2025-09-11",
    "output_base_dir": "data/04_analysis",
    "procedure": "IV",
    "poly_orders": [1, 3, 5, 7],
    "compute_hysteresis": true,
    "analyze_peaks": true
  },
  "plotting": {
    "output_dir": "plots/2025-09-11",
    "dpi": 300,
    "compact_layout": true,
    "show_residuals": true
  },
  "run_staging": true,
  "run_analysis": true,
  "run_plotting": true
}
EOF

# Run entire pipeline with one command
python run_pipeline_with_pydantic.py --config config/my_pipeline.json
```

#### Method 2B: Programmatic (Quick Analysis)

```bash
# Run analysis + plotting only (data already staged)
python run_pipeline_with_pydantic.py \
  --date 2025-09-11 \
  --procedure IV \
  --workers 8 \
  --poly-orders 1 3 5 7 \
  --dpi 300
```

**Status:** ✨ New addition, provides validation and easier configuration

---

## Comparison

| Feature | Original (argparse) | New (Pydantic) |
|---------|-------------------|----------------|
| **Still works?** | ✅ Yes | ✅ Yes |
| **Breaking changes?** | ❌ No | ❌ No |
| **Validation** | Runtime only | Before execution |
| **Type safety** | No | Yes |
| **JSON config** | No | Yes |
| **Error messages** | Basic | Detailed |
| **Required changes** | None | Optional |
| **Learning curve** | Familiar | Small |

---

## What Changed?

### Nothing Broke!

✅ All existing scripts work as before
✅ `stage_raw_measurements.py` - unchanged, fully functional
✅ `aggregate_iv_stats.py` - unchanged, fully functional
✅ `compute_hysteresis.py` - unchanged, fully functional
✅ All plotting scripts - unchanged, fully functional

### What Was Added?

✨ **New optional tools:**
1. `src/models/parameters.py` - Pydantic validation models
2. `run_pipeline_with_pydantic.py` - Wrapper script (optional)
3. `tests/test_parameters.py` - Validation tests
4. `examples/use_pydantic_config.py` - Usage examples
5. JSON config templates in `config/examples/`

---

## Can I Run the Full Pipeline Right Now?

### Yes! Two ways:

#### Quick Test (Original Method)

```bash
# Just run staging to verify everything works
python src/staging/stage_raw_measurements.py \
  --raw-root data/01_raw \
  --stage-root data/02_stage/raw_measurements \
  --procedures-yaml config/procedures.yml \
  --workers 4
```

#### Quick Test (New Pydantic Method)

```bash
# If you already have staged data, run analysis
python run_pipeline_with_pydantic.py \
  --date 2025-09-11 \
  --procedure IV
```

---

## Migration Path (Optional)

You can adopt Pydantic **gradually**:

### Phase 1: Keep Using Original Scripts (Current)
- No changes needed
- Everything works as before

### Phase 2: Use Pydantic for Configuration (Optional)
- Create JSON configs for common workflows
- Use `run_pipeline_with_pydantic.py` wrapper
- Original scripts still available

### Phase 3: Integrate Pydantic Into Scripts (Future)
- Modify individual scripts to accept Pydantic models
- Example:
  ```python
  # Old
  def aggregate_iv_stats(stage_root, date, output_dir, ...):
      ...

  # New (optional refactor)
  def aggregate_iv_stats(params: IVAnalysisParameters):
      stage_root = params.stage_root
      date = params.date
      output_dir = params.get_stats_dir()
      ...
  ```

---

## Recommended Workflow

### For Quick Analysis (Use Original)

```bash
# You know the commands, just run them
python src/analysis/IV/aggregate_iv_stats.py --date 2025-09-11 ...
```

### For Reproducible Pipelines (Use Pydantic)

```bash
# Create config once
vim config/project_X.json

# Run many times with consistent settings
python run_pipeline_with_pydantic.py --config config/project_X.json
```

### For Automation (Use Pydantic)

```python
# In your automation script
from models.parameters import PipelineParameters

params = PipelineParameters.from_json("config/production.json")
# Validation happens here - fails fast if config is bad

# Run pipeline with validated config
...
```

---

## Testing the New System

### 1. Verify Original Scripts Work

```bash
python src/staging/stage_raw_measurements.py --help
# Should show help message - confirms script works
```

### 2. Test Pydantic Validation

```bash
# Run validation tests
pytest tests/test_parameters.py -v

# See validation in action
python examples/use_pydantic_config.py
```

### 3. Try the Wrapper Script

```bash
# Analysis + plotting only (assumes data already staged)
python run_pipeline_with_pydantic.py \
  --date 2025-09-11 \
  --procedure IV
```

---

## Summary

### Can the pipeline run right now?

**YES!** In fact, you have **two** ways to run it:

1. **Original way** (argparse, individual scripts) - ✅ Works perfectly
2. **New way** (Pydantic config) - ✅ Works perfectly

### Do I need to change anything?

**NO!** The Pydantic system is **completely optional**. It's an **addition**, not a replacement.

### What's the advantage of Pydantic?

- ✅ Catches configuration errors **before** running
- ✅ JSON config files for reproducibility
- ✅ Type safety and autocomplete
- ✅ Better error messages
- ✅ Self-documenting parameters

### What if I just want to run the pipeline now?

Use the original scripts - they work perfectly:

```bash
# Stage data
python src/staging/stage_raw_measurements.py \
  --raw-root data/01_raw \
  --stage-root data/02_stage/raw_measurements \
  --procedures-yaml config/procedures.yml \
  --workers 8

# Analyze
python src/analysis/IV/aggregate_iv_stats.py \
  --stage-root data/02_stage/raw_measurements \
  --date 2025-09-11 \
  --output-dir data/04_analysis/iv_stats

# Compute hysteresis
python src/analysis/IV/compute_hysteresis.py \
  --stats-dir data/04_analysis/iv_stats \
  --output-dir data/04_analysis/hysteresis

# Plot
python src/ploting/IV/compare_polynomial_orders.py \
  --hysteresis-dir data/04_analysis/hysteresis \
  --output-dir plots \
  --compact
```

**All these commands work exactly as before!**

---

## Next Steps

1. ✅ **Test original pipeline** - Make sure your existing workflow works
2. ✨ **Explore Pydantic examples** - Run `python examples/use_pydantic_config.py`
3. 🎯 **Create JSON configs** - For workflows you run repeatedly
4. 🚀 **Use wrapper script** - Try `run_pipeline_with_pydantic.py` when convenient
5. 📈 **Gradual adoption** - Migrate scripts to Pydantic models over time (optional)

The Pydantic system is there when you want **better validation and configuration management**, but your existing scripts continue to work perfectly!
