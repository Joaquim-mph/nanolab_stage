# 4-Layer Architecture - Implementation Guide

## Overview

I've set up the **foundation** for a true 4-layer medallion architecture. This document explains what's been done, what's ready to use, and how to complete the implementation when you're ready.

---

## Architecture Layers

### Current Status

```
✅ IMPLEMENTED:
├── Layer 1: Raw (data/01_raw/)
│   └── Raw CSV files from measurement equipment
├── Layer 2: Stage (data/02_stage/raw_measurements/)
│   └── Schema-validated, type-cast Parquet files
│       ✅ Fully functional with Pydantic parameters
│
🔶 PARTIALLY IMPLEMENTED:
├── Layer 3: Intermediate (data/03_intermediate/)
│   └── Procedure-specific preprocessing
│       ✅ Pydantic parameters created (IntermediateParameters)
│       ✅ Sophisticated preprocessing script exists (iv_preprocessing_script.py)
│       ⚠️  NOT YET integrated with main pipeline
│
✅ IMPLEMENTED:
└── Layer 4: Analysis (data/04_analysis/)
    └── Final statistics, fits, results
        ✅ Fully functional with Pydantic parameters
        ⚠️  Currently reads from Stage (Layer 2), not Intermediate (Layer 3)
```

---

## What's Been Created

### 1. Pydantic Parameters

**`IntermediateParameters`** - NEW parameter class for intermediate preprocessing:

```python
from models.parameters import IntermediateParameters

params = IntermediateParameters(
    stage_root=Path("data/02_stage/raw_measurements"),
    output_root=Path("data/03_intermediate"),
    procedure="IV",
    voltage_col="Vsd (V)",
    dv_threshold=0.001,  # Noise filtering
    min_segment_points=5,  # Minimum valid segment size
    workers=8,
    force=False
)

# Helper method
output_dir = params.get_output_dir()  # data/03_intermediate/iv_segments
```

**Updated `IVAnalysisParameters`** - Can now use intermediate data:

```python
params = IVAnalysisParameters(
    stage_root=Path("data/02_stage/raw_measurements"),
    date="2025-10-18",
    output_base_dir=Path("data/04_analysis"),

    # NEW: 4-layer architecture support
    intermediate_root=Path("data/03_intermediate/iv_segments"),
    use_segments=True,  # Read from segmented data
)
```

**Updated `PipelineParameters`** - Supports both 3-layer and 4-layer:

```python
# 3-layer (current default)
params = PipelineParameters(
    staging=...,
    analysis=...,
    plotting=...,
    run_staging=True,
    run_intermediate=False,  # Skip intermediate
    run_analysis=True,
    run_plotting=True
)

# 4-layer (when you want segment-level processing)
params = PipelineParameters(
    staging=...,
    intermediate=...,  # NEW
    analysis=...,
    plotting=...,
    run_staging=True,
    run_intermediate=True,  # Enable intermediate
    run_analysis=True,
    run_plotting=True
)
```

### 2. Existing Preprocessing Script

You already have **`src/intermediate/IV/iv_preprocessing_script.py`** which:

✅ Detects voltage sweep segments automatically
✅ Classifies segments:
- `forward_negative`: 0 → -Vmax
- `return_negative`: -Vmax → 0
- `forward_positive`: 0 → +Vmax
- `return_positive`: +Vmax → 0

✅ Adds segment metadata columns
✅ Writes partitioned Parquet files by segment
✅ Uses parallel processing
✅ Creates audit manifest

**Output structure:**
```
03_intermediate/iv_segments/
└── proc=IV/
    └── date=2025-10-18/
        └── run_id=abc123/
            ├── segment=0/part-000.parquet  (forward_negative)
            ├── segment=1/part-000.parquet  (return_negative)
            ├── segment=2/part-000.parquet  (forward_positive)
            └── segment=3/part-000.parquet  (return_positive)
```

---

## Why 4-Layer Architecture?

### Current 3-Layer Problems

**Mixing concerns:**
```python
# src/analysis/IV/aggregate_iv_stats.py (lines 150-210)
# Analysis script ALSO does preprocessing:
for run_id in group_df["run_id"].unique():
    run_df = group_df.filter(pl.col("run_id") == run_id)
    v = run_pd[v_col].values

    # Find min and max voltage indices
    min_idx = np.argmin(v)
    max_idx = np.argmax(v)

    # Detect segments...
    fwd_neg = run_pd.iloc[:min_idx+1].copy()
    ret_neg = run_pd.iloc[min_idx:zero_cross_idx+1].copy()
    # ... more segment logic ...
```

**Problems:**
- ❌ Preprocessing logic duplicated in every analysis script
- ❌ Can't analyze individual segments separately
- ❌ Segment detection runs every time you analyze
- ❌ No caching of preprocessed segments
- ❌ Harder to debug segment detection issues

### 4-Layer Benefits

**Separation of concerns:**
```
Stage (02_stage/)
  → Raw measurement data, one file per run

Intermediate (03_intermediate/)  ← NEW LAYER
  → Preprocessed segments, one file per segment
  → Segment metadata (type, voltage range, direction)
  → Can be analyzed independently

Analysis (04_analysis/)
  → Statistics and fits
  → Reads clean segments from intermediate
```

**Benefits:**
- ✅ Preprocessing runs once, analysis runs many times
- ✅ Each segment can be analyzed independently
- ✅ Easy to filter specific segment types (e.g., only return traces)
- ✅ Debugging: inspect intermediate outputs
- ✅ Reusability: multiple analyses use same preprocessed data

---

## How the 4-Layer Workflow Works

### Data Flow

```
Step 1: STAGING
python src/staging/stage_raw_measurements.py --config config/staging.json
Input:  01_raw/*.csv
Output: 02_stage/raw_measurements/proc=IV/date=*/run_id=*/part-000.parquet
        (One Parquet file per measurement run)

Step 2: INTERMEDIATE (NEW)
python src/intermediate/IV/iv_preprocessing_script.py --config config/intermediate.json
Input:  02_stage/raw_measurements/proc=IV/...
Output: 03_intermediate/iv_segments/proc=IV/date=*/run_id=*/segment=*/part-000.parquet
        (Multiple files: one per segment per run)

Step 3: ANALYSIS
python src/analysis/IV/aggregate_iv_stats.py --config config/analysis.json --use-segments
Input:  03_intermediate/iv_segments/...  (reads segmented data)
Output: 04_analysis/iv_stats/...

Step 4: PLOTTING
python src/ploting/IV/compare_polynomial_orders.py ...
Input:  04_analysis/...
Output: plots/...
```

### Segment-Level Analysis Example

With 4-layer architecture, you can analyze segments independently:

```python
# Analyze ONLY return segments (for hysteresis)
df_return = pl.scan_parquet(
    "03_intermediate/iv_segments/proc=IV/date=2025-10-18/**/segment=*/part-000.parquet"
).filter(
    pl.col("segment_type").is_in(["return_negative", "return_positive"])
).collect()

# Now you have ONLY the return traces, already separated!
# No need to detect segments again in analysis
```

---

## Implementation Status

### ✅ What's Done

1. **Pydantic Parameters Created**
   - `IntermediateParameters` with full validation
   - Updated `IVAnalysisParameters` with `use_segments` flag
   - Updated `PipelineParameters` for 4-layer support

2. **Preprocessing Script Exists**
   - `src/intermediate/IV/iv_preprocessing_script.py` (850 lines)
   - Sophisticated segment detection algorithm
   - Parallel processing support
   - Atomic writes, manifest generation

3. **Cross-Validation**
   - Pipeline validates consistency between layers
   - Prevents using intermediate without proper config

### 🔶 What Needs Integration

1. **Migrate preprocessing script to Pydantic** (30 min)
   - Add Pydantic parameter support to `iv_preprocessing_script.py`
   - Similar to how I migrated `stage_raw_measurements.py`

2. **Update analysis to read from intermediate** (1 hour)
   - Modify `aggregate_iv_stats.py` to read from segments when `use_segments=True`
   - Remove duplicate segment detection logic
   - Read from `03_intermediate/` instead of `02_stage/`

3. **Update pipeline runner** (30 min)
   - Add intermediate step to `run_pipeline.py`
   - Call preprocessing between staging and analysis

4. **Create example configs** (15 min)
   - JSON config for 4-layer workflow
   - Documentation updates

---

## Quick Start (When You Implement)

### Step 1: Run Preprocessing

The existing script already works! Just needs Pydantic integration:

```bash
# Current way (works now)
python src/intermediate/IV/iv_preprocessing_script.py \
  --stage-root data/02_stage/raw_measurements \
  --output-root data/03_intermediate \
  --voltage-col "Vsd (V)" \
  --workers 8

# Future way (after Pydantic migration)
python src/intermediate/IV/iv_preprocessing_script.py \
  --config config/intermediate.json
```

### Step 2: Analyze Segments

Once analysis is updated:

```bash
# Tell analysis to use segments
python src/analysis/IV/aggregate_iv_stats.py \
  --config config/analysis.json \
  --use-segments \
  --intermediate-root data/03_intermediate/iv_segments
```

### Step 3: Full 4-Layer Pipeline

```bash
python run_pipeline.py --config config/4layer_pipeline.json
```

With config:
```json
{
  "staging": { ... },
  "intermediate": {
    "stage_root": "data/02_stage/raw_measurements",
    "output_root": "data/03_intermediate",
    "procedure": "IV",
    "voltage_col": "Vsd (V)",
    "workers": 8
  },
  "analysis": {
    "use_segments": true,
    "intermediate_root": "data/03_intermediate/iv_segments",
    ...
  },
  "plotting": { ... },
  "run_intermediate": true
}
```

---

## Segment Detection Algorithm

Your existing `iv_preprocessing_script.py` uses a sophisticated algorithm:

### Detection Logic

1. **Calculate voltage derivative** (dV/dt)
2. **Filter noise** using `dv_threshold`
3. **Detect direction changes** (dV sign flips)
4. **Detect zero-crossings** (V changes sign)
5. **Classify segments** based on voltage range and direction

### Example

For a sweep: `0V → -8V → 0V → +8V → 0V`

**Detected segments:**
```
Segment 0: [0V → -8V]    type=forward_negative  (going into negative)
Segment 1: [-8V → 0V]    type=return_negative   (returning from negative)
Segment 2: [0V → +8V]    type=forward_positive  (going into positive)
Segment 3: [+8V → 0V]    type=return_positive   (returning from positive)
```

**Output files:**
```
03_intermediate/iv_segments/proc=IV/date=2025-10-18/run_id=abc123/
├── segment=0/part-000.parquet  # Forward negative sweep
├── segment=1/part-000.parquet  # Return negative sweep
├── segment=2/part-000.parquet  # Forward positive sweep
└── segment=3/part-000.parquet  # Return positive sweep
```

Each file contains:
- All original columns (I, V, timestamps, etc.)
- **NEW:** `segment_id`, `segment_type`, `segment_v_start`, `segment_v_end`, `segment_direction`, `point_in_segment`

---

## Use Cases for Segment-Level Analysis

### 1. Hysteresis Analysis

**Current way (3-layer):**
- Read all data
- Detect forward/backward in analysis script
- Compare forward vs backward

**4-layer way:**
- Read only segments with `type=return_*`
- Already separated, no detection needed
- Direct comparison

### 2. Directional Studies

**Question:** Do forward and return sweeps have different characteristics?

**4-layer approach:**
```python
# Get forward segments only
fwd_segments = pl.scan_parquet(
    "03_intermediate/*/segment=*/part-000.parquet"
).filter(
    pl.col("segment_type").str.contains("forward")
).collect()

# Get return segments only
ret_segments = pl.scan_parquet(
    "03_intermediate/*/segment=*/part-000.parquet"
).filter(
    pl.col("segment_type").str.contains("return")
).collect()

# Compare directly
```

### 3. Voltage-Range-Specific Analysis

**Question:** How do negative vs positive voltage sweeps differ?

```python
# Negative voltage segments only
neg_segments = pl.scan_parquet(...).filter(
    pl.col("segment_type").str.contains("negative")
).collect()

# Positive voltage segments only
pos_segments = pl.scan_parquet(...).filter(
    pl.col("segment_type").str.contains("positive")
).collect()
```

---

## Migration Checklist

When you're ready to implement the full 4-layer architecture:

### Phase 1: Integrate Preprocessing (1 hour)

- [ ] Migrate `iv_preprocessing_script.py` to use `IntermediateParameters`
- [ ] Add JSON config support
- [ ] Test preprocessing standalone

### Phase 2: Update Analysis (2 hours)

- [ ] Modify `aggregate_iv_stats.py` to support `use_segments` flag
- [ ] When `use_segments=True`, read from `intermediate_root`
- [ ] Remove duplicate segment detection code
- [ ] Test with segmented data

### Phase 3: Pipeline Integration (1 hour)

- [ ] Add `run_intermediate()` function to `run_pipeline.py`
- [ ] Wire up intermediate step between staging and analysis
- [ ] Create example 4-layer JSON configs
- [ ] Test end-to-end workflow

### Phase 4: Documentation (30 min)

- [ ] Update QUICK_START.md with 4-layer examples
- [ ] Create example configs in `config/examples/`
- [ ] Add tests for intermediate layer

---

## Current Recommendation

**You have two options:**

### Option A: Keep 3-Layer for Now ✅ (Recommended)

The current 3-layer system works perfectly:
```
Stage → Analysis → Plotting
```

**Use this if:**
- You run full analyses each time
- Preprocessing is fast enough
- Simpler workflow is preferred

### Option B: Implement 4-Layer 🚀 (When Needed)

Switch to 4-layer when you want:
- Segment-level analysis
- Preprocessing caching
- Independent loop studies
- Better separation of concerns

**The foundation is ready!** Pydantic parameters are created, validation is in place, preprocessing script exists. Just needs integration (3-4 hours of work).

---

## Summary

**What you have now:**

✅ **Pydantic parameters** for all 4 layers
✅ **Sophisticated preprocessing script** (segment detection)
✅ **3-layer pipeline** fully functional
✅ **Foundation** for 4-layer architecture

**What you can do immediately:**

```bash
# Current 3-layer workflow (fully functional)
python run_pipeline.py --date 2025-10-18 --procedure IV
```

**What's possible with 4-layer (needs 3-4 hours integration):**

```bash
# Future 4-layer workflow
python run_pipeline.py --config config/4layer_pipeline.json

# Or run steps separately
python src/intermediate/IV/iv_preprocessing_script.py --config config/intermediate.json
python run_pipeline.py --date 2025-10-18 --use-segments
```

The architecture is **designed and ready**. Implementation is straightforward when you need the segment-level capabilities!

---

## Files Modified/Created

**Modified:**
- `src/models/parameters.py` - Added `IntermediateParameters`
- `src/models/__init__.py` - Export new parameter class
- `src/models/parameters.py` - Updated `IVAnalysisParameters` with `use_segments`
- `src/models/parameters.py` - Updated `PipelineParameters` for 4-layer

**Existing (Ready to Integrate):**
- `src/intermediate/IV/iv_preprocessing_script.py` - 850 lines, production-ready
- `src/intermediate/IV/fit_return_segments.py` - Additional utilities
- `src/intermediate/IV/visualize_iv_segments.py` - Segment visualization

**To Be Created (When Implementing):**
- `config/examples/4layer_pipeline.json` - Example 4-layer config
- `config/examples/intermediate_only.json` - Preprocessing-only config
- Updated `run_pipeline.py` with intermediate step

Let me know when you want to complete the 4-layer implementation!
