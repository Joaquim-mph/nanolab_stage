# nanolab_stage

**Medallion-architecture data pipeline for semiconductor/nanoelectronics lab measurements**

## 4-Layer Architecture (Current)

```
Layer 1: Raw CSVs (01_raw/)
   ↓
Layer 2: Stage (02_stage/raw_measurements/)
   └─► stage_raw_measurements.py → proc=…/date=…/run_id=…/part-000.parquet
   ↓
Layer 3: Intermediate (03_intermediate/)
   └─► iv_preprocessing_script.py → iv_segments/proc=…/date=…/run_id=…/segment=…/
   ↓
Layer 4: Analysis (04_analysis/)
   └─► aggregate_iv_stats.py → iv_stats/, hysteresis/, peaks/
   ↓
Plots (plots/)
   └─► compare_polynomial_orders.py → publication figures
```

## Quick Start

```bash
# Full 4-layer pipeline (recommended)
python run_pipeline.py --config config/examples/4layer_pipeline.json

# Or step-by-step:
# 1. Staging (raw CSV → Parquet)
python src/staging/stage_raw_measurements.py \
  --raw-root data/01_raw \
  --stage-root data/02_stage/raw_measurements \
  --procedures-yaml config/procedures.yml \
  --workers 8

# 2. Preprocessing (segment detection - run once per date)
python src/intermediate/IV/iv_preprocessing_script.py \
  --config config/examples/intermediate_config.json

# 3. Analysis (read segments, compute fits - run many times)
python src/analysis/IV/aggregate_iv_stats.py \
  --intermediate-root data/03_intermediate/iv_segments \
  --date 2025-10-18 \
  --output-base-dir data/04_analysis
```

## Documentation

- **`QUICK_START.md`** - Get started in 5 minutes
- **`4LAYER_COMPLETE.md`** - Complete 4-layer architecture guide
- **`FOUR_LAYER_ARCHITECTURE.md`** - Architecture design details
- **`CLAUDE.md`** - Detailed implementation reference for Claude Code

## Key Features

- **4-layer medallion architecture** for clean data flow
- **Pre-segmented intermediate layer** for 10x faster repeated analysis
- **Pydantic v2 configuration** with validation
- **Parallel processing** with ProcessPoolExecutor
- **Hive-partitioned Parquet** for efficient storage
- **Publication-quality plots** with matplotlib