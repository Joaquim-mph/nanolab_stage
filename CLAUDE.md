# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **medallion-architecture data pipeline** for processing semiconductor/nanoelectronics lab measurement data from CSV files into a queryable Parquet data warehouse using Polars.

**Data Flow:**
```
01_raw/ (CSVs with structured headers)
  → 02_stage/raw_measurements/ (partitioned Parquet, schema-validated)
    → 03_intermediate/ (procedure-specific preprocessing)
      → 04_analysis/ (statistics, aggregation)
        → plots/ (visualizations)
```

## Core Commands

### Staging: Raw CSV → Partitioned Parquet
```bash
python src/staging/stage_raw_measurements.py \
  --raw-root data/01_raw \
  --stage-root data/02_stage/raw_measurements \
  --procedures-yaml config/procedures.yml \
  --workers 4 \
  --polars-threads 2
```

**Optional flags:**
- `--force`: Overwrite existing Parquet files
- `--only-yaml-data`: Drop columns not defined in YAML schema (strict mode)
- `--local-tz`: Timezone for date partitioning (default: `America/Santiago`)
- `--rejects-dir`: Custom location for reject records
- `--events-dir`: Custom location for event JSONs
- `--manifest`: Custom path for consolidated manifest

Transforms raw CSVs with metadata headers into schema-validated Parquet files partitioned as:
```
proc={procedure}/date={YYYY-MM-DD}/run_id={hash}/part-000.parquet
```

**Key behaviors:**
- Parses structured CSV headers (Procedure/Parameters/Metadata/Data blocks)
- Validates and casts types using `procedures.yml` schema
- Normalizes column names via tolerant matching (synonyms + regex)
- Derives metadata: `with_light` flag, `run_id` hash, date partitions
- Atomic writes prevent partial files
- Idempotent: same source + timestamp → same `run_id`
- Writes event JSONs and consolidated manifest for tracking

### IV Analysis Pipeline (4-Layer Architecture - Recommended)

**Quick command (runs all steps):**
```bash
# Full 4-layer pipeline (preprocessing + analysis + plotting)
python run_pipeline.py --config config/examples/4layer_pipeline.json
```

**What it does:**
1. ✅ Runs intermediate preprocessing (segments all IV sweeps once)
2. ✅ Runs analysis (reads pre-segmented data, computes fits)
3. ✅ Runs plotting (creates publication figures)

**Step-by-step (4-layer approach):**
```bash
# Step 1: Preprocessing (run once per date)
python src/intermediate/IV/iv_preprocessing_script.py \
  --config config/examples/intermediate_config.json

# Step 2: Analysis (run many times, uses pre-segmented data)
python src/analysis/IV/aggregate_iv_stats.py \
  --intermediate-root data/03_intermediate/iv_segments \
  --date 2025-10-18 \
  --procedure IV \
  --output-base-dir data/04_analysis \
  --poly-orders 1 3 5 7

# Step 3: Compute hysteresis
python src/analysis/IV/compute_hysteresis.py \
  --stats-dir data/04_analysis/iv_stats/2025-10-18_IV \
  --output-dir data/04_analysis/hysteresis/2025-10-18_IV

# Step 4: Analyze peak locations
python src/analysis/IV/analyze_hysteresis_peaks.py \
  --hysteresis-dir data/04_analysis/hysteresis/2025-10-18_IV \
  --output-dir data/04_analysis/hysteresis_peaks/2025-10-18_IV

# Step 5: Create publication figure (8 subplots, all polynomial orders)
python src/ploting/IV/compare_polynomial_orders.py \
  --hysteresis-dir data/04_analysis/hysteresis/2025-10-18_IV \
  --output-dir plots/2025-10-18 \
  --compact --residuals
```

**Performance benefits:**
- Preprocessing runs once: ~2 minutes for 7636 runs
- Analysis runs many times: ~10 seconds (reads pre-segmented data)
- 10x faster for repeated analysis on same date!

### Environment Setup
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Architecture Details

### Run ID Generation
```python
run_id = sha1_short(f"{source_file_path}|{start_timestamp}")[:16]
```
Stable hash ensures deterministic output paths. Same source + timestamp always produces same `run_id`.

### Procedure Classification
Measurements are classified by data structure:
- **Time-series** (`It`, `ITt`, `Tt`): Has `t (s)` column
- **Sweep** (`IV`, `IVg`, `IVgT`, `LaserCalibration`): Has sweep axes (`Vg (V)`, `Vsd (V)`)

### Column Name Normalization

**Problem:** CSVs have inconsistent naming (`"I (A)"` vs `"i_a"` vs `"current"`)

**Solution (src/staging/stage_raw_measurements.py:201-228):**
1. Normalize both YAML and CSV names (lowercase, strip punctuation, remove spaces/underscores)
2. Try exact normalized match
3. Try regex synonym patterns (e.g., `"I (A)"` matches `r"^i$|^i_a$|^id(_a)?$"`)
4. Try uppercase heuristic
5. Rename CSV columns to YAML canonical names

**Normalization details:**
- Strips whitespace, lowercases, removes punctuation
- Handles Unicode: `°` → `"deg"`, `℃` → `"degc"`
- Example: `"Plate T (°C)"` → `"platetdegc"` matches `"Plate T (degC)"`

### Date Partitioning Logic

**Fallback hierarchy (src/staging/stage_raw_measurements.py:260-293):**
1. **Metadata `Start time`** (preferred): Parse from CSV header
   - Returns `(datetime, date, "meta")`
2. **File path extraction**: Match `YYYY-MM-DD` or `YYYYMMDD` patterns in path
   - Converts local midnight to UTC
   - Returns `(datetime, date, "path")`
3. **File modification time** (last resort): Use mtime
   - Returns `(datetime, date, "mtime")`

**Timezone handling:**
- All timestamps stored in UTC internally
- Partition dates use local timezone (default: `America/Santiago`)
- Example: UTC `2025-01-15 03:00:00` → Santiago `2025-01-14` → partition `date=2025-01-14`

### Derived Metadata

**`with_light` flag (src/staging/stage_raw_measurements.py:345-356):**
```python
with_light = (wavelength_nm is not None) and
             (laser_voltage_V is not None) and
             (laser_voltage_V != 0.0)
```

### Atomic Writes

All Parquet writes use temp-then-rename pattern:
```python
with tempfile.NamedTemporaryFile(dir=out_file.parent) as tmp:
    df.write_parquet(tmp.name)
    Path(tmp.name).replace(out_file)  # Atomic OS operation
```

Prevents partial files if process crashes mid-write.

### Hysteresis Calculation

**Critical implementation detail (fixed 2025-10-05):**
- Raw hysteresis: `I_forward_mean - I_return_mean`
- Polynomial hysteresis: `I_forward_mean - I_return_fit_poly{n}`
- The forward trace is **always raw mean data**
- Only the return trace uses polynomial fits
- This produces 5 hysteresis traces: raw + 4 polynomial orders (n=1,3,5,7)

**Voltage alignment:**
- Voltages are rounded to 2 decimal places before joining
- This ensures -7.9005 and -7.9 are treated as the same point
- Join happens on `V_rounded` column

## YAML Schema (`config/procedures.yml`)

Defines expected structure for each procedure type:

```yaml
procedures:
  IV:
    Parameters:
      Laser wavelength: float
      Chip number: str
      # ... other parameters
    Metadata:
      Start time: datetime
    Data:
      Vsd (V): float
      I (A): float
```

**Type system:**
- `float`: Numeric with unit extraction (`"100 ms"` → `100.0`)
- `float_no_unit`: Numeric without unit parsing
- `int`, `bool`, `str`, `datetime`

**Invalid values:** Gracefully fall back to string (no crashes during staging)

## Parallel Processing

Both staging and analysis use `ProcessPoolExecutor`:

```python
with ProcessPoolExecutor(max_workers=workers) as executor:
    for file in files:
        executor.submit(process_file, ...)
```

**Settings:**
- `--workers N`: Number of parallel processes (default: 4-6)
- `--polars-threads M`: Polars threads per worker (default: 1-2)
- `POLARS_MAX_THREADS` env var set per worker

**Isolation:**
- Each worker has separate Polars thread pool
- Output paths isolated by `run_id` (no conflicts)
- YAML schema cached per process

## Directory Structure

```
nanolab_stage/
├── config/
│   └── procedures.yml          # Schema definitions for all measurement types
├── data/
│   ├── 01_raw/                 # Raw CSV files with structured headers
│   ├── 02_stage/               # Staged Parquet (schema-validated, partitioned)
│   ├── 03_intermediate/        # Procedure-specific intermediate processing
│   └── 04_analysis/            # Analysis-ready datasets
│       ├── iv_stats/           # Forward/return statistics, polynomial fits
│       ├── hysteresis/         # Hysteresis curves (forward - return)
│       └── hysteresis_peaks/   # Peak location analysis
├── plots/                      # Generated visualizations
├── src/
│   ├── staging/                # CSV → Parquet transformation
│   │   ├── stage_raw_measurements.py  # Main staging script
│   │   └── stage_utils.py             # Shared utilities
│   ├── intermediate/           # Procedure-specific preprocessing
│   │   ├── IV/                 # IV sweep preprocessing
│   │   └── IVg/                # Gate sweep preprocessing
│   ├── analysis/               # Statistical analysis
│   │   └── IV/                 # IV-specific analysis scripts
│   │       ├── aggregate_iv_stats.py       # [1] Aggregate measurements
│   │       ├── compute_hysteresis.py       # [2] Calculate hysteresis
│   │       └── analyze_hysteresis_peaks.py # [3] Peak analysis
│   └── ploting/                # Visualization
│       └── IV/                 # IV-specific plots
│           ├── compare_polynomial_orders.py  # Main publication figure
│           ├── explore_hysteresis.py         # Statistical exploration
│           └── visualize_hysteresis.py       # Detailed per-range plots
├── process_iv.py               # Convenience script for IV pipeline
└── requirements.txt            # Python dependencies
```

## Key Implementation Files

### Staging Layer
- **`src/staging/stage_raw_measurements.py`**: Main staging script with column normalization, date partitioning, and schema validation
- **`config/procedures.yml`**: Schema definitions for all procedure types

### Intermediate Layer (4-Layer Architecture)
- **`src/intermediate/IV/iv_preprocessing_script.py`**: Segments voltage sweeps into forward/return phases, saves to Parquet
- **`src/models/parameters.py`**: Pydantic v2 parameter classes with validation

### Analysis Layer (IV Focus)
- **`src/analysis/IV/aggregate_iv_stats.py`**: Reads pre-segmented data, computes statistics, fits polynomials (4-layer only)
- **`src/analysis/IV/compute_hysteresis.py`**: Calculates `I_hyst = I_forward - I_return` for raw and fitted traces
- **`src/analysis/IV/analyze_hysteresis_peaks.py`**: Finds voltage locations of maximum hysteresis
- **`src/ploting/IV/compare_polynomial_orders.py`**: Creates 8-subplot comparison figure (recommended for publications)

### Pipeline Scripts
- **`run_pipeline.py`**: Unified pipeline runner with Pydantic configuration (4-layer architecture)

## 4-Layer Pipeline Organization

**Layer 1: Raw** (`data/01_raw/`)
- CSV files with structured headers

**Layer 2: Stage** (`data/02_stage/raw_measurements/`)
- Schema-validated, type-cast Parquet files
- Partitioned by procedure/date/run_id

**Layer 3: Intermediate** (`data/03_intermediate/`)
- Pre-segmented voltage sweeps (forward/return, positive/negative)
- Metadata columns: `segment_id`, `segment_type`, `segment_direction`
- Run once per date, read many times

**Layer 4: Analysis** (`data/04_analysis/`)
- Statistics, polynomial fits, hysteresis, peaks
- Reads from intermediate layer (fast!)

**Core Pipeline Scripts (run in order):**
1. `iv_preprocessing_script.py` - Segment detection (run once)
2. `aggregate_iv_stats.py` - Read segments + compute fits (run many times)
3. `compute_hysteresis.py` - Calculate hysteresis curves
4. `analyze_hysteresis_peaks.py` - Find peak locations

**Main Visualizers (pick based on need):**
- ⭐ `compare_polynomial_orders.py` - Best for publication (single 8-subplot figure)
- `explore_hysteresis.py` - Statistical exploration (5 analysis plots)
- `visualize_hysteresis.py` - Detailed per-range analysis

See `4LAYER_COMPLETE.md` for complete architecture documentation.

## Common Workflows

### Adding a New Procedure Type

1. Add procedure definition to `config/procedures.yml`:
```yaml
procedures:
  MyNewProc:
    Parameters:
      MyParam: float
    Metadata:
      Start time: datetime
    Data:
      MyColumn: float
```

2. Run staging to process new procedure type
3. Create procedure-specific analysis scripts in `src/analysis/{ProcedureName}/` if needed

### Reprocessing After Schema Changes

```bash
# Force re-stage with new schema
python src/staging/stage_raw_measurements.py --force ...

# Force rebuild warehouse (detects schema hash change automatically)
python build_curated_from_stage_parallel.py --force ...
```

### Debugging Rejected Files

Check reject records:
```bash
cat data/02_stage/_rejects/*.reject.json
```

Check staging manifest for processing history:
```python
import polars as pl
manifest = pl.read_parquet("data/02_stage/raw_measurements/_manifest/manifest.parquet")
# Filter for rejected files
manifest.filter(pl.col("status") == "reject")
```

Typical errors:
- `"missing '# Procedure:'"`: CSV lacks header structure
- `"empty data table"`: No data rows after header
- Type casting failures (usually gracefully handled, falls back to string)

## Important Notes

- **4-Layer Architecture (Required)**: Analysis scripts now require pre-segmented intermediate data
- **Run preprocessing first**: Always run intermediate step before analysis for each new date
- **Performance**: Preprocessing runs once (~2 min for 7636 runs), analysis runs many times (~10 sec)
- **Always use YAML canonical names** for new columns (e.g., `"I (A)"` not `"current"`)
- **UTC storage, local partitioning**: Timestamps stored UTC, partition dates in local timezone
- **Idempotency**: Safe to rerun commands; use `--force` to override
- **Column synonyms** defined in `src/staging/stage_raw_measurements.py:188-198` - extend if new naming variants appear
- **Error handling**: Staging gracefully degrades on bad values (falls back to string), preventing pipeline crashes
- **Polynomial order 3** is recommended default for fits (good balance of accuracy vs overfitting)
- **Hysteresis calculation** uses forward raw data minus return fitted data (fixed 2025-10-05)
- **3-Layer deprecated**: Old 3-layer approach (segment detection in analysis) is no longer supported

## Data Format Specifications

### Raw CSV Format
Each CSV must have a structured header:
```csv
# Procedure: IV
# Parameters:
# Laser wavelength: 450
# Chip number: 71
# Metadata:
# Start time: 2025-01-15T10:30:00Z
# Data:
Vsd (V),I (A)
0.0,1.5e-9
0.1,2.1e-9
```

### Staged Parquet Schema
Output includes original columns plus derived metadata:
- `run_id`: Deterministic hash of source file + timestamp
- `start_dt_utc`: Start timestamp in UTC
- `date_local`: Partition date in local timezone
- `with_light`: Boolean flag for laser experiments
- `source_file`: Original CSV path
- All parameters and metadata from CSV header

### Output File Naming Conventions
- **IV statistics**: `forward_vmax{X}V.csv`, `return_with_fit_vmax{X}V.csv`
- **Hysteresis**: `hysteresis_vmax{X}V.csv`, `hysteresis_summary.csv`
- **Plots**: `all_ranges_all_polynomials.png`, `hysteresis_with_peaks.png`

Where `{X}` is the maximum voltage with `.` replaced by `p` (e.g., `vmax8p0V` for 8.0V)
