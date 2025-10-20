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

### CLI Interface (Recommended - Production Ready)

The project now includes a **professional CLI** built with Typer and Rich that provides:
- Clean, unified interface for all pipeline operations
- Beautiful progress bars and status reporting
- Idempotent operations (safe to rerun)
- Subprocess-based isolation (prevents freezing)
- Professional terminal output

**Quick Start:**
```bash
# Full end-to-end pipeline (staging + preprocessing)
# Uses default paths: data/01_raw → data/02_stage → data/03_intermediate
python nanolab-pipeline.py pipeline

# With custom worker count
python nanolab-pipeline.py pipeline --workers 12

# Or use JSON config file
python nanolab-pipeline.py pipeline --config config/complete_run.json
```

**Individual Commands:**

```bash
# 1. Stage only: Raw CSV → Partitioned Parquet
# Uses defaults: data/01_raw → data/02_stage/raw_measurements
python nanolab-pipeline.py stage

# With custom workers
python nanolab-pipeline.py stage --workers 12

# 2. Preprocess only: Segment voltage sweeps
# Uses defaults: data/02_stage/raw_measurements → data/03_intermediate/iv_segments
python nanolab-pipeline.py preprocess

# With custom procedure
python nanolab-pipeline.py preprocess --procedure IVg
```

**Custom Paths (if needed):**
```bash
# Override default paths
python nanolab-pipeline.py pipeline \
  --raw-root /custom/path/raw \
  --stage-root /custom/path/stage \
  --output-root /custom/path/intermediate \
  --workers 6
```

**Key Features:**
- **Idempotent**: Safe to rerun; skips already-processed files
- **Fast**: Parallel processing with configurable worker counts
- **Reliable**: Subprocess isolation prevents multiprocessing conflicts
- **Professional**: Clean progress bars, no verbose logging
- **Production-ready**: Atomic writes, error handling, comprehensive logging

See `docs/CLI_QUICK_REFERENCE.md` for complete CLI documentation.

### Legacy Direct Script Usage (Still Supported)

You can still call the underlying scripts directly if needed:

```bash
# Staging: Raw CSV → Partitioned Parquet
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

### Analysis Pipeline (Post-Preprocessing)

After staging and preprocessing, run analysis and visualization:

```bash
# Step 1: Aggregate statistics and polynomial fits
python src/analysis/IV/aggregate_iv_stats.py \
  --intermediate-root data/03_intermediate/iv_segments \
  --date 2025-10-18 \
  --procedure IV \
  --output-base-dir data/04_analysis \
  --poly-orders 1 3 5 7

# Step 2: Compute hysteresis
python src/analysis/IV/compute_hysteresis.py \
  --stats-dir data/04_analysis/iv_stats/2025-10-18_IV \
  --output-dir data/04_analysis/hysteresis/2025-10-18_IV

# Step 3: Analyze peak locations
python src/analysis/IV/analyze_hysteresis_peaks.py \
  --hysteresis-dir data/04_analysis/hysteresis/2025-10-18_IV \
  --output-dir data/04_analysis/hysteresis_peaks/2025-10-18_IV

# Step 4: Create publication figure (8 subplots, all polynomial orders)
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

# Verify CLI is working
python nanolab-pipeline.py --help
```

### CLI Command Reference

**Default Paths:**
All commands use sensible defaults - no need to specify paths for standard usage!

- **Raw data**: `data/01_raw/`
- **Staged data**: `data/02_stage/raw_measurements/`
- **Intermediate**: `data/03_intermediate/iv_segments/`
- **Config**: `config/procedures.yml`

**Full Pipeline:**
```bash
# Simplest usage (uses all defaults)
python nanolab-pipeline.py pipeline

# With custom workers
python nanolab-pipeline.py pipeline --workers 12

# Or use config file
python nanolab-pipeline.py pipeline --config config/complete_run.json
```

**Individual Stages:**
```bash
# Stage only (uses defaults)
python nanolab-pipeline.py stage

# Preprocess only (uses defaults)
python nanolab-pipeline.py preprocess

# Preprocess different procedure
python nanolab-pipeline.py preprocess --procedure IVg
```

**Common Flags:**
- `--config PATH`: Load settings from JSON config file
- `--workers N`: Number of parallel processes (default: 8)
- `--polars-threads N`: Polars threads per worker (default: 2)
- `--procedure NAME`: Procedure to process (default: "IV")
- `--force`: Overwrite existing files (use with caution)
- `--help`: Show command help

**Override Defaults (when needed):**
```bash
# Custom paths
python nanolab-pipeline.py pipeline \
  --raw-root /custom/raw \
  --stage-root /custom/stage \
  --output-root /custom/intermediate
```

**Example Config File** (`config/complete_run.json`):
```json
{
  "raw_root": "data/01_raw",
  "stage_root": "data/02_stage/raw_measurements",
  "output_root": "data/03_intermediate/iv_segments",
  "procedure": "IV",
  "workers": 8,
  "polars_threads": 2
}
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
├── nanolab-pipeline.py         # CLI entry point (main interface)
├── README.md                   # Main project documentation
├── CLAUDE.md                   # This file - Claude Code instructions
├── requirements.txt            # Python dependencies
│
├── config/
│   ├── procedures.yml          # Schema definitions for all measurement types
│   └── complete_run.json       # Example pipeline configuration
│
├── docs/                       # Documentation
│   ├── README.md               # Documentation index
│   ├── QUICK_START.md          # Quick start guide
│   ├── 4LAYER_COMPLETE.md      # Architecture guide
│   ├── CLI_ARCHITECTURE.md     # Complete CLI technical docs
│   ├── CLI_QUICK_REFERENCE.md  # CLI command reference
│   ├── CLI_IMPLEMENTATION_SUMMARY.md  # Implementation summary
│   ├── PIPELINE_FREEZE_FIX.md  # Multiprocessing debugging history
│   └── archive/                # Historical documentation
│
├── scripts/                    # Analysis and utility scripts
│   ├── explore_mean_traces.py  # Example analysis script
│   └── visualize_pipeline.py   # Generate Graphviz pipeline diagrams
│
├── src/ploting/                # Visualization configuration
│   ├── matplotlibrc            # Publication-ready matplotlib config
│   ├── plotting_config.py      # Plotting themes and helpers
│   ├── plotting_example.py     # Usage examples
│   └── styles_legacy.py        # Legacy styles (for reference)
│
├── data/
│   ├── 01_raw/                 # Raw CSV files with structured headers
│   ├── 02_stage/               # Staged Parquet (schema-validated, partitioned)
│   ├── 03_intermediate/        # Procedure-specific intermediate processing
│   └── 04_analysis/            # Analysis-ready datasets
│       ├── iv_stats/           # Forward/return statistics, polynomial fits
│       ├── hysteresis/         # Hysteresis curves (forward - return)
│       └── hysteresis_peaks/   # Peak location analysis
│
├── plots/                      # Generated visualizations
│
└── src/
    ├── cli/                    # CLI implementation (Typer + Rich)
    │   ├── main.py             # CLI app initialization
    │   ├── commands/           # CLI command implementations
    │   │   ├── pipeline_subprocess.py  # Full pipeline (subprocess-based)
    │   │   ├── stage.py        # Staging command
    │   │   └── preprocess.py   # Preprocessing command
    │   └── utils/              # CLI utilities
    │       ├── console.py      # Rich console helpers
    │       ├── staging_wrapper.py      # Staging with progress bars
    │       └── preprocessing_wrapper.py # Preprocessing with progress bars
    │
    ├── models/                 # Pydantic data models
    │   └── parameters.py       # Parameter validation classes
    │
    ├── staging/                # CSV → Parquet transformation
    │   ├── stage_raw_measurements.py  # Main staging script
    │   └── stage_utils.py             # Shared utilities
    │
    ├── intermediate/           # Procedure-specific preprocessing
    │   ├── IV/                 # IV sweep preprocessing
    │   │   └── iv_preprocessing_script.py
    │   └── IVg/                # Gate sweep preprocessing
    │
    ├── analysis/               # Statistical analysis
    │   └── IV/                 # IV-specific analysis scripts
    │       ├── aggregate_iv_stats.py       # [1] Aggregate measurements
    │       ├── compute_hysteresis.py       # [2] Calculate hysteresis
    │       └── analyze_hysteresis_peaks.py # [3] Peak analysis
    │
    └── ploting/                # Visualization
        └── IV/                 # IV-specific plots
            ├── compare_polynomial_orders.py  # Main publication figure
            ├── explore_hysteresis.py         # Statistical exploration
            └── visualize_hysteresis.py       # Detailed per-range plots
```

## Key Implementation Files

### CLI Layer (User Interface)
- **`nanolab-pipeline.py`**: Main CLI entry point - use this for all pipeline operations
- **`src/cli/main.py`**: Typer app initialization and command registration
- **`src/cli/commands/pipeline_subprocess.py`**: Full pipeline implementation (subprocess-based for isolation)
- **`src/cli/commands/stage.py`**: Staging command with Rich progress bars
- **`src/cli/commands/preprocess.py`**: Preprocessing command with Rich progress bars
- **`src/cli/utils/staging_wrapper.py`**: Wraps staging with progress tracking
- **`src/cli/utils/preprocessing_wrapper.py`**: Wraps preprocessing with progress tracking
- **`src/cli/utils/console.py`**: Rich console styling helpers

### Staging Layer (Layer 2)
- **`src/staging/stage_raw_measurements.py`**: Main staging script with column normalization, date partitioning, and schema validation
- **`config/procedures.yml`**: Schema definitions for all procedure types

### Intermediate Layer (Layer 3 - 4-Layer Architecture)
- **`src/intermediate/IV/iv_preprocessing_script.py`**: Segments voltage sweeps into forward/return phases, saves to Parquet
- **`src/models/parameters.py`**: Pydantic v2 parameter classes with validation

### Analysis Layer (Layer 4 - IV Focus)
- **`src/analysis/IV/aggregate_iv_stats.py`**: Reads pre-segmented data, computes statistics, fits polynomials
- **`src/analysis/IV/compute_hysteresis.py`**: Calculates `I_hyst = I_forward - I_return` for raw and fitted traces
- **`src/analysis/IV/analyze_hysteresis_peaks.py`**: Finds voltage locations of maximum hysteresis
- **`src/ploting/IV/compare_polynomial_orders.py`**: Creates 8-subplot comparison figure (recommended for publications)

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

**Core Pipeline Flow:**
1. **Stage** (`python nanolab-pipeline.py stage`) - Raw CSV → Parquet (run once)
2. **Preprocess** (`python nanolab-pipeline.py preprocess`) - Segment detection (run once per date)
3. **Analyze** - Statistics, fits, hysteresis (run many times, fast!)
4. **Visualize** - Publication-ready figures

**Or run all at once:**
```bash
python nanolab-pipeline.py pipeline --config config/complete_run.json
```

**Main Visualizers (pick based on need):**
- `compare_polynomial_orders.py` - Best for publication (single 8-subplot figure)
- `explore_hysteresis.py` - Statistical exploration (5 analysis plots)
- `visualize_hysteresis.py` - Detailed per-range analysis

See `docs/4LAYER_COMPLETE.md` for complete architecture documentation.
See `docs/CLI_ARCHITECTURE.md` for complete CLI technical documentation.

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

## CLI Architecture and Technical Details

The CLI is built using **Typer** (type-safe CLI framework) and **Rich** (terminal UI library) to provide:
- Type-safe command definitions with automatic help generation
- Beautiful progress bars, tables, and panels
- Professional error messages and status reporting
- Subprocess-based pipeline isolation (prevents multiprocessing conflicts)

**Key Technical Decisions:**

1. **Subprocess Isolation**: Each pipeline stage runs as a separate subprocess
   - Prevents Rich Console singleton conflicts
   - Guarantees complete resource cleanup between stages
   - OS-level process isolation prevents freezing issues
   - Industry-standard pattern (Airflow, Luigi, Nextflow)

2. **Idempotent Operations**: All commands are safe to rerun
   - Staging: Skips files with existing output (based on `run_id`)
   - Preprocessing: Skips already-segmented runs
   - "All skipped" is treated as success (not error)

3. **Professional Output**: Clean, minimal terminal output
   - Single progress bar per stage (no per-batch spam)
   - No emojis or celebration messages
   - Clear status messages with styled paths
   - Comprehensive logging to files (not terminal)

See `docs/CLI_ARCHITECTURE.md` for complete technical documentation.
See `docs/PIPELINE_FREEZE_FIX.md` for debugging history and lessons learned.

## Important Notes

- **USE THE CLI**: Always use `python nanolab-pipeline.py` commands (not direct script calls)
- **4-Layer Architecture (Required)**: Analysis scripts require pre-segmented intermediate data
- **Run preprocessing first**: Always run staging + preprocessing before analysis for each new date
- **Performance**: Preprocessing runs once (~2 min for 7636 runs), analysis runs many times (~10 sec)
- **Idempotency**: Safe to rerun all CLI commands; they skip already-processed files
- **Always use YAML canonical names** for new columns (e.g., `"I (A)"` not `"current"`)
- **UTC storage, local partitioning**: Timestamps stored UTC, partition dates in local timezone
- **Column synonyms** defined in `src/staging/stage_raw_measurements.py:188-198` - extend if new naming variants appear
- **Error handling**: Staging gracefully degrades on bad values (falls back to string), preventing pipeline crashes
- **Polynomial order 3** is recommended default for fits (good balance of accuracy vs overfitting)
- **Hysteresis calculation** uses forward raw data minus return fitted data (fixed 2025-10-05)
- **Professional output**: No emojis, minimal logging, clean progress bars

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
