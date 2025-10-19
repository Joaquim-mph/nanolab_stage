# CLI Architecture Documentation

## Overview

The Nanolab Pipeline CLI is a modern, production-grade command-line interface built with **Typer** (CLI framework) and **Rich** (terminal UI). It provides a professional user experience for running multi-stage data processing pipelines.

## Technology Stack

- **Typer**: Type-safe CLI framework with automatic help generation
- **Rich**: Beautiful terminal output with progress bars, tables, and panels
- **Subprocess-based Pipeline**: Complete isolation between pipeline stages
- **ProcessPoolExecutor**: Parallel processing for staging and preprocessing

## Architecture

### Directory Structure

```
src/cli/
├── __init__.py                           # Package initialization with version
├── main.py                               # Main CLI application (entry point)
├── commands/                             # CLI command implementations
│   ├── __init__.py
│   ├── stage.py                          # Stage command (CSV → Parquet)
│   ├── preprocess.py                     # Preprocess command (Parquet → Segments)
│   └── pipeline_subprocess.py            # Full pipeline (subprocess-based)
└── utils/                                # Shared utilities
    ├── __init__.py
    ├── console.py                        # Rich console singleton and helpers
    ├── logging.py                        # Rich-integrated logging
    ├── staging_wrapper.py                # Staging with Rich progress bars
    └── preprocessing_wrapper.py          # Preprocessing with Rich progress bars
```

### Key Design Decisions

#### 1. Subprocess-Based Pipeline (Production Pattern)

**Problem**: Running both staging and preprocessing in the same process caused deadlocks due to:
- Rich Console singleton state pollution
- ProcessPoolExecutor resource conflicts
- Nested context managers interfering with event loops

**Solution**: Each pipeline stage runs as a **separate subprocess**:

```python
# Stage 1 as subprocess
subprocess.run([
    "python", "nanolab-pipeline.py", "stage",
    "--raw-root", "...", "--stage-root", "..."
])  # Exits cleanly, OS cleans all resources ✓

# Stage 2 as subprocess (fresh start!)
subprocess.run([
    "python", "nanolab-pipeline.py", "preprocess",
    "--stage-root", "...", "--output-root", "..."
])  # No conflicts! ✓
```

**Benefits**:
- ✅ **Complete isolation** - No shared state between stages
- ✅ **Guaranteed cleanup** - OS cleans all resources on process exit
- ✅ **Fault tolerance** - One stage crash doesn't kill entire pipeline
- ✅ **Debuggability** - Each stage can be tested independently
- ✅ **Industry standard** - Used by Airflow, Luigi, Nextflow, Snakemake

**Location**: `src/cli/commands/pipeline_subprocess.py`

#### 2. Dedicated Console Instances

**Problem**: Both staging and preprocessing used the **same Rich Console singleton**, causing state conflicts.

**Solution**: Each wrapper creates its own Console instance:

```python
# staging_wrapper.py
staging_console = Console()
progress = Progress(..., console=staging_console)

# preprocessing_wrapper.py
preprocessing_console = Console()
progress = Progress(..., console=preprocessing_console)
```

**Location**:
- `src/cli/utils/staging_wrapper.py:63`
- `src/cli/utils/preprocessing_wrapper.py:168`

#### 3. Idempotent Pipeline Behavior

**Problem**: Running the pipeline twice would fail if all files were already processed.

**Solution**: Treat "all skipped" as **success**, not failure:

```python
# Stage: 0 processed, 31698 skipped → SUCCESS ✓
if results["skipped"] > 0 and results["ok"] == 0:
    console.print("[cyan]All files already staged (skipped)[/cyan]")
    # Continue to next stage

# Preprocess: 0 processed, 7634 skipped → SUCCESS ✓
if results["skipped"] > 0 and results["ok"] == 0:
    console.print("[cyan]All runs already preprocessed (skipped)[/cyan]")
    # Pipeline completes successfully
```

**Location**:
- `src/cli/commands/stage.py:188-197`
- `src/cli/commands/preprocess.py:217-227`

#### 4. Module-Level Function Imports

**Problem**: Worker processes couldn't pickle locally-imported functions.

**Solution**: Import worker functions at **module level**:

```python
# ❌ WRONG: Local import (causes pickle failure)
def run_preprocessing_with_progress(...):
    from iv_preprocessing_script import process_iv_run
    executor.submit(process_iv_run, ...)  # Can't pickle!

# ✓ CORRECT: Module-level import
from iv_preprocessing_script import process_iv_run  # Top of file

def run_preprocessing_with_progress(...):
    executor.submit(process_iv_run, ...)  # Works!
```

**Location**:
- `src/cli/utils/staging_wrapper.py:28-32`
- `src/cli/utils/preprocessing_wrapper.py:31-40`

## Component Documentation

### 1. Main Entry Point (`main.py`)

**Purpose**: CLI application setup and command registration.

**Key Features**:
- Typer app configuration with Rich markup
- Version command (`--version`)
- Command registration (pipeline, stage, preprocess)
- Help text with 4-layer architecture overview

**Usage**:
```bash
python nanolab-pipeline.py --help
python nanolab-pipeline.py --version
python nanolab-pipeline.py <command> --help
```

### 2. Pipeline Command (`pipeline_subprocess.py`)

**Purpose**: Orchestrate full pipeline (Stage 1 + Stage 2) using subprocesses.

**Key Features**:
- Subprocess-based stage execution
- Complete resource isolation
- Pipeline overview panel
- Configuration display
- Success/error handling with exit codes

**Flow**:
```
1. Show pipeline overview
2. Display configuration
3. Run Stage 1 as subprocess
4. Check Stage 1 exit code
5. Run Stage 2 as subprocess
6. Check Stage 2 exit code
7. Show final summary
```

**Usage**:
```bash
python nanolab-pipeline.py pipeline \
  --raw-root data/01_raw \
  --stage-root data/02_stage/raw_measurements \
  --output-root data/03_intermediate \
  --workers 8
```

### 3. Stage Command (`stage.py`)

**Purpose**: Convert raw CSV files to schema-validated Parquet.

**Key Features**:
- CSV file discovery
- Pydantic parameter validation
- Rich progress bar via `staging_wrapper`
- Results table with statistics
- Idempotent behavior (skips existing files)

**Success Conditions**:
- All new files processed successfully
- All files already exist (skipped)
- Some files processed, some skipped (partial success)

**Failure Condition**:
- All files rejected (errors)

**Usage**:
```bash
python nanolab-pipeline.py stage \
  --raw-root data/01_raw \
  --stage-root data/02_stage/raw_measurements \
  --workers 8
```

### 4. Preprocess Command (`preprocess.py`)

**Purpose**: Segment voltage sweeps into forward/return phases.

**Key Features**:
- IV run discovery
- Pydantic parameter validation
- Rich progress bar via `preprocessing_wrapper`
- Results table with segment counts
- Idempotent behavior (skips existing segments)

**Success Conditions**:
- All new runs processed successfully
- All runs already exist (skipped)
- Some runs processed, some skipped (partial success)

**Failure Condition**:
- All runs failed (errors)

**Usage**:
```bash
python nanolab-pipeline.py preprocess \
  --stage-root data/02_stage/raw_measurements \
  --output-root data/03_intermediate \
  --workers 8
```

### 5. Console Utilities (`console.py`)

**Purpose**: Shared Rich console instance and helper functions.

**Exports**:
- `console`: Global Rich Console instance
- `print_step()`: Formatted step headers
- `print_success()`: Success panels (green border)
- `print_error()`: Error panels (red border)
- `print_warning()`: Warning messages
- `print_config()`: Configuration tables

**Design**: Clean, professional output (no emojis).

### 6. Logging Utilities (`logging.py`)

**Purpose**: Rich-integrated logging with custom formatting.

**Key Features**:
- Rich log handler with markup support
- Timestamp formatting
- Level-based coloring
- Results table generation

**Exports**:
- `setup_rich_logging()`: Configure Rich logger
- `create_results_table()`: Generate statistics tables

### 7. Staging Wrapper (`staging_wrapper.py`)

**Purpose**: Wrap staging pipeline with Rich progress bars.

**Key Features**:
- Dedicated Console instance (avoids conflicts)
- ProcessPoolExecutor with context manager
- Real-time progress updates
- Statistics tracking (ok, skipped, rejected)
- Results table generation

**Architecture**:
```python
with ProcessPoolExecutor(max_workers=workers) as executor:
    with progress:
        for csv in csvs:
            executor.submit(ingest_file_task, ...)

        for future in as_completed(futures):
            result = future.result()
            # Update stats and progress bar
```

### 8. Preprocessing Wrapper (`preprocessing_wrapper.py`)

**Purpose**: Wrap preprocessing pipeline with Rich progress bars.

**Key Features**:
- Dedicated Console instance (avoids conflicts)
- ProcessPoolExecutor with manual cleanup
- Real-time progress updates
- Statistics tracking (ok, skipped, error)
- Segment counting
- Results table generation

**Architecture**:
```python
executor = ProcessPoolExecutor(max_workers=workers)
try:
    progress.start()
    for run_path in iv_runs:
        executor.submit(process_iv_run, ...)

    for future in as_completed(futures):
        event = future.result(timeout=60)
        # Update stats and progress bar
finally:
    executor.shutdown(wait=True)
```

## Output Examples

### Successful Pipeline Run

```
╭─────────────────── Nanolab Data Pipeline ───────────────────╮
│                                                             │
│  Stage 1:  Raw CSV → Staged Parquet                         │
│            Schema validation, partitioning, ...             │
│                                                             │
│  Stage 2:  Staged Parquet → Segmented Data                  │
│            Voltage sweep detection, ...                     │
│                                                             │
╰─────────────────────────────────────────────────────────────╯

────────────── Stage 1: Raw CSV → Staged Parquet ──────────────

  Processing CSV files... ━━━━━━━━━━━━━━━━ 100% 31806/31806

                📊 Staging Results
 Total Files                     31,806
 ✓ Processed Successfully        31,698    99.7%
 ✗ Rejected (errors)                108     0.3%
 Elapsed Time                     15.6s

────────────── Stage 2: Staged Parquet → Segments ─────────────

  Processing IV runs... ━━━━━━━━━━━━━━━━━ 100% 7636/7636

                📊 Preprocessing Results
 Total Runs                       7,636
 ✓ Processed Successfully         7,634   100.0%
 ✗ Errors                             2     0.0%
 Total Segments Created          30,309
 Elapsed Time                      8.3s

────────────────────── Pipeline Complete ──────────────────────

╭──────────── Success ────────────╮
│ Full pipeline completed         │
│ successfully                    │
│                                 │
│ Staged data: ...                │
│ Segmented data: ...             │
╰─────────────────────────────────╯
```

### Idempotent Run (All Skipped)

```
────────────── Stage 1: Raw CSV → Staged Parquet ──────────────

  Processing CSV files... ━━━━━━━━━━━━━━━━ 100% 31806/31806

All files already staged (skipped)
Staged data location: data/02_stage/raw_measurements
Total files: 31,698

────────────── Stage 2: Staged Parquet → Segments ─────────────

  Processing IV runs... ━━━━━━━━━━━━━━━━━ 100% 7636/7636

All runs already preprocessed (skipped)
Segmented data location: data/03_intermediate/iv_segments
Total runs: 7,634
Total segments: 30,309
```

## Testing the CLI

### Test Individual Commands

```bash
# Stage only
python nanolab-pipeline.py stage \
  --raw-root data/01_raw \
  --stage-root data/02_stage/raw_measurements \
  --workers 8

# Preprocess only
python nanolab-pipeline.py preprocess \
  --stage-root data/02_stage/raw_measurements \
  --output-root data/03_intermediate \
  --workers 8
```

### Test Full Pipeline

```bash
# First run (processes all data)
python nanolab-pipeline.py pipeline \
  --raw-root data/01_raw \
  --stage-root data/02_stage/raw_measurements \
  --output-root data/03_intermediate \
  --workers 8

# Second run (skips all data, should succeed)
python nanolab-pipeline.py pipeline \
  --raw-root data/01_raw \
  --stage-root data/02_stage/raw_measurements \
  --output-root data/03_intermediate \
  --workers 8
```

### Test with Force Flag

```bash
# Force reprocessing
python nanolab-pipeline.py pipeline \
  --raw-root data/01_raw \
  --stage-root data/02_stage/raw_measurements \
  --output-root data/03_intermediate \
  --workers 8 \
  --force
```

## Performance Characteristics

### Staging Performance
- **Throughput**: ~2,000-4,500 files/second
- **Workers**: 8 (default)
- **Polars threads**: 2 per worker
- **31,806 files**: ~7-15 seconds

### Preprocessing Performance
- **Throughput**: ~900-1,000 runs/second
- **Workers**: 8 (default)
- **Polars threads**: 2 per worker
- **7,636 runs**: ~8 seconds

### Subprocess Overhead
- **Stage transition**: ~100ms
- **Impact**: 0.08% of total runtime (negligible)

## Error Handling

### Graceful Degradation
- Invalid CSV files → Rejected (logged, not fatal)
- Segmentation failures → Skipped (logged, not fatal)
- All files rejected → Pipeline fails with clear error

### Exit Codes
- `0`: Success
- `1`: Failure (staging/preprocessing error)

### Error Messages
- Clear error panels with red borders
- Suggested solutions included
- Full traceback available in terminal mode

## Future Enhancements

### Potential Improvements
1. **Live log tailing**: Stream subprocess logs in real-time
2. **JSON output mode**: Machine-readable results for automation
3. **Dry-run mode**: Preview what will be processed
4. **Resume capability**: Resume failed pipelines from checkpoint
5. **Parallel stage execution**: Run compatible stages concurrently

### Backwards Compatibility
- All current commands will remain supported
- New features will be opt-in via flags

## Related Documentation

- **PIPELINE_FREEZE_FIX.md**: Detailed explanation of multiprocessing issues and solutions
- **CLAUDE.md**: Overall project architecture and data flow
- **4LAYER_COMPLETE.md**: 4-layer architecture documentation
