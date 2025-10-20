# Adding New Procedures to the Pipeline

**Professional Guide for Extending the Nanolab Data Pipeline**

This document provides a systematic methodology for adding preprocessing and analysis for new measurement procedures (e.g., IVg, IVgT, ITt) to the existing 4-layer architecture.

**Document Version:** 1.0
**Date:** 2025-10-19
**Applies to:** 4-layer medallion architecture (Raw → Stage → Intermediate → Analysis)

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Step 1: Define YAML Schema](#step-1-define-yaml-schema)
4. [Step 2: Understand Procedure Characteristics](#step-2-understand-procedure-characteristics)
5. [Step 3: Create Pydantic Models](#step-3-create-pydantic-models)
6. [Step 4: Implement Preprocessing (Layer 3)](#step-4-implement-preprocessing-layer-3)
7. [Step 5: Implement Analysis (Layer 4)](#step-5-implement-analysis-layer-4)
8. [Step 6: Add Visualization](#step-6-add-visualization)
9. [Step 7: Integrate with CLI](#step-7-integrate-with-cli)
10. [Step 8: Testing and Validation](#step-8-testing-and-validation)
11. [Best Practices](#best-practices)
12. [Examples](#examples)

---

## Overview

The pipeline follows a **4-layer medallion architecture**:

```
Layer 1: Raw CSV files (data/01_raw/)
         ↓
Layer 2: Staged Parquet (data/02_stage/raw_measurements/)
         [Already handled by staging - works for all procedures!]
         ↓
Layer 3: Intermediate preprocessing (data/03_intermediate/{procedure}_segments/)
         [YOU IMPLEMENT THIS: Procedure-specific segmentation/preprocessing]
         ↓
Layer 4: Analysis results (data/04_analysis/)
         [YOU IMPLEMENT THIS: Statistics, fits, aggregations]
         ↓
         Visualizations (plots/)
         [YOU IMPLEMENT THIS: Publication-ready figures]
```

**Key Principle:** Staging (Layer 2) is **generic** and works for all procedures. You only need to implement Layers 3-4 for procedure-specific logic.

---

## Prerequisites

Before adding a new procedure, ensure:

1. **YAML schema exists** in `config/procedures.yml` for your procedure
2. **Sample data** exists in `data/01_raw/`
3. **Staged data** has been created by running: `python nanolab-pipeline.py stage`
4. You understand the **physical measurement structure** (sweep types, segments, expected patterns)

**Check your staged data:**
```bash
# Verify staging worked
ls -la data/02_stage/raw_measurements/proc={YourProcedure}/

# Quick inspection
python -c "
import polars as pl
df = pl.scan_parquet('data/02_stage/raw_measurements/proc={YourProcedure}/date=*/run_id=*/part-*.parquet').limit(100).collect()
print(df.head())
print(df.describe())
"
```

---

## Step 1: Define YAML Schema

**File:** `config/procedures.yml`

The YAML schema defines the expected structure of your CSV files. This is **already required for staging**, so you likely have this.

**Example: IVg (Gate Voltage Sweep)**
```yaml
procedures:
  IVg:
    Parameters:
      Irange: float
      N_avg: int
      NPLC: int
      Chip number: str
      Laser voltage: float
      Laser wavelength: float
      VDS: float              # Source-drain voltage (constant)
      VG start: float         # Gate sweep start
      VG end: float           # Gate sweep end
      VG step: float          # Gate sweep step
      Step time: float
    Metadata:
      Start time: datetime
    Data:
      Vg (V): float           # Swept variable
      I (A): float            # Measured current
```

**Key considerations:**
- **Parameters**: Experimental settings (constant for one run)
- **Metadata**: Timestamps, operator info, etc.
- **Data**: Columns in the measurement table (voltage, current, time, etc.)
- **Sweep variables**: Identify which column(s) are swept (e.g., `Vg (V)` for IVg, `Vsd (V)` for IV)

---

## Step 2: Understand Procedure Characteristics

Before coding, analyze your procedure's **physical structure**:

### 2.1 Classification

**Procedure Type:**
- **Time-series**: Has `t (s)` column, measurements over time (e.g., `It`, `ITt`, `Tt`)
- **Sweep**: Has swept voltage/parameter, measuring response (e.g., `IV`, `IVg`, `IVgT`)

**Sweep Structure (if applicable):**
- **Single sweep**: One direction only (e.g., 0V → 10V)
- **Bidirectional sweep**: Forward and return (e.g., 0V → 10V → 0V)
- **Hysteresis-capable**: Return path differs from forward (important for analysis!)

### 2.2 Physical Expectations

**For IVg example:**
- **What is swept?** Gate voltage (`Vg`)
- **What is measured?** Drain current (`I (A)`)
- **Expected pattern?** Transistor transfer characteristic (current vs gate voltage)
- **Hysteresis?** Yes, possible for FET devices
- **Segments?** Forward (Vg_start → Vg_end) and Return (Vg_end → Vg_start)
- **Zero-crossing?** Possibly crosses Vg=0V (positive and negative gate bias)

**For ITt example:**
- **What is swept?** Time (implicit)
- **What is measured?** Current over time, possibly with laser modulation
- **Expected pattern?** Photoresponse transients, decay curves
- **Segments?** Light ON vs OFF periods
- **Hysteresis?** No (time-series, not sweep)

### 2.3 Segmentation Logic

**Key question:** How should one experimental run be divided into meaningful segments?

**Sweep procedures (IV, IVg, IVgT):**
- Detect direction reversals (forward vs return)
- Detect zero-crossings (positive vs negative quadrants)
- Label: `forward_negative`, `return_negative`, `forward_positive`, `return_positive`

**Time-series procedures (It, ITt, Tt):**
- Detect laser ON/OFF periods (if applicable)
- Detect temperature ramps vs steady-state
- Label: `laser_on`, `laser_off`, `heating`, `cooling`, `steady_state`

**No segmentation needed:**
- Simple monotonic sweeps (LaserCalibration)
- Short single-direction measurements

---

## Step 3: Create Pydantic Models

**File:** `src/models/parameters.py`

Create **type-safe parameter classes** for configuration validation using Pydantic v2.

### 3.1 Preprocessing Parameters

**Purpose:** Validate config files for intermediate preprocessing scripts.

**Example for IVg:**
```python
from pydantic import BaseModel, Field, field_validator
from pathlib import Path
from typing import Optional

class IVgPreprocessingParameters(BaseModel):
    """
    Configuration for IVg preprocessing (gate sweep segmentation).

    Segments gate voltage sweeps into forward/return phases.
    """
    # Input/Output paths
    stage_root: Path = Field(
        ...,
        description="Root directory containing staged Parquet files"
    )
    output_root: Path = Field(
        ...,
        description="Output directory for segmented data"
    )

    # Procedure settings
    procedure: str = Field(
        default="IVg",
        description="Procedure name (must match staged data)"
    )

    # Processing settings
    workers: int = Field(
        default=6,
        ge=1,
        le=32,
        description="Number of parallel workers"
    )
    polars_threads: int = Field(
        default=2,
        ge=1,
        le=16,
        description="Polars threads per worker"
    )

    # Segmentation parameters
    voltage_col: str = Field(
        default="Vg (V)",
        description="Gate voltage column name"
    )
    dv_threshold: float = Field(
        default=0.001,
        gt=0.0,
        description="Minimum voltage change to consider valid (V)"
    )
    min_segment_points: int = Field(
        default=5,
        ge=3,
        description="Minimum points for valid segment"
    )

    # Optional filters
    chip_number: Optional[str] = Field(
        default=None,
        description="Filter by chip number"
    )
    date_filter: Optional[str] = Field(
        default=None,
        description="Process only this date (YYYY-MM-DD)"
    )

    @field_validator("stage_root", "output_root")
    def paths_must_exist_or_be_creatable(cls, v: Path) -> Path:
        """Ensure paths are valid"""
        if not isinstance(v, Path):
            v = Path(v)
        return v

    @field_validator("procedure")
    def procedure_must_be_valid(cls, v: str) -> str:
        """Validate procedure name"""
        valid_procedures = ["IVg", "IVgT"]  # Add others as implemented
        if v not in valid_procedures:
            raise ValueError(f"Procedure must be one of {valid_procedures}")
        return v

    class Config:
        """Pydantic v2 configuration"""
        extra = "forbid"  # Reject unknown fields
        str_strip_whitespace = True
        validate_assignment = True
```

### 3.2 Analysis Parameters

**Example for IVg:**
```python
class IVgAnalysisParameters(BaseModel):
    """
    Configuration for IVg analysis (transfer characteristics).

    Computes statistics, mobility, threshold voltage, on/off ratio.
    """
    # Input paths
    intermediate_root: Path = Field(
        ...,
        description="Root directory of segmented data"
    )

    # Output paths
    output_base_dir: Path = Field(
        default=Path("data/04_analysis"),
        description="Base directory for analysis results"
    )

    # Filters
    date: str = Field(
        ...,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Date to process (YYYY-MM-DD)"
    )
    procedure: str = Field(default="IVg")
    chip_number: Optional[str] = None

    # Analysis settings
    vds_value: Optional[float] = Field(
        default=None,
        description="Filter by VDS value (V)"
    )
    mobility_calc: bool = Field(
        default=True,
        description="Calculate field-effect mobility"
    )

    # FET parameters (for mobility calculation)
    channel_length: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Channel length (m)"
    )
    channel_width: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Channel width (m)"
    )
    oxide_capacitance: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Oxide capacitance per unit area (F/m²)"
    )

    @field_validator("date")
    def validate_date_format(cls, v: str) -> str:
        """Ensure valid date format"""
        import re
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", v):
            raise ValueError("Date must be YYYY-MM-DD format")
        return v

    class Config:
        extra = "forbid"
        str_strip_whitespace = True
```

### 3.3 Benefits of Pydantic Models

- **Type safety**: Catch configuration errors before processing
- **Automatic validation**: Range checks, format validation
- **Clear documentation**: Field descriptions become help text
- **JSON support**: Load from config files easily
- **Professional**: Industry-standard pattern

---

## Step 4: Implement Preprocessing (Layer 3)

**Goal:** Transform staged Parquet files into analysis-ready segments.

**Create:** `src/intermediate/{Procedure}/{procedure}_preprocessing_script.py`

### 4.1 File Structure Template

```python
"""
{Procedure} Preprocessing Pipeline

Segments {procedure} measurements into distinct phases:
- [List your segments here]

Reads from: 02_stage/raw_measurements/proc={Procedure}/
Writes to: 03_intermediate/{procedure}_segments/

Uses Pydantic validation for professional configuration management.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import polars as pl

# Add project paths
_PROJECT_ROOT = str(Path(__file__).parent.parent.parent.parent)
_STAGING_PATH = str(Path(__file__).parent.parent.parent / "staging")

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _STAGING_PATH not in sys.path:
    sys.path.insert(0, _STAGING_PATH)

from src.models.parameters import {Procedure}PreprocessingParameters
from pydantic import ValidationError

from stage_utils import ensure_dir, sha1_short, warn
from stage_raw_measurements import atomic_write_parquet


# ----------------------------- Configuration -----------------------------

DEFAULT_WORKERS = 6
DEFAULT_POLARS_THREADS = 1

# Procedure-specific defaults
DEFAULT_VOLTAGE_COL = "Vg (V)"  # Or whatever your swept variable is
DEFAULT_DV_THRESHOLD = 0.001
DEFAULT_MIN_SEGMENT_POINTS = 5


# ----------------------------- Segment Detection -----------------------------

@dataclass
class SegmentInfo:
    """
    Metadata for a detected segment.

    Attributes:
        segment_id: Sequential segment number (0, 1, 2, ...)
        segment_type: Descriptive label (forward, return, laser_on, etc.)
        start_idx: Starting row index
        end_idx: Ending row index (inclusive)
        # Add procedure-specific fields as needed
    """
    segment_id: int
    segment_type: str
    start_idx: int
    end_idx: int
    # Add more fields as needed


def detect_segments(
    df: pl.DataFrame,
    voltage_col: str = DEFAULT_VOLTAGE_COL,
    threshold: float = DEFAULT_DV_THRESHOLD,
    min_points: int = DEFAULT_MIN_SEGMENT_POINTS
) -> List[SegmentInfo]:
    """
    Detect segments in the measurement data.

    THIS IS THE CORE LOGIC - Customize for your procedure!

    Args:
        df: DataFrame with measurements (must be time-ordered)
        voltage_col: Column to analyze for segmentation
        threshold: Minimum change to consider significant
        min_points: Minimum points for valid segment

    Returns:
        List of SegmentInfo objects

    Algorithm:
        [Describe your segmentation logic here]

    Example:
        [Provide an example of expected segmentation]
    """
    # IMPLEMENT YOUR SEGMENTATION LOGIC HERE
    # See IV example for voltage sweep segmentation
    # See ITt example for time-series segmentation (if you create it)

    pass


# ----------------------------- Processing Functions -----------------------------

def process_single_run(
    run_parquet_path: Path,
    output_root: Path,
    voltage_col: str,
    dv_threshold: float,
    min_segment_points: int,
) -> dict:
    """
    Process a single run: detect segments and write to disk.

    Args:
        run_parquet_path: Path to staged Parquet file
        output_root: Output directory root
        voltage_col: Column name for segmentation
        dv_threshold: Voltage change threshold
        min_segment_points: Minimum segment size

    Returns:
        Result dict with status and statistics
    """
    try:
        # Load the run data
        df = pl.read_parquet(run_parquet_path)

        # Extract metadata from partition path
        # Example: .../proc=IVg/date=2025-10-18/run_id=abc123/part-000.parquet
        parts = run_parquet_path.parts
        proc_idx = [i for i, p in enumerate(parts) if p.startswith("proc=")]
        if not proc_idx:
            return {"status": "error", "reason": "Could not parse proc from path"}

        proc_part = parts[proc_idx[0]]
        date_part = parts[proc_idx[0] + 1]
        run_id_part = parts[proc_idx[0] + 2]

        procedure = proc_part.split("=")[1]
        date = date_part.split("=")[1]
        run_id = run_id_part.split("=")[1]

        # Detect segments
        segments = detect_segments(
            df,
            voltage_col=voltage_col,
            threshold=dv_threshold,
            min_points=min_segment_points
        )

        if not segments:
            return {
                "status": "no_segments",
                "run_id": run_id,
                "reason": "No valid segments detected"
            }

        # Write each segment to disk
        for seg in segments:
            # Extract segment data
            seg_df = df.slice(seg.start_idx, seg.end_idx - seg.start_idx + 1)

            # Add segment metadata columns
            seg_df = seg_df.with_columns([
                pl.lit(seg.segment_id).alias("segment_id"),
                pl.lit(seg.segment_type).alias("segment_type"),
                # Add more metadata as needed
            ])

            # Create output path
            # Pattern: proc={proc}/date={date}/run_id={run_id}/segment={segment_id}/part-000.parquet
            out_dir = (
                output_root
                / f"proc={procedure}"
                / f"date={date}"
                / f"run_id={run_id}"
                / f"segment={seg.segment_id}"
            )
            ensure_dir(out_dir)
            out_file = out_dir / "part-000.parquet"

            # Check if already exists (idempotency)
            if out_file.exists():
                continue

            # Atomic write
            atomic_write_parquet(seg_df, out_file)

        return {
            "status": "ok",
            "run_id": run_id,
            "segments": len(segments),
            "points": len(df)
        }

    except Exception as e:
        return {
            "status": "error",
            "run_id": run_parquet_path.stem,
            "reason": str(e)
        }


def process_file_wrapper(args):
    """Wrapper for multiprocessing (must be picklable)"""
    return process_single_run(*args)


# ----------------------------- Main Pipeline -----------------------------

def main():
    """Main preprocessing pipeline"""
    parser = argparse.ArgumentParser(
        description=f"{DEFAULT_VOLTAGE_COL.split()[0]} preprocessing pipeline"
    )

    # Use Pydantic model field names for arguments
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--procedure", type=str, default="IVg")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--polars-threads", type=int, default=DEFAULT_POLARS_THREADS)
    parser.add_argument("--config", type=Path, help="JSON config file")

    args = parser.parse_args()

    # Load and validate config
    if args.config:
        with open(args.config) as f:
            config_dict = json.load(f)
        try:
            params = {Procedure}PreprocessingParameters(**config_dict)
        except ValidationError as e:
            print(f"Configuration validation error:\n{e}")
            sys.exit(1)
    else:
        # Build from command-line args
        try:
            params = {Procedure}PreprocessingParameters(
                stage_root=args.stage_root,
                output_root=args.output_root,
                procedure=args.procedure,
                workers=args.workers,
                polars_threads=args.polars_threads,
            )
        except ValidationError as e:
            print(f"Parameter validation error:\n{e}")
            sys.exit(1)

    print(f"Starting {params.procedure} preprocessing...")
    print(f"  Stage root: {params.stage_root}")
    print(f"  Output root: {params.output_root}")
    print(f"  Workers: {params.workers}")

    # Find all staged runs
    pattern = str(params.stage_root / f"proc={params.procedure}" / "date=*" / "run_id=*" / "part-*.parquet")
    staged_files = list(Path().glob(pattern))

    if not staged_files:
        print(f"No staged files found matching: {pattern}")
        sys.exit(1)

    print(f"Found {len(staged_files)} staged run files")

    # Build task arguments
    tasks = [
        (
            f,
            params.output_root,
            params.voltage_col,
            params.dv_threshold,
            params.min_segment_points,
        )
        for f in staged_files
    ]

    # Process in parallel
    results = {"ok": 0, "error": 0, "no_segments": 0}

    with ProcessPoolExecutor(max_workers=params.workers) as executor:
        for result in executor.map(process_file_wrapper, tasks):
            status = result["status"]
            results[status] = results.get(status, 0) + 1

    # Summary
    print(f"\nPreprocessing complete!")
    print(f"  Successful: {results['ok']}")
    print(f"  No segments: {results.get('no_segments', 0)}")
    print(f"  Errors: {results['error']}")
    print(f"\nOutput: {params.output_root}")


if __name__ == "__main__":
    main()
```

### 4.2 Key Implementation Details

**Segmentation Logic (Critical!):**
- Study existing `detect_voltage_segments()` in `src/intermediate/IV/iv_preprocessing_script.py`
- For sweeps: detect direction reversals and zero-crossings
- For time-series: detect state changes (laser on/off, temperature ramps)
- Use derivative analysis for sweep direction detection
- Use threshold filtering to handle noisy data

**Partitioning Scheme:**
```
data/03_intermediate/{procedure}_segments/
    proc={Procedure}/
        date={YYYY-MM-DD}/
            run_id={hash}/
                segment={segment_id}/
                    part-000.parquet
```

**Metadata Columns to Add:**
- `segment_id`: Sequential integer (0, 1, 2, ...)
- `segment_type`: Descriptive label (e.g., "forward_positive", "laser_on")
- Any other segment-specific metadata

**Idempotency:**
- Check if output file exists before writing
- Same input → same output (deterministic)
- Safe to rerun

---

## Step 5: Implement Analysis (Layer 4)

**Goal:** Compute statistics, fits, and derived quantities from segmented data.

**Create:** `src/analysis/{Procedure}/{procedure}_analysis.py`

### 5.1 Analysis Script Template

```python
#!/usr/bin/env python3
"""
{Procedure} Analysis Pipeline

Computes statistics and derived quantities for {procedure} measurements.

Reads from: 03_intermediate/{procedure}_segments/
Writes to: 04_analysis/{procedure}_stats/

For each experimental condition, compute:
- [List your analysis outputs]
"""

import polars as pl
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
import argparse
from typing import Optional, Tuple
import sys

# Add project root for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.models.parameters import {Procedure}AnalysisParameters
from pydantic import ValidationError


# ----------------------------- Analysis Functions -----------------------------

def load_segmented_data(
    intermediate_root: Path,
    date: str,
    procedure: str = "IVg",
    chip_number: Optional[str] = None,
    vds_value: Optional[float] = None,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Load pre-segmented data from intermediate layer.

    Returns forward and return segments separately (if applicable).
    For time-series data, return all segments or grouped by state.

    Args:
        intermediate_root: Root directory of segmented data
        date: Date in YYYY-MM-DD format
        procedure: Procedure name
        chip_number: Optional filter by chip
        vds_value: Optional filter by VDS (for IVg)

    Returns:
        Tuple of DataFrames (forward_df, return_df) for sweeps
        OR (laser_on_df, laser_off_df) for time-series
    """
    # Load all segments
    pattern = str(
        intermediate_root
        / f"proc={procedure}"
        / f"date={date}"
        / "run_id=*"
        / "segment=*"
        / "part-*.parquet"
    )

    print(f"Loading segmented data: {pattern}")

    try:
        lf = pl.scan_parquet(pattern)
        df_all = lf.collect()
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

    print(f"Loaded {len(df_all)} points, {df_all['run_id'].n_unique()} runs")

    # Apply filters
    if chip_number and "chip_number" in df_all.columns:
        df_all = df_all.filter(pl.col("chip_number") == chip_number)
        print(f"Filtered to chip {chip_number}: {df_all['run_id'].n_unique()} runs")

    if vds_value is not None and "VDS" in df_all.columns:
        # Allow small tolerance for floating point comparison
        df_all = df_all.filter(
            (pl.col("VDS") - vds_value).abs() < 0.01
        )
        print(f"Filtered to VDS={vds_value}V: {df_all['run_id'].n_unique()} runs")

    # Separate by segment type (customize for your procedure!)
    # Example for sweeps:
    forward_df = df_all.filter(pl.col("segment_type").str.contains("forward"))
    return_df = df_all.filter(pl.col("segment_type").str.contains("return"))

    print(f"  Forward: {len(forward_df)} points")
    print(f"  Return: {len(return_df)} points")

    return forward_df, return_df


def compute_statistics(
    df: pl.DataFrame,
    group_by_cols: list[str],
    value_col: str = "I (A)"
) -> pl.DataFrame:
    """
    Compute mean and std dev grouped by specified columns.

    Args:
        df: Input DataFrame
        group_by_cols: Columns to group by (e.g., ["Vg (V)", "VDS"])
        value_col: Column to compute statistics on

    Returns:
        DataFrame with mean, std, count per group
    """
    stats = df.group_by(group_by_cols).agg([
        pl.col(value_col).mean().alias(f"{value_col.split()[0]}_mean"),
        pl.col(value_col).std().alias(f"{value_col.split()[0]}_std"),
        pl.col(value_col).count().alias("n_measurements"),
        pl.col("run_id").n_unique().alias("n_runs"),
    ]).sort(group_by_cols)

    return stats


def compute_mobility(
    vg: np.ndarray,
    ids: np.ndarray,
    vds: float,
    L: float,
    W: float,
    Cox: float
) -> Tuple[float, float]:
    """
    Compute field-effect mobility from transfer characteristic.

    Mobility: μ_FE = (L / (W * Cox * VDS)) * (dIds / dVg)

    Args:
        vg: Gate voltage array (V)
        ids: Drain-source current array (A)
        vds: Drain-source voltage (V)
        L: Channel length (m)
        W: Channel width (m)
        Cox: Oxide capacitance per area (F/m²)

    Returns:
        (max_mobility, vg_at_max_mobility)
    """
    # Compute transconductance (dIds/dVg)
    gm = np.gradient(ids, vg)

    # Compute mobility
    mobility = (L / (W * Cox * vds)) * gm

    # Find maximum mobility
    max_idx = np.argmax(np.abs(mobility))
    max_mobility = mobility[max_idx]
    vg_max = vg[max_idx]

    return max_mobility, vg_max


def compute_threshold_voltage(
    vg: np.ndarray,
    ids: np.ndarray,
    method: str = "linear_extrapolation"
) -> float:
    """
    Compute threshold voltage using specified method.

    Args:
        vg: Gate voltage array
        ids: Drain-source current array
        method: "linear_extrapolation" or "constant_current"

    Returns:
        Threshold voltage (V)
    """
    if method == "linear_extrapolation":
        # Find region of maximum transconductance
        gm = np.gradient(ids, vg)
        max_gm_idx = np.argmax(np.abs(gm))

        # Fit linear region around max gm
        fit_range = 10  # points
        start = max(0, max_gm_idx - fit_range // 2)
        end = min(len(vg), max_gm_idx + fit_range // 2)

        # Linear fit
        coeffs = np.polyfit(vg[start:end], ids[start:end], 1)
        # Extrapolate to I=0
        vth = -coeffs[1] / coeffs[0]

        return vth

    elif method == "constant_current":
        # Find Vg where |Ids| reaches threshold current
        i_threshold = 1e-9  # 1 nA (adjust for your device)
        idx = np.argmax(np.abs(ids) > i_threshold)
        return vg[idx]

    else:
        raise ValueError(f"Unknown method: {method}")


def compute_onoff_ratio(ids: np.ndarray) -> float:
    """
    Compute on/off current ratio.

    Args:
        ids: Drain-source current array

    Returns:
        Ion/Ioff ratio
    """
    i_on = np.max(np.abs(ids))
    i_off = np.min(np.abs(ids[np.abs(ids) > 0]))  # Exclude zeros

    return i_on / i_off if i_off > 0 else np.inf


# ----------------------------- Main Analysis -----------------------------

def main():
    """Main analysis pipeline"""
    parser = argparse.ArgumentParser(
        description="{Procedure} analysis pipeline"
    )

    parser.add_argument("--intermediate-root", type=Path, required=True)
    parser.add_argument("--output-base-dir", type=Path, default=Path("data/04_analysis"))
    parser.add_argument("--date", type=str, required=True)
    parser.add_argument("--procedure", type=str, default="IVg")
    parser.add_argument("--config", type=Path, help="JSON config file")

    args = parser.parse_args()

    # Load and validate parameters
    if args.config:
        import json
        with open(args.config) as f:
            config_dict = json.load(f)
        try:
            params = {Procedure}AnalysisParameters(**config_dict)
        except ValidationError as e:
            print(f"Configuration error:\n{e}")
            sys.exit(1)
    else:
        try:
            params = {Procedure}AnalysisParameters(
                intermediate_root=args.intermediate_root,
                output_base_dir=args.output_base_dir,
                date=args.date,
                procedure=args.procedure,
            )
        except ValidationError as e:
            print(f"Parameter error:\n{e}")
            sys.exit(1)

    print(f"Starting {params.procedure} analysis for {params.date}...")

    # Load segmented data
    forward_df, return_df = load_segmented_data(
        params.intermediate_root,
        params.date,
        params.procedure,
        params.chip_number,
        params.vds_value,
    )

    # Compute statistics
    # Group by voltage (and any other relevant parameters)
    forward_stats = compute_statistics(
        forward_df,
        group_by_cols=["Vg (V)"],  # Adjust for your procedure
        value_col="I (A)"
    )

    return_stats = compute_statistics(
        return_df,
        group_by_cols=["Vg (V)"],
        value_col="I (A)"
    )

    # Create output directory
    output_dir = params.output_base_dir / f"{params.procedure.lower()}_stats" / f"{params.date}_{params.procedure}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save statistics
    forward_stats.write_csv(output_dir / "forward_stats.csv")
    return_stats.write_csv(output_dir / "return_stats.csv")

    print(f"\nStatistics saved to: {output_dir}")

    # Compute derived quantities (if applicable)
    if params.mobility_calc and all([
        params.channel_length,
        params.channel_width,
        params.oxide_capacitance,
        params.vds_value
    ]):
        print("\nComputing mobility...")

        vg = forward_stats["Vg (V)"].to_numpy()
        ids = forward_stats["I_mean"].to_numpy()

        max_mobility, vg_max = compute_mobility(
            vg, ids,
            params.vds_value,
            params.channel_length,
            params.channel_width,
            params.oxide_capacitance
        )

        print(f"  Max mobility: {max_mobility:.2e} cm²/V·s")
        print(f"  At Vg = {vg_max:.2f} V")

        # Compute threshold voltage
        vth = compute_threshold_voltage(vg, ids)
        print(f"  Threshold voltage: {vth:.2f} V")

        # Compute on/off ratio
        onoff = compute_onoff_ratio(ids)
        print(f"  On/off ratio: {onoff:.2e}")

        # Save device parameters
        device_params = {
            "max_mobility_cm2_per_Vs": float(max_mobility),
            "vg_at_max_mobility_V": float(vg_max),
            "threshold_voltage_V": float(vth),
            "onoff_ratio": float(onoff),
            "vds_V": params.vds_value,
            "date": params.date,
        }

        import json
        with open(output_dir / "device_parameters.json", "w") as f:
            json.dump(device_params, f, indent=2)

        print(f"Device parameters saved to: {output_dir / 'device_parameters.json'}")


if __name__ == "__main__":
    main()
```

### 5.2 Analysis Outputs

**Organize results by analysis type:**

```
data/04_analysis/
    {procedure}_stats/           # Basic statistics (mean, std)
        {date}_{Procedure}/
            forward_stats.csv
            return_stats.csv

    {procedure}_fits/            # Polynomial/model fits
        {date}_{Procedure}/
            fit_parameters.csv

    {procedure}_derived/         # Derived quantities (mobility, hysteresis, etc.)
        {date}_{Procedure}/
            device_parameters.json
            hysteresis.csv
```

**CSV Format Example (forward_stats.csv):**
```csv
Vg (V),I_mean,I_std,n_measurements,n_runs
-10.0,1.23e-12,5.4e-14,150,50
-9.5,2.45e-12,6.1e-14,150,50
...
```

---

## Step 6: Add Visualization

**Create:** `src/ploting/{Procedure}/visualize_{procedure}.py`

### 6.1 Plotting Script Template

```python
#!/usr/bin/env python3
"""
{Procedure} Visualization

Creates publication-ready figures for {procedure} analysis.

Plots:
- Transfer characteristics (Ids vs Vg)
- Forward vs return comparison
- Hysteresis analysis
- Multi-panel overview
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import polars as pl
import numpy as np
from pathlib import Path
import argparse

# Set publication-quality defaults
mpl.rcParams['font.size'] = 10
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['axes.linewidth'] = 1.0
mpl.rcParams['xtick.major.width'] = 1.0
mpl.rcParams['ytick.major.width'] = 1.0


def plot_transfer_characteristic(
    forward_stats: pl.DataFrame,
    return_stats: pl.DataFrame,
    output_path: Path,
    log_scale: bool = True,
    show_hysteresis: bool = True
):
    """
    Plot transfer characteristic (Ids vs Vg).

    Args:
        forward_stats: Forward sweep statistics
        return_stats: Return sweep statistics
        output_path: Output file path
        log_scale: Use log scale for current
        show_hysteresis: Highlight hysteresis
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Extract data
    vg_fwd = forward_stats["Vg (V)"].to_numpy()
    ids_fwd = np.abs(forward_stats["I_mean"].to_numpy())
    ids_fwd_std = forward_stats["I_std"].to_numpy()

    vg_ret = return_stats["Vg (V)"].to_numpy()
    ids_ret = np.abs(return_stats["I_mean"].to_numpy())
    ids_ret_std = return_stats["I_std"].to_numpy()

    # Plot forward sweep
    ax.plot(vg_fwd, ids_fwd, 'o-', label='Forward', color='C0', markersize=4)
    ax.fill_between(
        vg_fwd,
        ids_fwd - ids_fwd_std,
        ids_fwd + ids_fwd_std,
        alpha=0.2,
        color='C0'
    )

    # Plot return sweep
    ax.plot(vg_ret, ids_ret, 's-', label='Return', color='C1', markersize=4)
    ax.fill_between(
        vg_ret,
        ids_ret - ids_ret_std,
        ids_ret + ids_ret_std,
        alpha=0.2,
        color='C1'
    )

    # Formatting
    ax.set_xlabel('Gate Voltage $V_g$ (V)', fontsize=12)
    ax.set_ylabel('Drain Current $|I_{ds}|$ (A)', fontsize=12)

    if log_scale:
        ax.set_yscale('log')
        ax.set_ylabel('Drain Current $|I_{ds}|$ (A, log scale)', fontsize=12)

    ax.legend(frameon=False, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add title with statistics
    n_runs = forward_stats["n_runs"][0]
    ax.set_title(f'Transfer Characteristic (n={n_runs} runs)', fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_hysteresis(
    forward_stats: pl.DataFrame,
    return_stats: pl.DataFrame,
    output_path: Path
):
    """
    Plot hysteresis (forward - return).

    Args:
        forward_stats: Forward sweep statistics
        return_stats: Return sweep statistics
        output_path: Output file path
    """
    # Align voltages (round to avoid floating point issues)
    fwd_df = forward_stats.with_columns([
        (pl.col("Vg (V)") * 100).round(0).cast(pl.Int64).alias("Vg_rounded")
    ])
    ret_df = return_stats.with_columns([
        (pl.col("Vg (V)") * 100).round(0).cast(pl.Int64).alias("Vg_rounded")
    ])

    # Join on rounded voltage
    merged = fwd_df.join(ret_df, on="Vg_rounded", suffix="_ret")

    # Compute hysteresis
    merged = merged.with_columns([
        (pl.col("I_mean") - pl.col("I_mean_ret")).alias("I_hysteresis")
    ])

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    vg = merged["Vg (V)"].to_numpy()
    i_hyst = merged["I_hysteresis"].to_numpy()

    ax.plot(vg, i_hyst * 1e9, 'o-', color='C2', markersize=5, label='Hysteresis')
    ax.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_xlabel('Gate Voltage $V_g$ (V)', fontsize=12)
    ax.set_ylabel('Hysteresis Current $\\Delta I_{ds}$ (nA)', fontsize=12)
    ax.set_title('Transfer Characteristic Hysteresis', fontsize=14)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    """Main plotting script"""
    parser = argparse.ArgumentParser(description="IVg visualization")
    parser.add_argument("--stats-dir", type=Path, required=True,
                       help="Directory with forward_stats.csv and return_stats.csv")
    parser.add_argument("--output-dir", type=Path, required=True,
                       help="Output directory for plots")
    parser.add_argument("--log-scale", action="store_true",
                       help="Use log scale for current")

    args = parser.parse_args()

    # Load statistics
    forward_stats = pl.read_csv(args.stats_dir / "forward_stats.csv")
    return_stats = pl.read_csv(args.stats_dir / "return_stats.csv")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    print("Generating plots...")

    plot_transfer_characteristic(
        forward_stats,
        return_stats,
        args.output_dir / "transfer_characteristic.png",
        log_scale=args.log_scale
    )

    plot_hysteresis(
        forward_stats,
        return_stats,
        args.output_dir / "hysteresis.png"
    )

    print(f"\nAll plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
```

### 6.2 Plotting Best Practices

**Publication Quality:**
- DPI ≥ 300 for publication figures
- Use vector formats (PDF, SVG) when possible
- Clear axis labels with units
- Readable font sizes (10-12pt)
- Colorblind-friendly palettes

**Multi-panel Figures:**
- Use `plt.subplots()` for organized layouts
- Label panels (a), (b), (c), ...
- Consistent styling across panels
- Shared axes when comparing

**Error Visualization:**
- Always show error bars or bands
- Use `fill_between()` for standard deviation
- Use `errorbar()` for discrete points
- Make error regions semi-transparent

---

## Step 7: Integrate with CLI

**Goal:** Add your new procedure to the CLI interface for seamless operation.

### 7.1 Update Preprocessing Wrapper

**File:** `src/cli/utils/preprocessing_wrapper.py`

Add imports for your new preprocessing script at the module level:

```python
# Add to imports section
_IVG_PATH = str(Path(__file__).parent.parent.parent / "intermediate" / "IVg")
if _IVG_PATH not in sys.path:
    sys.path.insert(0, _IVG_PATH)

from ivg_preprocessing_script import process_single_run as process_ivg_run
```

Add a dispatcher function or update existing wrapper to handle multiple procedures:

```python
def run_preprocessing_with_progress(
    stage_root: Path,
    output_root: Path,
    procedure: str,  # Now dynamic!
    workers: int = 6,
    polars_threads: int = 2,
    force: bool = False,
) -> Dict[str, int]:
    """
    Run preprocessing with Rich progress bars.

    Supports multiple procedures: IV, IVg, IVgT, etc.
    """
    # Select appropriate processing function
    if procedure == "IV":
        from iv_preprocessing_script import process_single_run
    elif procedure == "IVg":
        from ivg_preprocessing_script import process_single_run
    elif procedure == "IVgT":
        from ivgt_preprocessing_script import process_single_run
    else:
        raise ValueError(f"Unknown procedure: {procedure}")

    # Rest of the existing logic...
```

### 7.2 Update CLI Commands

**File:** `src/cli/commands/preprocess.py`

The existing command should already support multiple procedures if properly designed. Verify:

```python
@app.command()
def preprocess(
    # ... existing parameters ...
    procedure: str = typer.Option(
        "IV",
        "--procedure",
        "-p",
        help="Measurement procedure (IV, IVg, IVgT, etc.)"
    ),
    # ... rest of parameters ...
):
    """Preprocess staged measurements into segments"""

    # Validate procedure
    valid_procedures = ["IV", "IVg", "IVgT"]  # Add your new procedure here
    if procedure not in valid_procedures:
        console.print(f"[error]Unknown procedure: {procedure}[/error]")
        console.print(f"Valid procedures: {', '.join(valid_procedures)}")
        raise typer.Exit(1)

    # Rest of command logic...
```

### 7.3 Update Documentation

Add your new procedure to:
- `docs/CLI_QUICK_REFERENCE.md` - Usage examples
- `docs/4LAYER_COMPLETE.md` - Architecture updates
- `CLAUDE.md` - Procedure list
- `README.md` - Overview section

---

## Step 8: Testing and Validation

### 8.1 Unit Testing

Create test files: `tests/test_{procedure}_preprocessing.py`

```python
import pytest
import polars as pl
from pathlib import Path
from src.intermediate.IVg.ivg_preprocessing_script import detect_segments


def test_detect_segments_bidirectional():
    """Test segment detection for bidirectional sweep"""
    # Create synthetic sweep: 0 → 10 → 0
    voltages = list(range(0, 11)) + list(range(10, -1, -1))
    currents = [v * 1e-9 for v in voltages]  # Linear I-V

    df = pl.DataFrame({
        "Vg (V)": voltages,
        "I (A)": currents,
    })

    segments = detect_segments(df, voltage_col="Vg (V)")

    # Should detect 2 segments: forward (0→10) and return (10→0)
    assert len(segments) == 2
    assert segments[0].segment_type == "forward"
    assert segments[1].segment_type == "return"
    assert segments[0].v_start == 0
    assert segments[0].v_end == 10
    assert segments[1].v_start == 10
    assert segments[1].v_end == 0


def test_empty_dataframe():
    """Test handling of empty input"""
    df = pl.DataFrame({"Vg (V)": [], "I (A)": []})
    segments = detect_segments(df)
    assert len(segments) == 0


def test_noisy_data():
    """Test segment detection with noisy voltage"""
    # Sweep with noise
    clean = list(range(0, 11))
    noisy = [v + 0.0005 * (i % 2 - 0.5) for i, v in enumerate(clean)]

    df = pl.DataFrame({
        "Vg (V)": noisy,
        "I (A)": [v * 1e-9 for v in noisy],
    })

    # Should still detect 1 segment (forward sweep)
    # Noise should be filtered by dv_threshold
    segments = detect_segments(df, voltage_col="Vg (V)", dv_threshold=0.01)
    assert len(segments) == 1
```

Run tests:
```bash
pytest tests/test_ivg_preprocessing.py -v
```

### 8.2 Integration Testing

**Test the full pipeline:**

```bash
# 1. Stage (should already work)
python nanolab-pipeline.py stage \
  --raw-root data/01_raw \
  --stage-root data/02_stage/raw_measurements \
  --workers 4

# 2. Preprocess (your new code!)
python nanolab-pipeline.py preprocess \
  --stage-root data/02_stage/raw_measurements \
  --output-root data/03_intermediate/ivg_segments \
  --procedure IVg \
  --workers 4

# 3. Analyze
python src/analysis/IVg/ivg_analysis.py \
  --intermediate-root data/03_intermediate/ivg_segments \
  --date 2025-10-18 \
  --output-base-dir data/04_analysis

# 4. Visualize
python src/ploting/IVg/visualize_ivg.py \
  --stats-dir data/04_analysis/ivg_stats/2025-10-18_IVg \
  --output-dir plots/ivg/2025-10-18
```

### 8.3 Validation Checklist

- [ ] YAML schema defined in `config/procedures.yml`
- [ ] Pydantic models created in `src/models/parameters.py`
- [ ] Preprocessing script implements segmentation correctly
- [ ] Segments are labeled with meaningful `segment_type` values
- [ ] Output follows partitioning scheme: `proc=/date=/run_id=/segment=/`
- [ ] Analysis script loads segmented data correctly
- [ ] Statistics computed correctly (mean, std, counts)
- [ ] Derived quantities validated against known values
- [ ] Visualization generates publication-quality figures
- [ ] CLI integration works (can call from `nanolab-pipeline.py`)
- [ ] Idempotency verified (rerunning produces same results)
- [ ] Unit tests pass
- [ ] Integration test completes successfully
- [ ] Documentation updated (README, CLAUDE.md, CLI docs)

---

## Best Practices

### Code Organization

**Directory Structure:**
```
src/
    intermediate/{Procedure}/
        {procedure}_preprocessing_script.py    # Main preprocessing
        segment_utils.py                       # Shared utilities (optional)

    analysis/{Procedure}/
        {procedure}_analysis.py                # Main analysis
        compute_{derived_quantity}.py          # Additional analysis scripts

    ploting/{Procedure}/
        visualize_{procedure}.py               # Main visualization
        compare_{parameter}.py                 # Comparison plots
```

**Naming Conventions:**
- Scripts: `lowercase_with_underscores.py`
- Classes: `PascalCase`
- Functions: `lowercase_with_underscores()`
- Constants: `UPPER_CASE_WITH_UNDERSCORES`
- Procedures in paths: exact case from YAML (e.g., `IVg`, not `ivg`)

### Performance Optimization

**Parallel Processing:**
- Use `ProcessPoolExecutor` for CPU-bound tasks
- Set `max_workers` based on core count (typically 4-8)
- Limit Polars threads per worker (`POLARS_MAX_THREADS=2`)
- Avoid nested parallelism (don't parallelize inside parallel workers)

**Memory Efficiency:**
- Use `pl.scan_parquet()` with lazy evaluation when possible
- Process data in chunks if RAM-limited
- Use `.select()` to load only needed columns
- Use categorical types for repeated string values

**I/O Optimization:**
- Use atomic writes (`atomic_write_parquet()`)
- Partition data appropriately (not too many small files!)
- Use Parquet compression (`zstd` is good default)
- Batch writes when creating many small files

### Error Handling

**Graceful Degradation:**
```python
def process_single_run(run_path: Path, ...) -> dict:
    """Process with graceful error handling"""
    try:
        # Main processing logic
        df = pl.read_parquet(run_path)
        segments = detect_segments(df)

        if not segments:
            return {"status": "no_segments", "run_id": run_id}

        # Write segments...

        return {"status": "ok", "run_id": run_id, "segments": len(segments)}

    except FileNotFoundError as e:
        return {"status": "error", "run_id": run_id, "reason": f"File not found: {e}"}
    except Exception as e:
        # Log unexpected errors but don't crash the entire pipeline
        logger.error(f"Unexpected error processing {run_id}: {e}")
        return {"status": "error", "run_id": run_id, "reason": str(e)}
```

**Validation:**
- Use Pydantic for configuration validation
- Check DataFrame schemas before processing
- Validate physical constraints (e.g., voltage ranges, current sign)
- Log warnings for suspicious data (e.g., too few points, large noise)

### Documentation

**Code Documentation:**
- Module docstrings: Describe purpose, inputs, outputs
- Function docstrings: Args, Returns, Raises, Examples
- Inline comments: Explain *why*, not *what* (code explains what)
- Type hints: Use for all function signatures

**User Documentation:**
- CLI help text: Clear, concise command descriptions
- README examples: Show common workflows
- Tutorial notebooks: Jupyter notebooks for exploration
- Architecture docs: Explain design decisions

---

## Examples

### Example 1: IVg (Gate Voltage Sweep)

**Characteristics:**
- Sweep variable: `Vg (V)` (gate voltage)
- Measured: `I (A)` (drain current)
- Segments: forward (Vg_start → Vg_end), return (Vg_end → Vg_start)
- Analysis: Transfer characteristics, mobility, threshold voltage
- Visualization: Ids vs Vg (log scale), hysteresis

**Key differences from IV:**
- Different sweep axis (`Vg` instead of `Vsd`)
- Fixed `VDS` parameter (important for mobility calculation)
- Different derived quantities (mobility, Vth, Ion/Ioff)

**Segmentation:**
- Detect direction changes in `Vg (V)`
- May or may not cross Vg=0 (depends on sweep range)
- Typically bidirectional sweep

**Implementation Checklist:**
- [x] YAML schema: Same structure as IV, different parameter names
- [x] Pydantic models: `IVgPreprocessingParameters`, `IVgAnalysisParameters`
- [x] Preprocessing: Reuse `detect_voltage_segments()` with `voltage_col="Vg (V)"`
- [x] Analysis: Add mobility calculation, threshold extraction
- [x] Visualization: Transfer characteristic plot (log scale), hysteresis

### Example 2: ITt (Current vs Time with Temperature)

**Characteristics:**
- Time-series measurement
- Variables: `t (s)`, `I (A)`, `Plate T (degC)`, `Ambient T (degC)`
- Segments: laser ON/OFF periods (if `Laser ON+OFF period` > 0)
- Analysis: Photoresponse dynamics, rise/fall times, steady-state levels
- Visualization: Current vs time, temperature vs time

**Key differences from IV:**
- **Not a sweep!** Time-series data
- Segmentation based on laser state, not voltage direction
- Temperature as additional variable
- Transient analysis (rise time, fall time, decay constants)

**Segmentation Strategy:**
```python
def detect_laser_periods(
    df: pl.DataFrame,
    laser_period: float,
    sampling_time: float,
) -> List[SegmentInfo]:
    """
    Detect laser ON/OFF segments from ITt measurement.

    Args:
        df: DataFrame with 't (s)', 'I (A)', 'VL (V)' columns
        laser_period: Laser ON+OFF period (s)
        sampling_time: Measurement sampling time (s)

    Returns:
        List of segments labeled "laser_on" or "laser_off"
    """
    # Use VL (laser voltage) column to detect ON/OFF
    # VL > threshold → laser ON
    # VL ≈ 0 → laser OFF

    threshold = 0.5  # Volts
    df_analysis = df.with_columns([
        (pl.col("VL (V)") > threshold).alias("laser_state")
    ])

    # Detect state changes
    df_analysis = df_analysis.with_columns([
        pl.col("laser_state").shift(1).alias("prev_state")
    ])

    # Find transition indices
    transitions = df_analysis.filter(
        pl.col("laser_state") != pl.col("prev_state")
    )["_idx"].to_list()

    # Create segments between transitions
    segments = []
    for i, (start_idx, end_idx) in enumerate(zip(transitions[:-1], transitions[1:])):
        state = df_analysis[start_idx]["laser_state"]
        segment_type = "laser_on" if state else "laser_off"

        segments.append(SegmentInfo(
            segment_id=i,
            segment_type=segment_type,
            start_idx=start_idx,
            end_idx=end_idx - 1,
        ))

    return segments
```

**Analysis:**
- Compute mean current during ON vs OFF periods
- Calculate rise time (10% → 90% of steady state)
- Calculate fall time (90% → 10%)
- Fit exponential decay: `I(t) = I0 + A * exp(-t/tau)`
- Extract time constants

**Visualization:**
- Multi-panel: Current vs time, Temperature vs time, Laser state
- Highlight ON/OFF regions with shaded background
- Show exponential fit overlays
- Rise/fall time annotations

### Example 3: LaserCalibration (Simple Monotonic Sweep)

**Characteristics:**
- Sweep variable: `VL (V)` (laser voltage)
- Measured: `Power (W)` (optical power)
- **No segmentation needed!** (monotonic sweep, no hysteresis)
- Analysis: Power vs voltage calibration curve, linear fit
- Visualization: Power vs VL, fit residuals

**Simplified Pipeline:**
- **Skip Layer 3 preprocessing** - work directly with staged data!
- Analysis reads from Layer 2 (staged Parquet)
- Fit: `Power = slope * VL + offset`
- Compute R², residuals

**Implementation Note:**
Not all procedures need intermediate preprocessing. For simple monotonic sweeps without segmentation, you can skip Layer 3 and analyze directly from staged data.

---

## Summary

**Key Steps to Add New Procedure:**

1. **Define** - Add YAML schema (`config/procedures.yml`)
2. **Understand** - Analyze physical measurement structure
3. **Model** - Create Pydantic parameter classes (`src/models/parameters.py`)
4. **Segment** - Implement preprocessing script (`src/intermediate/{Proc}/`)
5. **Analyze** - Implement analysis script (`src/analysis/{Proc}/`)
6. **Visualize** - Create plotting scripts (`src/ploting/{Proc}/`)
7. **Integrate** - Update CLI wrappers and commands
8. **Test** - Unit tests, integration tests, validation
9. **Document** - Update README, CLAUDE.md, CLI docs

**Design Principles:**

- **Separation of Concerns**: Staging is generic, preprocessing is procedure-specific
- **4-Layer Architecture**: Raw → Stage → Intermediate → Analysis
- **Idempotency**: Safe to rerun any step
- **Type Safety**: Use Pydantic for configuration validation
- **Professional Code**: Clear naming, comprehensive docs, error handling
- **Performance**: Parallel processing, efficient I/O
- **Reproducibility**: Same input → same output

**Remember:**
- Study existing IV implementation as reference
- Start simple, add complexity incrementally
- Test with small datasets first
- Ask for help if segmentation logic is unclear
- Document your decisions and assumptions

---

**Questions or Issues?**

Refer to:
- `docs/4LAYER_COMPLETE.md` - Architecture overview
- `docs/CLI_ARCHITECTURE.md` - CLI technical details
- `src/intermediate/IV/iv_preprocessing_script.py` - Reference implementation
- `src/analysis/IV/aggregate_iv_stats.py` - Analysis example

Good luck extending the pipeline! 🔬
