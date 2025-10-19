"""
IV Curve Preprocessing Pipeline

Segments IV sweep measurements into distinct voltage sweep phases:
- Forward negative: 0 → -Vmax
- Return negative: -Vmax → 0
- Forward positive: 0 → +Vmax
- Return positive: +Vmax → 0

Reads from: 02_stage/raw_measurements/proc=IV/
Writes to: 03_intermediate/iv_segments/

Now with Pydantic validation for professional configuration management!
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import polars as pl

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.models.parameters import IntermediateParameters
from pydantic import ValidationError

# Add staging utilities to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "staging"))
from stage_utils import (
    ensure_dir,
    sha1_short,
    warn,
)

from stage_raw_measurements import (
    atomic_write_parquet
)

# ----------------------------- Config -----------------------------

DEFAULT_WORKERS = 6
DEFAULT_POLARS_THREADS = 1
DEFAULT_VOLTAGE_COL = "Vsd (V)"
DEFAULT_DV_THRESHOLD = 0.001  # Minimum voltage change to consider valid (filters noise)
DEFAULT_MIN_SEGMENT_POINTS = 5  # Minimum points to consider a valid segment


# ----------------------------- Segment Detection -----------------------------

@dataclass
class SegmentInfo:
    """
    Metadata for a detected sweep segment.
    
    Attributes:
        segment_id: Sequential segment number (0, 1, 2, ...)
        segment_type: Descriptive label (forward_negative, return_negative, etc.)
        start_idx: Starting row index in original DataFrame
        end_idx: Ending row index (inclusive)
        v_start: Starting voltage value
        v_end: Ending voltage value
        v_direction: "increasing" or "decreasing"
        point_count: Number of data points in segment
    """
    segment_id: int
    segment_type: str
    start_idx: int
    end_idx: int
    v_start: float
    v_end: float
    v_direction: str
    point_count: int


def detect_voltage_segments(
    df: pl.DataFrame,
    voltage_col: str = "Vsd (V)",
    dv_threshold: float = 0.001,
    min_points: int = 5
) -> List[SegmentInfo]:
    """
    Detect sweep segments based on voltage derivative sign changes and zero-crossings.
    
    Analyzes voltage progression to identify distinct sweep phases. A segment
    ends when either:
    1. The voltage direction reverses (derivative changes sign), OR
    2. The voltage crosses zero (changes sign from negative to positive or vice versa)
    
    Args:
        df: DataFrame with voltage measurements (must be time-ordered)
        voltage_col: Name of voltage column to analyze
        dv_threshold: Minimum |dV| to consider valid change (filters noise)
        min_points: Minimum points required for a valid segment
        
    Returns:
        List of SegmentInfo objects describing each detected segment
        
    Algorithm:
        1. Calculate voltage derivative (dV)
        2. Filter noise using threshold
        3. Detect sign changes in dV (direction reversals)
        4. Detect sign changes in V (zero-crossings)
        5. Label segments by quadrant and direction
        
    Example:
        Voltage sequence: [0, -1, -2, -3, -2, -1, 0, 1, 2, 1, 0]
        Segments detected:
        - Segment 0: 0→-3 (forward_negative)
        - Segment 1: -3→0 (return_negative)
        - Segment 2: 0→2 (forward_positive)   ← Detected via zero-crossing
        - Segment 3: 2→0 (return_positive)
        
    Note:
        Requires data to be sorted by time/measurement order.
        Handles noisy data by requiring dV > threshold for direction changes.
    """
    if df.height < min_points:
        warn(f"DataFrame has only {df.height} points, less than min_points={min_points}")
        return []
    
    if voltage_col not in df.columns:
        raise ValueError(f"Voltage column '{voltage_col}' not found in DataFrame")
    
    # Calculate voltage derivative
    df_analysis = df.with_row_count("_idx").with_columns([
        pl.col(voltage_col).diff().alias("dV"),
    ])
    
    # Extract as numpy/lists for analysis
    voltages = df_analysis[voltage_col].to_list()
    dvs = df_analysis["dV"].to_list()
    indices = df_analysis["_idx"].to_list()
    
    # Helper function to detect zero-crossing between two voltages
    def crosses_zero(v_prev: float, v_curr: float) -> bool:
        """Detect if voltage crosses zero between two consecutive points."""
        return (v_prev < 0 and v_curr >= 0) or (v_prev > 0 and v_curr <= 0)
    
    # Identify direction changes AND zero-crossings
    segments: List[SegmentInfo] = []
    current_start = 0
    current_direction = None
    prev_voltage = voltages[0] if voltages else 0
    
    for i in range(1, len(dvs)):
        dv = dvs[i]
        curr_voltage = voltages[i]
        
        # Determine if we should end the current segment
        should_end_segment = False
        
        # Check for direction change (if dV is significant)
        if dv is not None and abs(dv) >= dv_threshold:
            direction = "increasing" if dv > 0 else "decreasing"
            
            if current_direction is None:
                current_direction = direction
            elif direction != current_direction:
                should_end_segment = True
        
        # Check for zero-crossing (even if direction doesn't change)
        if crosses_zero(prev_voltage, curr_voltage):
            should_end_segment = True
        
        # End segment if boundary detected
        if should_end_segment:
            segment_length = i - current_start
            
            if segment_length >= min_points:
                v_start = voltages[current_start]
                v_end = voltages[i - 1]
                
                # Classify segment type
                seg_type = classify_segment_type(
                    v_start, v_end, current_direction, len(segments)
                )
                
                segments.append(SegmentInfo(
                    segment_id=len(segments),
                    segment_type=seg_type,
                    start_idx=current_start,
                    end_idx=i - 1,
                    v_start=v_start,
                    v_end=v_end,
                    v_direction=current_direction or "unknown",
                    point_count=segment_length,
                ))
            
            # Start new segment
            current_start = i
            # Update direction if dV is significant
            if dv is not None and abs(dv) >= dv_threshold:
                current_direction = "increasing" if dv > 0 else "decreasing"
        
        prev_voltage = curr_voltage
    
    # Handle final segment
    final_length = len(voltages) - current_start
    if final_length >= min_points:
        v_start = voltages[current_start]
        v_end = voltages[-1]
        seg_type = classify_segment_type(v_start, v_end, current_direction or "unknown", len(segments))
        
        segments.append(SegmentInfo(
            segment_id=len(segments),
            segment_type=seg_type,
            start_idx=current_start,
            end_idx=len(voltages) - 1,
            v_start=v_start,
            v_end=v_end,
            v_direction=current_direction or "unknown",
            point_count=final_length,
        ))
    
    return segments


def classify_segment_type(
    v_start: float,
    v_end: float,
    direction: str,
    segment_num: int
) -> str:
    """
    Classify segment by voltage range and direction.
    
    Determines descriptive label for a sweep segment based on starting/ending
    voltages and sweep direction.
    
    Args:
        v_start: Starting voltage
        v_end: Ending voltage
        direction: "increasing" or "decreasing"
        segment_num: Segment number (for fallback naming)
        
    Returns:
        Segment type label:
        - "forward_negative": Sweeping into negative voltages (0→-V)
        - "return_negative": Returning from negative voltages (-V→0)
        - "forward_positive": Sweeping into positive voltages (0→+V)
        - "return_positive": Returning from positive voltages (+V→0)
        - "unknown_N": Fallback for ambiguous cases
        
    Example:
        >>> classify_segment_type(0, -5, "decreasing", 0)
        'forward_negative'
        >>> classify_segment_type(-5, 0, "increasing", 1)
        'return_negative'
        >>> classify_segment_type(0, 5, "increasing", 2)
        'forward_positive'
    """
    v_min = min(v_start, v_end)
    v_max = max(v_start, v_end)
    
    # Both voltages negative
    if v_max <= 0:
        if direction == "decreasing":
            return "forward_negative"  # Going more negative
        else:
            return "return_negative"   # Coming back toward zero
    
    # Both voltages positive
    elif v_min >= 0:
        if direction == "increasing":
            return "forward_positive"  # Going more positive
        else:
            return "return_positive"   # Coming back toward zero
    
    # Crosses zero (mixed)
    else:
        if direction == "decreasing":
            return "forward_negative"  # Entering negative region
        else:
            return "forward_positive"  # Entering positive region
    
    # Fallback
    return f"unknown_{segment_num}"


def add_segment_columns(
    df: pl.DataFrame,
    segments: List[SegmentInfo]
) -> pl.DataFrame:
    """
    Add segment metadata columns to DataFrame.
    
    Enriches DataFrame with segment identification and metadata columns
    based on detected segment boundaries.
    
    Args:
        df: Original DataFrame
        segments: List of detected segments
        
    Returns:
        DataFrame with added columns:
        - segment_id: Integer segment number
        - segment_type: Descriptive segment label
        - segment_v_start: Starting voltage of segment
        - segment_v_end: Ending voltage of segment
        - segment_direction: "increasing" or "decreasing"
        - point_in_segment: Point number within segment (0-based)
        
    Example:
        >>> df = pl.DataFrame({"V": [0, -1, -2, -1, 0]})
        >>> segments = detect_voltage_segments(df, "V")
        >>> df_enriched = add_segment_columns(df, segments)
        >>> df_enriched["segment_id"].to_list()
        [0, 0, 0, 1, 1]
    """
    # Initialize segment columns with nulls
    df = df.with_row_count("_row_idx").with_columns([
        pl.lit(None, dtype=pl.Int64).alias("segment_id"),
        pl.lit(None, dtype=pl.Utf8).alias("segment_type"),
        pl.lit(None, dtype=pl.Float64).alias("segment_v_start"),
        pl.lit(None, dtype=pl.Float64).alias("segment_v_end"),
        pl.lit(None, dtype=pl.Utf8).alias("segment_direction"),
        pl.lit(None, dtype=pl.Int64).alias("point_in_segment"),
    ])
    
    # Fill segment info for each detected segment
    for seg in segments:
        mask = (pl.col("_row_idx") >= seg.start_idx) & (pl.col("_row_idx") <= seg.end_idx)
        
        df = df.with_columns([
            pl.when(mask)
              .then(pl.lit(seg.segment_id))
              .otherwise(pl.col("segment_id"))
              .alias("segment_id"),
            
            pl.when(mask)
              .then(pl.lit(seg.segment_type))
              .otherwise(pl.col("segment_type"))
              .alias("segment_type"),
            
            pl.when(mask)
              .then(pl.lit(seg.v_start))
              .otherwise(pl.col("segment_v_start"))
              .alias("segment_v_start"),
            
            pl.when(mask)
              .then(pl.lit(seg.v_end))
              .otherwise(pl.col("segment_v_end"))
              .alias("segment_v_end"),
            
            pl.when(mask)
              .then(pl.lit(seg.v_direction))
              .otherwise(pl.col("segment_direction"))
              .alias("segment_direction"),
            
            pl.when(mask)
              .then(pl.col("_row_idx") - pl.lit(seg.start_idx))
              .otherwise(pl.col("point_in_segment"))
              .alias("point_in_segment"),
        ])
    
    # Remove temporary index column
    df = df.drop("_row_idx")
    
    return df


# ----------------------------- Processing Pipeline -----------------------------

def process_iv_run(
    parquet_path: Path,
    output_root: Path,
    voltage_col: str,
    dv_threshold: float,
    min_points: int,
    force: bool,
    events_dir: Path,
) -> Dict[str, Any]:
    """
    Process a single IV sweep run: detect segments and write partitioned outputs.
    
    Main worker function that:
    1. Reads staged IV sweep Parquet file
    2. Detects voltage sweep segments
    3. Adds segment metadata columns
    4. Writes separate Parquet files per segment
    5. Logs processing event
    
    Args:
        parquet_path: Path to staged Parquet file from 02_stage
        output_root: Root directory for segmented output (03_intermediate)
        voltage_col: Name of voltage column to analyze
        dv_threshold: Voltage change threshold for noise filtering
        min_points: Minimum points for valid segment
        force: If True, overwrite existing outputs
        events_dir: Directory for event JSON files
        
    Returns:
        Event dictionary with processing results:
        - status: "ok", "skipped", or "error"
        - run_id: Unique run identifier
        - segments_detected: Number of segments found
        - segments_written: Number of segment files created
        - output_paths: List of created Parquet file paths
        
    Output structure:
        03_intermediate/iv_segments/
        └── proc=iv_sweep/
            └── date=YYYY-MM-DD/
                └── run_id=HASH/
                    ├── segment=0/
                    │   └── part-000.parquet
                    ├── segment=1/
                    │   └── part-000.parquet
                    └── ...
                    
    Example event (success):
        {
            "ts": "2025-01-15T10:30:00Z",
            "status": "ok",
            "run_id": "a1b2c3d4e5f6g7h8",
            "source_path": "/stage/proc=iv_sweep/.../part-000.parquet",
            "segments_detected": 4,
            "segments_written": 4,
            "output_paths": ["/intermediate/.../segment=0/part-000.parquet", ...]
        }
        
    Note:
        - Skips processing if all segment outputs exist and force=False
        - Handles errors gracefully, writing reject records
        - Preserves all original columns plus segment metadata
    """
    try:
        # Extract partition info from path
        # Expected: .../proc=X/date=Y/run_id=Z/part-000.parquet
        parts = parquet_path.parts
        proc_part = next((p for p in parts if p.startswith("proc=")), None)
        date_part = next((p for p in parts if p.startswith("date=")), None)
        run_id_part = next((p for p in parts if p.startswith("run_id=")), None)
        
        if not all([proc_part, date_part, run_id_part]):
            raise ValueError(f"Could not parse partition structure from path: {parquet_path}")
        
        run_id = run_id_part.split("=")[1]
        
        # Check if outputs exist (skip if not forcing)
        base_output_dir = output_root / proc_part / date_part / run_id_part
        
        if not force and base_output_dir.exists():
            # Check if any segment files exist
            existing_segments = list(base_output_dir.glob("segment=*/part-000.parquet"))
            if existing_segments:
                event = {
                    "ts": dt.datetime.now(tz=dt.timezone.utc),
                    "status": "skipped",
                    "run_id": run_id,
                    "source_path": str(parquet_path),
                    "segments_detected": len(existing_segments),
                    "segments_written": 0,
                    "output_paths": [str(p) for p in existing_segments],
                }
                
                # Write event
                ev_path = events_dir / f"segment-{run_id}.json"
                ensure_dir(ev_path.parent)
                with ev_path.open("w", encoding="utf-8") as f:
                    json.dump(event, f, ensure_ascii=False, default=str)
                
                return event
        
        # Read staged data
        df = pl.read_parquet(parquet_path)
        
        if df.height == 0:
            raise ValueError("Empty DataFrame")
        
        if voltage_col not in df.columns:
            raise ValueError(f"Voltage column '{voltage_col}' not found. Available: {df.columns}")
        
        # Detect segments
        segments = detect_voltage_segments(df, voltage_col, dv_threshold, min_points)
        
        if not segments:
            raise ValueError(f"No valid segments detected (min_points={min_points}, threshold={dv_threshold})")
        
        # Add segment columns
        df_segmented = add_segment_columns(df, segments)
        
        # Write separate file per segment
        output_paths = []
        for seg in segments:
            segment_df = df_segmented.filter(pl.col("segment_id") == seg.segment_id)
            
            if segment_df.height == 0:
                warn(f"Segment {seg.segment_id} is empty after filtering, skipping")
                continue
            
            seg_dir = base_output_dir / f"segment={seg.segment_id}"
            seg_file = seg_dir / "part-000.parquet"
            
            atomic_write_parquet(segment_df, seg_file)
            output_paths.append(seg_file)
        
        event = {
            "ts": dt.datetime.now(tz=dt.timezone.utc),
            "status": "ok",
            "run_id": run_id,
            "source_path": str(parquet_path),
            "segments_detected": len(segments),
            "segments_written": len(output_paths),
            "output_paths": [str(p) for p in output_paths],
            "segment_details": [
                {
                    "segment_id": s.segment_id,
                    "type": s.segment_type,
                    "v_range": f"{s.v_start:.3f}→{s.v_end:.3f}V",
                    "points": s.point_count,
                }
                for s in segments
            ],
        }
        
        # Write event
        ev_path = events_dir / f"segment-{run_id}.json"
        ensure_dir(ev_path.parent)
        with ev_path.open("w", encoding="utf-8") as f:
            json.dump(event, f, ensure_ascii=False, default=str, indent=2)
        
        return event
        
    except Exception as e:
        error_event = {
            "ts": dt.datetime.now(tz=dt.timezone.utc),
            "status": "error",
            "source_path": str(parquet_path),
            "error": str(e),
        }
        
        # Write error event
        error_hash = sha1_short(str(parquet_path), 12)
        ev_path = events_dir / f"segment-error-{error_hash}.json"
        ensure_dir(ev_path.parent)
        with ev_path.open("w", encoding="utf-8") as f:
            json.dump(error_event, f, ensure_ascii=False, default=str, indent=2)
        
        return error_event


def discover_iv_runs(stage_root: Path) -> List[Path]:
    """
    Discover all IV sweep Parquet files in staged data.
    
    Recursively searches for IV sweep measurements in the staged data directory.
    
    Args:
        stage_root: Root of staged data (02_stage/raw_measurements)
        
    Returns:
        Sorted list of paths to IV sweep Parquet files
        
    Example:
        >>> runs = discover_iv_runs(Path("02_stage/raw_measurements"))
        >>> len(runs)
        48
        >>> runs[0]
        Path('02_stage/.../proc=iv_sweep/date=2025-01-15/run_id=abc123/part-000.parquet')
        
    Note:
        Only returns files matching pattern: proc=iv_sweep/**/*.parquet
    """
    iv_proc_dir = stage_root / "proc=IV"
    
    if not iv_proc_dir.exists():
        warn(f"IV sweep directory not found: {iv_proc_dir}")
        return []
    
    parquet_files = sorted(iv_proc_dir.rglob("*.parquet"))
    
    # Filter out manifest files
    parquet_files = [p for p in parquet_files if "_manifest" not in p.parts]
    
    return parquet_files


def merge_events_to_manifest(events_dir: Path, manifest_path: Path) -> None:
    """
    Consolidate segment event JSON files into unified Parquet manifest.
    
    Reads all segment-*.json files and merges them with any existing manifest.
    Provides audit trail of all segmentation operations.
    
    Args:
        events_dir: Directory containing event JSON files
        manifest_path: Output path for consolidated manifest
        
    Manifest schema:
        - ts: Processing timestamp
        - status: "ok", "skipped", or "error"
        - run_id: Unique run identifier
        - source_path: Original staged Parquet path
        - segments_detected: Number of segments found
        - segments_written: Number of output files created
        - output_paths: List of created segment files
        - segment_details: Per-segment metadata (optional)
        
    Note:
        Deduplicates by (run_id, ts, status), keeping latest occurrence.
    """
    ev_files = sorted(events_dir.glob("segment-*.json"))
    
    if not ev_files:
        warn("No event files found to merge")
        return
    
    rows = []
    for e in ev_files:
        try:
            data = json.loads(e.read_text(encoding="utf-8"))
            # Flatten segment_details if present (optional)
            if "segment_details" in data:
                data["segment_details"] = json.dumps(data["segment_details"])
            rows.append(data)
        except Exception as ex:
            warn(f"Failed to read event file {e}: {ex}")
            continue
    
    if not rows:
        warn("No valid event data to merge")
        return
    
    df = pl.DataFrame(rows)
    ensure_dir(manifest_path.parent)
    
    if manifest_path.exists():
        prev = pl.read_parquet(manifest_path)
        all_df = pl.concat([prev, df], how="vertical_relaxed")
        
        # Deduplicate
        if "run_id" in all_df.columns:
            all_df = all_df.unique(subset=["run_id", "ts", "status"], keep="last")
        
        all_df.write_parquet(manifest_path)
    else:
        df.write_parquet(manifest_path)


# ----------------------------- Pydantic Pipeline Function -----------------------------

def run_iv_preprocessing(params: IntermediateParameters) -> None:
    """
    Run IV preprocessing pipeline with Pydantic-validated parameters.

    Args:
        params: Validated IntermediateParameters instance

    Example:
        >>> from models.parameters import IntermediateParameters
        >>> params = IntermediateParameters(
        ...     stage_root=Path("data/02_stage/raw_measurements"),
        ...     output_root=Path("data/03_intermediate"),
        ...     procedure="IV",
        ...     workers=8
        ... )
        >>> run_iv_preprocessing(params)
    """
    # Extract validated parameters
    stage_root = params.stage_root
    output_root = params.get_output_dir()  # Use helper method
    voltage_col = params.voltage_col
    dv_threshold = params.dv_threshold
    min_points = params.min_segment_points
    workers = params.workers
    polars_threads = params.polars_threads
    force = params.force
    events_dir = params.events_dir
    manifest_path = params.manifest

    # Create output directories
    ensure_dir(output_root)
    ensure_dir(events_dir)
    ensure_dir(manifest_path.parent)

    # Set Polars threading
    os.environ["POLARS_MAX_THREADS"] = str(polars_threads)

    # Discover IV runs
    iv_runs = discover_iv_runs(stage_root)
    print(f"[info] discovered {len(iv_runs)} IV sweep runs in {stage_root}")

    if not iv_runs:
        print("[done] no IV sweep data found")
        return

    # Process runs in parallel
    ok = skipped = errors = 0

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = []
        for run_path in iv_runs:
            fut = ex.submit(
                process_iv_run,
                run_path,
                output_root,
                voltage_col,
                dv_threshold,
                min_points,
                force,
                events_dir,
            )
            futures.append((run_path, fut))

        # Collect results
        for i, (run_path, fut) in enumerate(futures, 1):
            try:
                event = fut.result()
            except Exception as e:
                errors += 1
                print(f"[{i:04d}]   ERROR {run_path} :: {e}")
                continue

            status = event.get("status")

            if status == "ok":
                ok += 1
                seg_count = event.get("segments_detected", 0)
                run_id = event.get("run_id", "unknown")
                print(
                    f"[{i:04d}]      OK run_id={run_id:<16} "
                    f"segments={seg_count}  → {event.get('segments_written', 0)} files written"
                )
            elif status == "skipped":
                skipped += 1
                seg_count = event.get("segments_detected", 0)
                run_id = event.get("run_id", "unknown")
                print(
                    f"[{i:04d}] SKIPPED run_id={run_id:<16} "
                    f"segments={seg_count}  (already exists)"
                )
            elif status == "error":
                errors += 1
                print(f"[{i:04d}]   ERROR {run_path} :: {event.get('error', 'unknown')}")

    # Merge events to manifest
    print("[info] merging events to manifest...")
    merge_events_to_manifest(events_dir, manifest_path)

    # Summary
    total = len(iv_runs)
    print(
        f"[done] segmentation complete  |  "
        f"ok={ok}  skipped={skipped}  errors={errors}  total={total}"
    )
    print(f"[info] manifest written to: {manifest_path}")


# ----------------------------- CLI Orchestration -----------------------------

def main() -> None:
    """
    Main entry point with both Pydantic and legacy argparse support.

    Supports two modes:
    1. JSON config file (--config) - Recommended
    2. Legacy argparse - Backward compatibility

    Command-line arguments:
        --config: Path to JSON configuration file (Pydantic mode)

        OR legacy arguments:
        --stage-root: Staged data directory
        --output-root: Intermediate output directory
        --procedure: Procedure name (IV, IVg, etc.)
        --voltage-col: Voltage column name
        --dv-threshold: Noise threshold
        --min-points: Minimum points per segment
        --workers: Number of parallel workers
        --polars-threads: Polars threads per worker
        --force: Overwrite existing files
    """
    ap = argparse.ArgumentParser(
        description="Segment IV sweeps into voltage phases with Pydantic validation.",
        epilog="""
Examples:
  # Using JSON config (recommended)
  python iv_preprocessing_script.py --config config/intermediate.json

  # Using command-line arguments (legacy)
  python iv_preprocessing_script.py \\
    --stage-root data/02_stage/raw_measurements \\
    --output-root data/03_intermediate \\
    --procedure IV \\
    --workers 8 --force
        """
    )

    # Pydantic mode
    ap.add_argument("--config", type=Path, help="Path to JSON configuration file (Pydantic mode)")

    # Legacy arguments
    ap.add_argument("--stage-root", type=Path, help="Staged data root")
    ap.add_argument("--output-root", type=Path, help="Output root for intermediate data")
    ap.add_argument("--procedure", type=str, default="IV", help="Procedure name (IV, IVg, IVgT)")
    ap.add_argument("--voltage-col", type=str, default=DEFAULT_VOLTAGE_COL, help="Voltage column name")
    ap.add_argument("--dv-threshold", type=float, default=DEFAULT_DV_THRESHOLD, help="Voltage change threshold")
    ap.add_argument("--min-points", type=int, default=DEFAULT_MIN_SEGMENT_POINTS, help="Minimum points per segment")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Number of parallel workers")
    ap.add_argument("--polars-threads", type=int, default=DEFAULT_POLARS_THREADS, help="Polars threads per worker")
    ap.add_argument("--force", action="store_true", help="Overwrite existing files")
    ap.add_argument("--events-dir", type=Path, help="Event JSON directory")
    ap.add_argument("--manifest", type=Path, help="Manifest file path")

    args = ap.parse_args()

    try:
        # Mode 1: JSON config file (Pydantic)
        if args.config:
            print(f"[info] Loading configuration from {args.config}")
            params = IntermediateParameters.model_validate_json(args.config.read_text())
            print("[info] Configuration validated successfully")

        # Mode 2: Legacy argparse
        elif args.stage_root and args.output_root:
            print("[info] Using command-line arguments (creating Pydantic parameters)")
            params = IntermediateParameters(
                stage_root=args.stage_root,
                output_root=args.output_root,
                procedure=args.procedure,
                voltage_col=args.voltage_col,
                dv_threshold=args.dv_threshold,
                min_segment_points=args.min_points,
                workers=args.workers,
                polars_threads=args.polars_threads,
                force=args.force,
                events_dir=args.events_dir,
                manifest=args.manifest,
            )

        else:
            ap.print_help()
            print("\n[error] Must provide either --config or (--stage-root and --output-root)")
            sys.exit(1)

        # Run pipeline with validated parameters
        run_iv_preprocessing(params)

    except ValidationError as e:
        print(f"\n[error] Parameter validation failed:")
        print(e)
        sys.exit(1)
    except Exception as e:
        print(f"\n[error] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()