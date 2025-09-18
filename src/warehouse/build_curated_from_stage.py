# file: build_curated_from_stage.py
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, List, Set, Tuple, Union

import polars as pl
import yaml

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None  # type: ignore


# ----------------------------- Config -----------------------------

DEFAULT_LOCAL_TZ = "America/Santiago"
DEFAULT_WORKERS = 4
DEFAULT_POLARS_THREADS = 2

# Canonical column name mapping (YAML names → SI units)
CANONICAL_MAPPING = {
    "Vg (V)": "VG_V",
    "Vsd (V)": "VD_V", 
    "VL (V)": "VL_V",
    "I (A)": "I_A",
    "t (s)": "t_s",
    "Time (s)": "t_s",
    "Plate T (degC)": "plate_C",
    "Ambient T (degC)": "ambient_C",
    "Ambient Temperature (degC)": "ambient_C",
    "Clock (ms)": "clock_ms",
    "Clock": "clock_ms",
}

# Procedure classification
TIME_SERIES_PROCS = {"It", "ITt", "Tt"}
SWEEP_PROCS = {"IV", "IVg", "IVgT", "LaserCalibration"}

# Time series columns (optional ones will be null if missing)
TS_COLUMNS = ["run_id", "t_s", "I_A", "VG_V", "VD_V", "VL_V", "plate_C", "ambient_C", "clock_ms"]

# Sweep fact columns 
SWEEP_AXES = ["VG_V", "VD_V", "VL_V"]
SWEEP_RESPONSES = ["I_A", "plate_C", "ambient_C", "clock_ms"]
SWEEP_COLUMNS = ["run_id"] + SWEEP_AXES + SWEEP_RESPONSES


def warn(msg: str) -> None:
    print(f"[warn] {msg}", file=sys.stderr)

def info(msg: str) -> None:
    print(f"[info] {msg}", file=sys.stdout)

def sha1_short(s: str, n: int = 16) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:n]

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def local_date_for_partition(ts_utc: dt.datetime, tz_name: str) -> str:
    if tz_name and ZoneInfo is not None:
        try:
            return ts_utc.astimezone(ZoneInfo(tz_name)).date().isoformat()
        except Exception:
            pass
    return ts_utc.date().isoformat()

def month_partition(date_str: str) -> str:
    """Convert YYYY-MM-DD to YYYY-MM for monthly partitioning"""
    return date_str[:7]

def parse_datetime_any(x: Any) -> Optional[dt.datetime]:
    if x is None:
        return None
    if isinstance(x, dt.datetime):
        return x.astimezone(dt.timezone.utc)
    if isinstance(x, (int, float)):
        try:
            return dt.datetime.fromtimestamp(float(x), tz=dt.timezone.utc)
        except Exception:
            return None
    s = str(x).strip()
    try:
        return dt.datetime.fromtimestamp(float(s), tz=dt.timezone.utc)
    except Exception:
        pass
    try:
        d = dt.datetime.fromisoformat(s)
        if d.tzinfo is None:
            d = d.replace(tzinfo=dt.timezone.utc)
        return d.astimezone(dt.timezone.utc)
    except Exception:
        return None


# ----------------------------- YAML Procedures Schema -----------------------------

@dataclass
class ProcSpec:
    params: Dict[str, str]
    meta: Dict[str, str]
    data: Dict[str, str]

_PROC_CACHE: Dict[str, ProcSpec] | None = None
_PROC_YAML_PATH: Path | None = None

def load_procedures_yaml(path: Path) -> Dict[str, ProcSpec]:
    with path.open("r", encoding="utf-8") as f:
        y = yaml.safe_load(f) or {}
    procs = {}
    root = y.get("procedures", {}) or {}
    for name, blocks in root.items():
        procs[name] = ProcSpec(
            params=(blocks.get("Parameters") or {}),
            meta=(blocks.get("Metadata") or {}),
            data=(blocks.get("Data") or {}),
        )
    return procs

def get_procs_cached(path: Path) -> Dict[str, ProcSpec]:
    global _PROC_CACHE, _PROC_YAML_PATH
    if _PROC_CACHE is None or _PROC_YAML_PATH != path:
        _PROC_CACHE = load_procedures_yaml(path)
        _PROC_YAML_PATH = path
    return _PROC_CACHE


# ----------------------------- Stage File Discovery -----------------------------

@dataclass
class StageFile:
    path: Path
    proc: str
    date: str
    run_id: str
    size: int
    mtime: float

def discover_stage_files(stage_root: Path) -> List[StageFile]:
    """Find all staged parquet files in proc=*/date=*/run_id=*/part-000.parquet structure"""
    files = []
    pattern = re.compile(r"proc=([^/]+)/date=(\d{4}-\d{2}-\d{2})/run_id=([^/]+)/part-000\.parquet$")
    
    for parquet_file in stage_root.rglob("part-000.parquet"):
        rel_path = parquet_file.relative_to(stage_root)
        match = pattern.search(str(rel_path))
        if match:
            proc, date, run_id = match.groups()
            stat = parquet_file.stat()
            files.append(StageFile(
                path=parquet_file,
                proc=proc,
                date=date, 
                run_id=run_id,
                size=stat.st_size,
                mtime=stat.st_mtime
            ))
    
    files.sort(key=lambda x: (x.proc, x.date, x.run_id))
    return files


# ----------------------------- Data Classification & Transformation -----------------------------

def classify_procedure(proc: str, df: pl.DataFrame) -> str:
    """Classify procedure as time-series, sweep, or other based on data columns"""
    columns = set(df.columns)
    canonical_cols = set()
    
    # Map to canonical names to check
    for col in columns:
        canonical_cols.add(CANONICAL_MAPPING.get(col, col))
    
    # Override classification by procedure name first
    if proc in TIME_SERIES_PROCS:
        return "time_series"
    elif proc in SWEEP_PROCS:
        return "sweep"
    
    # Classification by data fields
    has_time = "t_s" in canonical_cols or any("time" in col.lower() for col in canonical_cols)
    has_sweep = any(axis in canonical_cols for axis in SWEEP_AXES)
    
    if has_time:
        return "time_series"
    elif has_sweep:
        return "sweep"
    else:
        return "other"

def apply_canonical_mapping(df: pl.DataFrame) -> pl.DataFrame:
    """Rename columns to canonical SI unit names"""
    rename_map = {}
    for col in df.columns:
        if col in CANONICAL_MAPPING:
            rename_map[col] = CANONICAL_MAPPING[col]
    
    if rename_map:
        df = df.rename(rename_map)
    return df

def extract_runs_metadata(df: pl.DataFrame, stage_file: StageFile) -> Dict[str, Any]:
    """Extract runs metadata from first row of staged data"""
    if df.height == 0:
        raise ValueError("Empty dataframe")
    
    first_row = df.head(1).to_dicts()[0]
    
    # Parse start_dt
    start_dt = None
    if "start_dt" in first_row and first_row["start_dt"] is not None:
        start_dt = parse_datetime_any(first_row["start_dt"])
    
    # Create device_id hash
    chip_group = first_row.get("chip_group", "")
    chip_number = first_row.get("chip_number", "")
    device_id = sha1_short(f"{chip_group}|{chip_number}", 12) if chip_group or chip_number else None
    
    # Compute with_light flag
    wavelength_nm = first_row.get("wavelength_nm")
    laser_voltage_V = first_row.get("laser_voltage_V")
    with_light = (wavelength_nm is not None and 
                  laser_voltage_V is not None and 
                  laser_voltage_V != 0.0)
    
    metadata = {
        "run_id": stage_file.run_id,
        "proc": stage_file.proc,
        "device_id": device_id,
        "start_dt": start_dt,
        "chip_group": chip_group,
        "chip_number": chip_number,
        "sample": first_row.get("sample"),
        "procedure_version": first_row.get("procedure_version"),
        "wavelength_nm": wavelength_nm,
        "laser_voltage_V": laser_voltage_V,
        "with_light": with_light,
        "source_stage_path": str(stage_file.path),
        "n_rows": df.height,
        "stage_size": stage_file.size,
        "stage_mtime": dt.datetime.fromtimestamp(stage_file.mtime, tz=dt.timezone.utc),
    }
    
    return metadata

def prepare_ts_fact(df: pl.DataFrame, run_id: str) -> Optional[pl.DataFrame]:
    """Prepare time series fact table data"""
    df_canonical = apply_canonical_mapping(df)
    
    # Check if we have time axis
    if "t_s" not in df_canonical.columns:
        return None
    
    # Select only the columns we want, filling missing with null
    select_exprs = []
    for col in TS_COLUMNS:
        if col == "run_id":
            select_exprs.append(pl.lit(run_id).alias("run_id"))
        elif col in df_canonical.columns:
            select_exprs.append(pl.col(col))
        else:
            select_exprs.append(pl.lit(None).alias(col))
    
    result = df_canonical.select(select_exprs)
    
    # Validate time axis is non-decreasing
    if result.height > 1:
        time_diff = result.select(pl.col("t_s").diff().min()).item()
        if time_diff is not None and time_diff < -1e-9:  # Allow small numerical errors
            warn(f"Time axis not monotonic for run_id {run_id}")
    
    return result

def prepare_sweep_fact(df: pl.DataFrame, run_id: str) -> Optional[pl.DataFrame]:
    """Prepare sweep fact table data"""
    df_canonical = apply_canonical_mapping(df)
    
    # Check if we have sweep axes
    has_sweep_axis = any(axis in df_canonical.columns for axis in SWEEP_AXES)
    if not has_sweep_axis:
        return None
    
    # Select columns
    select_exprs = []
    for col in SWEEP_COLUMNS:
        if col == "run_id":
            select_exprs.append(pl.lit(run_id).alias("run_id"))
        elif col in df_canonical.columns:
            select_exprs.append(pl.col(col))
        else:
            select_exprs.append(pl.lit(None).alias(col))
    
    result = df_canonical.select(select_exprs)
    return result


# ----------------------------- Atomic Write Operations -----------------------------

def atomic_write_parquet(df: pl.DataFrame, out_file: Path) -> None:
    """Write parquet file atomically using temporary file"""
    ensure_dir(out_file.parent)
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=out_file.parent, suffix=".parquet") as tmp:
        tmp_path = Path(tmp.name)
    try:
        df.write_parquet(tmp_path)
        tmp_path.replace(out_file)
    except Exception:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        raise


# ----------------------------- Processing Worker -----------------------------

def process_stage_file_task(
    stage_file_data: Dict[str, Any],
    warehouse_root_str: str,
    procedures_yaml_str: str,
    local_tz: str,
    force: bool
) -> Dict[str, Any]:
    """Process a single stage file into warehouse tables"""
    
    # Reconstruct stage file object
    stage_file = StageFile(
        path=Path(stage_file_data["path"]),
        proc=stage_file_data["proc"],
        date=stage_file_data["date"],
        run_id=stage_file_data["run_id"],
        size=stage_file_data["size"],
        mtime=stage_file_data["mtime"]
    )
    
    warehouse_root = Path(warehouse_root_str)
    procedures_yaml = Path(procedures_yaml_str)
    
    try:
        # Load procedures schema
        procs = get_procs_cached(procedures_yaml)
        
        # Read staged data
        df = pl.read_parquet(stage_file.path)
        if df.height == 0:
            raise ValueError("Empty staged data")
        
        # Extract metadata
        metadata = extract_runs_metadata(df, stage_file)
        start_dt = metadata.get("start_dt")
        
        if not start_dt:
            raise ValueError("No valid start_dt found")
        
        # Determine monthly partition
        month_part = month_partition(local_date_for_partition(start_dt, local_tz))
        
        results = {
            "run_id": stage_file.run_id,
            "proc": stage_file.proc,
            "status": "ok",
            "month_part": month_part,
            "tables_written": []
        }
        
        # Write runs_metadata
        runs_meta_dir = warehouse_root / "runs_metadata" / f"date={month_part}"
        runs_meta_file = runs_meta_dir / f"runs-{stage_file.run_id}.parquet"
        
        if not runs_meta_file.exists() or force:
            meta_df = pl.DataFrame([metadata])
            atomic_write_parquet(meta_df, runs_meta_file)
            results["tables_written"].append("runs_metadata")
        
        # Classify and write fact tables
        classification = classify_procedure(stage_file.proc, df)
        
        if classification == "time_series":
            ts_data = prepare_ts_fact(df, stage_file.run_id)
            if ts_data is not None and ts_data.height > 0:
                ts_dir = warehouse_root / "ts_fact" / f"date={month_part}"
                ts_file = ts_dir / f"ts-{stage_file.run_id}.parquet"
                
                if not ts_file.exists() or force:
                    atomic_write_parquet(ts_data, ts_file)
                    results["tables_written"].append("ts_fact")
        
        elif classification == "sweep":
            sweep_data = prepare_sweep_fact(df, stage_file.run_id)
            if sweep_data is not None and sweep_data.height > 0:
                sweep_dir = warehouse_root / "sweep_fact" / f"date={month_part}"
                sweep_file = sweep_dir / f"sw-{stage_file.run_id}.parquet"
                
                if not sweep_file.exists() or force:
                    atomic_write_parquet(sweep_data, sweep_file)
                    results["tables_written"].append("sweep_fact")
        
        results["classification"] = classification
        results["n_rows"] = df.height
        
        return results
        
    except Exception as e:
        return {
            "run_id": stage_file.run_id,
            "proc": stage_file.proc,
            "status": "error",
            "error": str(e),
            "source_path": str(stage_file.path)
        }


# ----------------------------- Incremental Processing -----------------------------

@dataclass
class ProcessingManifest:
    processed_files: Dict[str, Dict[str, Any]]  # run_id -> file info
    schema_hash: str
    last_updated: dt.datetime

def load_warehouse_manifest(manifest_path: Path) -> Optional[ProcessingManifest]:
    """Load existing warehouse processing manifest"""
    if not manifest_path.exists():
        return None
    
    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        return ProcessingManifest(
            processed_files=data.get("processed_files", {}),
            schema_hash=data.get("schema_hash", ""),
            last_updated=dt.datetime.fromisoformat(data.get("last_updated", dt.datetime.now().isoformat()))
        )
    except Exception:
        return None

def save_warehouse_manifest(manifest_path: Path, manifest: ProcessingManifest) -> None:
    """Save warehouse processing manifest"""
    ensure_dir(manifest_path.parent)
    data = {
        "processed_files": manifest.processed_files,
        "schema_hash": manifest.schema_hash,
        "last_updated": manifest.last_updated.isoformat()
    }
    
    with tempfile.NamedTemporaryFile("w", delete=False, dir=manifest_path.parent, encoding="utf-8") as tmp:
        json.dump(data, tmp, ensure_ascii=False, indent=2)
        tmp_path = Path(tmp.name)
    
    tmp_path.replace(manifest_path)

def compute_schema_hash(procedures_yaml: Path) -> str:
    """Compute hash of procedures YAML for change detection"""
    return sha1_short(procedures_yaml.read_text(encoding="utf-8"))

def needs_processing(stage_file: StageFile, manifest: Optional[ProcessingManifest], schema_hash: str, force: bool) -> bool:
    """Determine if a stage file needs processing"""
    if force:
        return True
    
    if manifest is None:
        return True
    
    if manifest.schema_hash != schema_hash:
        return True
    
    file_info = manifest.processed_files.get(stage_file.run_id)
    if file_info is None:
        return True
    
    # Check if file changed
    if (file_info.get("size") != stage_file.size or 
        abs(file_info.get("mtime", 0) - stage_file.mtime) > 1):
        return True
    
    return False


# ----------------------------- Main Orchestration -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build curated data warehouse from staged measurements"
    )
    ap.add_argument("--stage-root", type=Path, required=True, 
                   help="Staged data root (02_stage/raw_measurements)")
    ap.add_argument("--warehouse-root", type=Path, required=True,
                   help="Warehouse output root (03_curated/warehouse)")
    ap.add_argument("--procedures-yaml", type=Path, required=True,
                   help="Procedures schema YAML")
    ap.add_argument("--local-tz", type=str, default=DEFAULT_LOCAL_TZ,
                   help="Local timezone for partitioning")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                   help="Number of worker processes")
    ap.add_argument("--polars-threads", type=int, default=DEFAULT_POLARS_THREADS,
                   help="Polars threads per worker")
    ap.add_argument("--force", action="store_true",
                   help="Force reprocessing all files")
    ap.add_argument("--manifest", type=Path, default=None,
                   help="Warehouse manifest file (default: warehouse_root/_manifest.json)")
    
    args = ap.parse_args()
    
    # Set up paths
    stage_root: Path = args.stage_root
    warehouse_root: Path = args.warehouse_root
    procedures_yaml: Path = args.procedures_yaml
    manifest_path: Path = args.manifest or (warehouse_root / "_manifest.json")
    
    # Validate inputs
    if not stage_root.exists():
        raise SystemExit(f"[error] Stage root does not exist: {stage_root}")
    if not procedures_yaml.exists():
        raise SystemExit(f"[error] Procedures YAML does not exist: {procedures_yaml}")
    
    # Set polars threads
    os.environ["POLARS_MAX_THREADS"] = str(args.polars_threads)
    
    # Load schema and compute hash
    schema_hash = compute_schema_hash(procedures_yaml)
    
    # Load existing manifest
    manifest = load_warehouse_manifest(manifest_path)
    
    # Discover stage files
    stage_files = discover_stage_files(stage_root)
    info(f"Discovered {len(stage_files)} staged files")
    
    if not stage_files:
        info("No stage files found, nothing to do")
        return
    
    # Filter files that need processing
    files_to_process = [
        sf for sf in stage_files 
        if needs_processing(sf, manifest, schema_hash, args.force)
    ]
    
    info(f"Processing {len(files_to_process)} files (skipping {len(stage_files) - len(files_to_process)})")
    
    if not files_to_process:
        info("All files up to date")
        return
    
    # Ensure warehouse directories exist
    ensure_dir(warehouse_root)
    ensure_dir(warehouse_root / "runs_metadata")
    ensure_dir(warehouse_root / "ts_fact") 
    ensure_dir(warehouse_root / "sweep_fact")
    
    # Process files
    stats = {"ok": 0, "error": 0, "tables": {"runs_metadata": 0, "ts_fact": 0, "sweep_fact": 0}}
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Submit tasks
        futures = []
        for stage_file in files_to_process:
            stage_file_data = {
                "path": str(stage_file.path),
                "proc": stage_file.proc,
                "date": stage_file.date,
                "run_id": stage_file.run_id,
                "size": stage_file.size,
                "mtime": stage_file.mtime
            }
            
            future = executor.submit(
                process_stage_file_task,
                stage_file_data,
                str(warehouse_root),
                str(procedures_yaml),
                args.local_tz,
                args.force
            )
            futures.append((stage_file, future))
        
        # Collect results
        processed_files = manifest.processed_files if manifest else {}
        
        for i, (stage_file, future) in enumerate(futures, 1):
            try:
                result = future.result()
                
                if result["status"] == "ok":
                    stats["ok"] += 1
                    tables_written = result.get("tables_written", [])
                    for table in tables_written:
                        stats["tables"][table] += 1
                    
                    # Update manifest
                    processed_files[stage_file.run_id] = {
                        "proc": stage_file.proc,
                        "date": stage_file.date,
                        "size": stage_file.size,
                        "mtime": stage_file.mtime,
                        "processed_at": dt.datetime.now(tz=dt.timezone.utc).isoformat(),
                        "month_part": result.get("month_part"),
                        "classification": result.get("classification"),
                        "n_rows": result.get("n_rows"),
                        "tables_written": tables_written
                    }
                    
                    tables_str = ",".join(tables_written) if tables_written else "none"
                    info(f"[{i:04d}] OK      {result['proc']:<8} {result['run_id']:<16} "
                         f"rows={result.get('n_rows', 0):<6} → {tables_str}")
                
                else:
                    stats["error"] += 1
                    info(f"[{i:04d}] ERROR   {result['proc']:<8} {result['run_id']:<16} → {result.get('error', 'unknown')}")
                
            except Exception as e:
                stats["error"] += 1
                info(f"[{i:04d}] ERROR   {stage_file.proc:<8} {stage_file.run_id:<16} → {e}")
    
    # Save updated manifest
    updated_manifest = ProcessingManifest(
        processed_files=processed_files,
        schema_hash=schema_hash,
        last_updated=dt.datetime.now(tz=dt.timezone.utc)
    )
    save_warehouse_manifest(manifest_path, updated_manifest)
    
    # Print summary
    info(f"Warehouse build complete:")
    info(f"  ✓ Processed: {stats['ok']}")
    info(f"  ✗ Errors: {stats['error']}")
    info(f"  📊 Tables written:")
    for table, count in stats["tables"].items():
        info(f"    {table}: {count} files")
    
    info(f"Warehouse structure:")
    info(f"  {warehouse_root}/")
    info(f"  ├── runs_metadata/date=YYYY-MM/runs-<RID>.parquet")
    info(f"  ├── ts_fact/date=YYYY-MM/ts-<RID>.parquet")
    info(f"  └── sweep_fact/date=YYYY-MM/sw-<RID>.parquet")


if __name__ == "__main__":
    main()