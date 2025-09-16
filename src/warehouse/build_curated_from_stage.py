#!/usr/bin/env python3
"""
Parallel builder for curated tables from the staged layer.

Input (staged files):
  02_stage/raw_measurements/proc=<PROC>/date=<YYYY-MM-DD>/run_id=<RID>/part-000.parquet

Output (curated tables):
  03_curated/warehouse/
    runs_metadata/date=<YYYY-MM>/runs-<RID>.parquet   (1 row per run)
    ts_fact/date=<YYYY-MM>/ts-<RID>.parquet           (runs with t_s)
    sweep_fact/date=<YYYY-MM>/sw-<RID>.parquet        (runs without t_s)

Parallelization model:
- One staged Parquet (run) == one task.
- ProcessPoolExecutor with configurable worker count.
- Optional cap on Polars threads per worker to avoid oversubscription.
- Atomic writes (temp file then rename) to avoid partial outputs.
- Idempotent: skip writing if output exists unless --force.

Usage:
  python build_curated_from_stage_parallel.py \
    --stage-root 02_stage/raw_measurements \
    --warehouse-root 03_curated/warehouse \
    --workers 6 \
    --polars-threads 1 \
    [--raw-root 01_raw] \
    [--force]
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import polars as pl

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None  # type: ignore


# ----------------------------- Helpers --------------------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_stage_file(p: Path) -> pl.DataFrame:
    return pl.read_parquet(p)


def md5_file(p: Path, chunk: int = 1 << 20) -> Optional[str]:
    if not p or not p.exists():
        return None
    h = hashlib.md5()
    with p.open("rb") as f:
        while (b := f.read(chunk)):
            h.update(b)
    return h.hexdigest()


def month_partition_from_start_dt(df: pl.DataFrame, tz: Optional[str]) -> str:
    if "start_dt" not in df.columns:
        return "unknown"
    start = df["start_dt"][0]
    # Ensure it's a timezone-aware datetime
    if isinstance(start, dt.datetime):
        dtu = start
    else:
        try:
            dtu = pl.Series([start]).cast(pl.Datetime).to_list()[0]  # type: ignore
        except Exception:
            return "unknown"
    if tz and ZoneInfo is not None and dtu.tzinfo is not None:
        local = dtu.astimezone(ZoneInfo(tz))
        return f"{local.year:04d}-{local.month:02d}"
    else:
        return f"{dtu.year:04d}-{dtu.month:02d}"


def collect_stage_files(stage_root: Path) -> list[Path]:
    return sorted(stage_root.glob("proc=*/date=*/run_id=*/part-000.parquet"))


def atomic_write_parquet(df: pl.DataFrame, out_file: Path) -> None:
    """Write Parquet to temp file in same dir, then rename (atomic on same FS)."""
    ensure_dir(out_file.parent)
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=out_file.parent) as tmp:
        tmp_path = Path(tmp.name)
    try:
        df.write_parquet(tmp_path)
        tmp_path.replace(out_file)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        raise


# ----------------------------- Worker task ----------------------------------

def build_one_run(
    stage_path_str: str,
    warehouse_root_str: str,
    raw_root_str: Optional[str],
    tz: str,
    force: bool,
    compute_md5: bool,
) -> Dict[str, Any]:
    stage_path = Path(stage_path_str)
    wh_root = Path(warehouse_root_str)
    raw_root = Path(raw_root_str) if raw_root_str else None

    df = read_stage_file(stage_path)
    if "run_id" not in df.columns:
        raise RuntimeError("staged file missing 'run_id'")
    rid = df["run_id"][0]
    month = month_partition_from_start_dt(df, tz)

    runs_out = Path(wh_root) / "runs_metadata" / f"date={month}"
    ts_out   = Path(wh_root) / "ts_fact"        / f"date={month}"
    sw_out   = Path(wh_root) / "sweep_fact"     / f"date={month}"

    # --- runs_metadata (1 row per run) ---
    n_rows = df.height
    # Optional md5 of raw source file
    src_md5 = None
    if compute_md5 and "source_file" in df.columns:
        src = df["source_file"][0]
        sp = Path(str(src))
        if not sp.exists() and raw_root is not None:
            sp2 = raw_root / sp
            if sp2.exists():
                sp = sp2
        src_md5 = md5_file(sp) if sp.exists() else None

    md_row = {
        "run_id": rid,
        "proc": df.get_column("proc")[0] if "proc" in df.columns else None,
        "source_file": df.get_column("source_file")[0] if "source_file" in df.columns else None,
        "start_dt": df.get_column("start_dt")[0] if "start_dt" in df.columns else None,
        "chip_group": df.get_column("chip_group")[0] if "chip_group" in df.columns else None,
        "chip_number": df.get_column("chip_number")[0] if "chip_number" in df.columns else None,
        "sample": df.get_column("sample")[0] if "sample" in df.columns else None,
        "with_light": df.get_column("with_light")[0] if "with_light" in df.columns else None,
        "wavelength_nm": df.get_column("wavelength_nm")[0] if "wavelength_nm" in df.columns else None,
        "laser_voltage_V": df.get_column("laser_voltage_V")[0] if "laser_voltage_V" in df.columns else None,
        "info": df.get_column("info")[0] if "info" in df.columns else None,
        "procedure_version": df.get_column("procedure_version")[0] if "procedure_version" in df.columns else None,
        "n_rows": n_rows,
        "md5": src_md5,
    }
    md_file = runs_out / f"runs-{rid}.parquet"
    if not md_file.exists() or force:
        atomic_write_parquet(pl.DataFrame([md_row]), md_file)

    # --- fact tables ---
    if "t_s" in df.columns:  # time-series
        keep = [c for c in ["run_id","t_s","I_A","VG_V","VD_V","VL_V","plate_C","ambient_C","clock_ms"] if c in df.columns]
        if "run_id" not in keep:
            df = df.with_columns(pl.lit(rid).alias("run_id"))
            keep.insert(0, "run_id")
        slim = df.select(keep)
        out_file = ts_out / f"ts-{rid}.parquet"
        if not out_file.exists() or force:
            atomic_write_parquet(slim, out_file)
        fact = "ts"
    else:  # sweep-like
        keep_candidates = ["run_id", "VG_V", "VD_V", "I_A", "step_idx"]
        keep = [c for c in keep_candidates if c in df.columns]
        if "run_id" not in keep:
            df = df.with_columns(pl.lit(rid).alias("run_id"))
            keep.insert(0, "run_id")
        slim = df.select(keep)
        out_file = sw_out / f"sw-{rid}.parquet"
        if not out_file.exists() or force:
            atomic_write_parquet(slim, out_file)
        fact = "sw"

    return {"status": "ok", "run_id": rid, "month": month, "fact": fact, "paths": [str(md_file), str(out_file)]}


# ----------------------------- Main routine ---------------------------------

def main():
    ap = argparse.ArgumentParser(description="Parallel build curated tables from staged Parquet runs.")
    ap.add_argument("--stage-root", type=Path, required=True, help="02_stage/raw_measurements path")
    ap.add_argument("--warehouse-root", type=Path, required=True, help="03_curated/warehouse path")
    ap.add_argument("--raw-root", type=Path, default=None, help="(Optional) 01_raw path to compute MD5 of source_file")
    ap.add_argument("--local-tz", type=str, default="America/Santiago", help="For month partitioning")
    ap.add_argument("--workers", type=int, default=6, help="Process workers (default: 6)")
    ap.add_argument("--polars-threads", type=int, default=1, help="POLARS_MAX_THREADS per worker (default: 1)")
    ap.add_argument("--force", action="store_true", help="Overwrite existing curated files")
    ap.add_argument("--compute-md5", action="store_true", help="Compute md5 of raw CSVs for runs_metadata")
    args = ap.parse_args()

    stage_root: Path = args.stage_root
    wh_root: Path = args.warehouse_root
    raw_root: Optional[Path] = args.raw_root
    tz: str = args.local_tz
    workers: int = max(1, args.workers)
    polars_threads: int = max(1, args.polars_threads)
    force: bool = args.force
    compute_md5: bool = args.compute_md5

    if not stage_root.exists():
        raise SystemExit(f"[error] stage root does not exist: {stage_root}")
    ensure_dir(wh_root)

    # Cap Polars threads per process to avoid oversubscription
    os.environ["POLARS_MAX_THREADS"] = str(polars_threads)

    stage_files = collect_stage_files(stage_root)
    print(f"[info] discovered {len(stage_files)} staged runs under {stage_root}")
    if not stage_files:
        print("[done] nothing to do.")
        return

    ok = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [
            ex.submit(
                build_one_run,
                str(p),
                str(wh_root),
                str(raw_root) if raw_root else None,
                tz,
                force,
                compute_md5,
            )
            for p in stage_files
        ]

        for i, fut in enumerate(as_completed(futs), 1):
            try:
                out = fut.result()
                ok += 1
                print(f"[{i:04d}] OK run_id={out['run_id']} fact={out['fact']} month={out['month']}")
            except Exception as e:
                print(f"[{i:04d}] REJECT task :: {e}")

    print(f"[done] curated parallel build complete | ok={ok} / {len(stage_files)}")


if __name__ == "__main__":
    main()
