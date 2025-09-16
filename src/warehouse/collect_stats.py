#!/usr/bin/env python3
"""
collect_stats.py
-----------------
Collect general experiment statistics from the curated warehouse and write easy-to-consume
Parquet/CSV artifacts (and optionally print a readable summary).

Expected repo layout (defaults; can be overridden via CLI):
  <repo>/
    data/03_curated/warehouse/
      runs_metadata/date=YYYY-MM/runs-*.parquet
    docs/                                   # CSV snapshots are written here by default

Usage (from repo root):
  python src/warehouse/collect_stats.py
  # or, if placed elsewhere:
  python collect_stats.py --repo-root . --local-tz America/Santiago

Examples:
  # Restrict to local dates on/after 2024-01-01 and before 2025-01-01:
  python collect_stats.py --start-from 2024-01-01 --end-excl 2025-01-01 --print

  # Use a custom warehouse path and output directory:
  python collect_stats.py \
    --warehouse data/03_curated/warehouse \
    --out-stats data/03_curated/warehouse/stats

Notes:
  - Uses Polars only (no pandas).
  - Timezone handling is DST-safe. We interpret --start-from/--end-excl as LOCAL DATES
    in --local-tz, then convert to UTC boundaries for filtering.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Optional
import glob as _glob
import polars as pl

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception as e:  # pragma: no cover
    ZoneInfo = None

# ---------------------------
# CLI
# ---------------------------

@dataclass
class Config:
    repo_root: Path
    warehouse: Path
    out_stats: Path
    docs_dir: Path
    local_tz: str
    start_from: Optional[date]
    end_excl: Optional[date]
    do_print: bool

def parse_args(argv=None) -> Config:
    parser = argparse.ArgumentParser(
        description="Collect experiment statistics (Polars) and write Parquet/CSV artifacts."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd().resolve(),
        help="Path to repo root (default: current working directory)",
    )
    parser.add_argument(
        "--warehouse",
        type=Path,
        default=Path("data/03_curated/warehouse"),
        help="Path to curated warehouse (relative to --repo-root if not absolute)",
    )
    parser.add_argument(
        "--out-stats",
        type=Path,
        default=Path("data/03_curated/warehouse/stats"),
        help="Output directory for Parquet stats (relative to --repo-root if not absolute)",
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=Path("docs"),
        help="Directory for human-friendly CSVs (relative to --repo-root if not absolute)",
    )
    parser.add_argument(
        "--local-tz",
        default="America/Santiago",
        help="IANA timezone for local-day aggregations and date filtering",
    )
    parser.add_argument(
        "--start-from",
        type=str,
        default=None,
        help="(Optional) Local DATE (YYYY-MM-DD) inclusive lower bound"
    )
    parser.add_argument(
        "--end-excl",
        type=str,
        default=None,
        help="(Optional) Local DATE (YYYY-MM-DD) exclusive upper bound"
    )
    parser.add_argument(
        "--print",
        dest="do_print",
        action="store_true",
        help="Print a short console summary (totals, summary, head of monthly wide)",
    )

    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    warehouse = (args.warehouse if args.warehouse.is_absolute() else repo_root / args.warehouse).resolve()
    out_stats = (args.out_stats if args.out_stats.is_absolute() else repo_root / args.out_stats).resolve()
    docs_dir = (args.docs_dir if args.docs_dir.is_absolute() else repo_root / args.docs_dir).resolve()

    sf = date.fromisoformat(args.start_from) if args.start_from else None
    ee = date.fromisoformat(args.end_excl) if args.end_excl else None

    return Config(
        repo_root=repo_root,
        warehouse=warehouse,
        out_stats=out_stats,
        docs_dir=docs_dir,
        local_tz=args.local_tz,
        start_from=sf,
        end_excl=ee,
        do_print=args.do_print,
    )

# ---------------------------
# Helpers
# ---------------------------

def to_local(expr: pl.Expr, tz: str) -> pl.Expr:
    """Ensure timestamps are UTC-aware then convert to local tz safely across DST."""
    e = expr.cast(pl.Datetime(time_zone="UTC"))
    try:
        return e.dt.convert_time_zone(
            tz,
            ambiguous="earliest",         # fall-back hour occurs twice -> pick first
            non_existent="shift_forward", # spring-forward gap -> shift to next valid
        )
    except TypeError:
        # Older Polars without the keyword args above
        return e.dt.convert_time_zone(tz)

def local_date_bounds_to_utc(tzname: str, start: Optional[date], end_excl: Optional[date]):
    """Translate local DATE bounds into UTC-aware datetime bounds suitable for filtering.

    start (inclusive) and end_excl (exclusive) are *local* dates in tzname.
    Returns (start_utc, end_utc) as aware datetimes in UTC, allowing either to be None.
    """
    if ZoneInfo is None:
        raise RuntimeError("Python's zoneinfo is unavailable; please use Python 3.9+")

    tzinfo = ZoneInfo(tzname)

    def d_to_utc(d: date, end: bool = False) -> datetime:
        # midnight local for start; midnight local for end_excl
        dt_local = datetime.combine(d, time(0, 0, 0), tzinfo=tzinfo)
        return dt_local.astimezone(timezone.utc)

    start_utc = d_to_utc(start) if start else None
    end_utc = d_to_utc(end_excl) if end_excl else None
    return start_utc, end_utc

def ensure_non_empty(paths: list[str], where: Path) -> None:
    if not paths:
        raise FileNotFoundError(
            f"No files found under {where/'runs_metadata'}. Run your curated builder first."
        )

# ---------------------------
# Core logic
# ---------------------------

def main(argv=None) -> int:
    cfg = parse_args(argv)

    path_glob = str(cfg.warehouse / "runs_metadata" / "date=*" / "runs-*.parquet")
    paths = _glob.glob(path_glob)
    ensure_non_empty(paths, cfg.warehouse)
    lf_md = pl.scan_parquet(paths)  # pl.scan_parquet accepts a list of paths


    # Optional date filtering using UTC bounds derived from local dates
    if cfg.start_from or cfg.end_excl:
        start_utc, end_utc = local_date_bounds_to_utc(cfg.local_tz, cfg.start_from, cfg.end_excl)
        expr = pl.col("start_dt").cast(pl.Datetime(time_zone="UTC"))
        conds = []
        if start_utc is not None:
            conds.append(expr >= pl.lit(start_utc, dtype=pl.Datetime(time_zone="UTC")))
        if end_utc is not None:
            conds.append(expr < pl.lit(end_utc, dtype=pl.Datetime(time_zone="UTC")))
        if conds:
            from functools import reduce
            import operator
            lf_md = lf_md.filter(reduce(operator.and_, conds))

    # Create local-day key
    lf_md_local = lf_md.with_columns(
        to_local(pl.col("start_dt"), cfg.local_tz).alias("start_local")
    ).with_columns(
        pl.col("start_local").dt.date().alias("start_day_local")
    )

    # 1) Counts per local day
    per_day = (
        lf_md_local
        .group_by("start_day_local")
        .agg(pl.len().alias("n"))
        .sort("start_day_local")
        .collect()
    )

    # 2) All-time totals per experiment kind (proc)
    totals = (
        lf_md_local.group_by("proc")
        .agg(pl.len().alias("n_runs"))
        .sort("n_runs", descending=True)
        .collect()
    )

    # 3) One-row summary
    summary = (
        lf_md_local.select(
            pl.len().alias("total_runs"),
            pl.col("proc").n_unique().alias("distinct_kinds"),
            pl.col("start_dt").min().alias("first_run_utc"),
            pl.col("start_dt").max().alias("last_run_utc"),
        )
        .collect()
    )

    # 4) Monthly counts per proc (UTC month buckets for reproducibility)
    monthly = (
        lf_md_local
        .with_columns(
            pl.col("start_dt")
            .cast(pl.Datetime(time_zone="UTC"))
            .dt.truncate("1mo")
            .alias("month")
        )
        .group_by(["month", "proc"])
        .agg(pl.len().alias("n_runs"))
        .sort(["month", "proc"])
        .collect()
    )

    # 5) Monthly wide (string month YYYY-MM)
    monthly_wide = (
        monthly
        .with_columns(pl.col("month").dt.strftime("%Y-%m").alias("month"))
        .pivot(
            values="n_runs",
            index="month",
            columns="proc",
            aggregate_function="sum",
        )
        .fill_null(0)
        .sort("month")
    )

    # Ensure output dirs
    cfg.out_stats.mkdir(parents=True, exist_ok=True)
    cfg.docs_dir.mkdir(parents=True, exist_ok=True)

    # Save artifacts
    per_day.write_parquet(cfg.out_stats / "experiments_per_day_local.parquet")
    totals.write_parquet(cfg.out_stats / "experiments_by_proc.parquet")
    summary.write_parquet(cfg.out_stats / "experiments_summary.parquet")
    monthly.write_parquet(cfg.out_stats / "experiments_by_proc_monthly.parquet")
    monthly_wide.write_parquet(cfg.out_stats / "experiments_by_proc_monthly_wide.parquet")

    # Human CSVs
    per_day.write_csv(cfg.docs_dir / "experiments_per_day_local.csv")
    totals.write_csv(cfg.docs_dir / "experiments_by_proc_totals.csv")
    summary.write_csv(cfg.docs_dir / "experiments_summary.csv")
    monthly.write_csv(cfg.docs_dir / "experiments_by_proc_monthly.csv")
    monthly_wide.write_csv(cfg.docs_dir / "experiments_by_proc_monthly_wide.csv")

    if cfg.do_print:
        print("\n=== Summary ===")
        print(summary)
        print("\n=== Totals by proc ===")
        print(totals)
        print("\n=== Per day (tail) ===")
        print(per_day.tail(10))
        print("\n=== Monthly wide (head) ===")
        print(monthly_wide.head(12))

    return 0

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
