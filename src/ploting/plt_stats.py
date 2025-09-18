#!/usr/bin/env python3
"""
collect_stats_with_requested_plots.py
-------------------------------------
Collect experiment statistics (Polars-only) and generate the exact plots requested:

1) hist_runs(lf_md, bins=100)
2) hist_compare_procs(lf_md, procs=("IVg","It","IV"), bins=120)
3) hist_compare_procs(lf_md, procs, start="2025-06-01", end="2025-09-01", bins=100, histtype="bar", stacked=True)
4) plot_experiments_by_proc(...) with cleaned totals (MIN_RUNS=1000, exclude FakeProcedure, rename LaserCalibration->PVl)
5) Same bar plot but MIN_RUNS=300
6) plot_monthly(monthly_wide, ...) stacked
7) plot_monthly(monthly_wide, ...) stacked pct
8) plot_monthly(monthly_wide, ...) lines

Outputs go to --plots-dir (default: docs/figures).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Optional, List

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import scienceplots
from styles import set_plot_style
set_plot_style('prism_rain')


try:
    from zoneinfo import ZoneInfo
except Exception:
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
    plots_dir: Path
    local_tz: str
    start_from: Optional[date]
    end_excl: Optional[date]
    do_print: bool
    make_plots: bool

def parse_date_opt(s: Optional[str]) -> Optional[date]:
    return date.fromisoformat(s) if s else None

def parse_args(argv=None) -> Config:
    p = argparse.ArgumentParser(description="Collect stats and generate requested plots (Polars-only).")
    p.add_argument("--repo-root", type=Path, default=Path.cwd().resolve(), help="Repo root (default: CWD)")
    p.add_argument("--warehouse", type=Path, default=Path("data/03_curated"),
                   help="Warehouse path (relative OK)")
    p.add_argument("--out-stats", type=Path, default=Path("data/03_curated/warehouse/stats"),
                   help="Directory for Parquet stats (relative OK)")
    p.add_argument("--docs-dir", type=Path, default=Path("docs"),
                   help="Directory for CSV outputs (relative OK)")
    p.add_argument("--plots-dir", type=Path, default=Path("figures/warehouse_stats"),
                help="Directory for figures")
    p.add_argument("--local-tz", default="America/Santiago", help="IANA timezone for local-day ops")
    p.add_argument("--start-from", type=str, default=None, help="Local DATE (YYYY-MM-DD) inclusive lower bound")
    p.add_argument("--end-excl", type=str, default=None, help="Local DATE (YYYY-MM-DD) exclusive upper bound")
    p.add_argument("--print", dest="do_print", action="store_true", help="Print summary tables")
    p.add_argument("--plots", dest="make_plots", action="store_true", help="Produce figures")
    args = p.parse_args(argv)

    repo_root = args.repo_root.resolve()
    warehouse = (args.warehouse if args.warehouse.is_absolute() else repo_root / args.warehouse).resolve()
    out_stats = (args.out_stats if args.out_stats.is_absolute() else repo_root / args.out_stats).resolve()
    docs_dir = (args.docs_dir if args.docs_dir.is_absolute() else repo_root / args.docs_dir).resolve()
    plots_dir = (args.plots_dir if args.plots_dir.is_absolute() else repo_root / args.plots_dir).resolve()

    return Config(
        repo_root=repo_root,
        warehouse=warehouse,
        out_stats=out_stats,
        docs_dir=docs_dir,
        plots_dir=plots_dir,
        local_tz=args.local_tz,
        start_from=parse_date_opt(args.start_from),
        end_excl=parse_date_opt(args.end_excl),
        do_print=args.do_print,
        make_plots=args.make_plots,
    )

# ---------------------------
# Time helpers and DST-safe local conversions
# ---------------------------

def to_local(expr: pl.Expr, tz: str) -> pl.Expr:
    e = expr.cast(pl.Datetime(time_zone="UTC"))
    try:
        return e.dt.convert_time_zone(tz, ambiguous="earliest", non_existent="shift_forward")
    except TypeError:
        return e.dt.convert_time_zone(tz)

def local_date_bounds_to_utc(tzname: str, start: Optional[date], end_excl: Optional[date]):
    if ZoneInfo is None:
        raise RuntimeError("zoneinfo unavailable; use Python 3.9+")
    tzinfo = ZoneInfo(tzname)
    def d_to_utc(d: date) -> datetime:
        dt_local = datetime.combine(d, time(0, 0, 0), tzinfo=tzinfo)
        return dt_local.astimezone(timezone.utc)
    return (d_to_utc(start) if start else None, d_to_utc(end_excl) if end_excl else None)

# ---------------------------
# Plotting helpers (adapted from your functions)
# ---------------------------

def _to_date(x):
    if x is None:
        return None
    if isinstance(x, date) and not isinstance(x, datetime):
        return x
    if isinstance(x, datetime):
        return x.date()
    return date.fromisoformat(str(x))

def _rc_cycle_colors(labels, by: str = "order"):
    base = plt.rcParams.get("axes.prop_cycle", None)
    palette = base.by_key().get("color", list(plt.cm.tab10.colors)) if base else list(plt.cm.tab10.colors)
    if by == "name":
        uniq = sorted(set(labels))
        mapping = {name: palette[i % len(palette)] for i, name in enumerate(uniq)}
        return [mapping[name] for name in labels]
    return [palette[i % len(palette)] for i, _ in enumerate(labels)]

def hist_runs(lf_md, *, tz, bins=100, proc=None, start=None, end=None,
              figsize=(15, 6), histtype="step", out_path: Optional[Path] = None):
    import matplotlib.dates as mdates
    lf = (
        lf_md.with_columns(to_local(pl.col("start_dt"), tz).alias("start_local"))
             .with_columns(pl.col("start_local").dt.date().alias("start_day_local"))
             .select("start_local", "start_day_local", "proc")
    )
    if proc is not None:
        lf = lf.filter(pl.col("proc") == proc)
    start_d = _to_date(start); end_d = _to_date(end)
    if start_d is not None:
        lf = lf.filter(pl.col("start_day_local") >= pl.lit(start_d))
    if end_d is not None:
        lf = lf.filter(pl.col("start_day_local") < pl.lit(end_d))

    df = lf.select("start_local").collect()
    if df.height == 0:
        print("No runs match the filter.")
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax

    dates = df["start_local"].to_list()
    dates_num = mdates.date2num(dates)

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(dates_num, bins=bins, histtype=histtype)
    title = "Histogram of experiments" + ("" if proc is None else f" ({proc})")
    ax.set_title(title)
    ax.set_ylabel("Number of experiments")
    ax.set_xlabel("Start time (local)")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.set_xlim(min(dates_num), max(dates_num))
    ax.xaxis_date()
    fig.autofmt_xdate()
    plt.tight_layout()
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Saved histogram to: {out_path}")
    plt.close(fig)
    return fig, ax

def hist_compare_procs(lf_md, tz, procs, *, bins=100, start=None, end=None,
                       histtype="step", stacked=False, density=False, figsize=(15, 6),
                       out_path: Optional[Path] = None):
    import matplotlib.dates as mdates
    lf = (
        lf_md.with_columns(to_local(pl.col("start_dt"), tz).alias("start_local"))
             .with_columns(pl.col("start_local").dt.date().alias("start_day_local"))
             .select("start_local", "start_day_local", "proc")
    )
    start_d = _to_date(start); end_d = _to_date(end)
    if start_d is not None:
        lf = lf.filter(pl.col("start_day_local") >= pl.lit(start_d))
    if end_d is not None:
        lf = lf.filter(pl.col("start_day_local") < pl.lit(end_d))

    series = []; labels = []
    for p in procs:
        col = (lf.filter(pl.col("proc") == p)
                 .select("start_local").collect()
                 .get_column("start_local").to_list())
        if col:
            series.append(col); labels.append(p)

    if not series:
        print("No runs match the filters for overlay histogram.")
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax

    nums = [mdates.date2num(s) for s in series]
    gmin = min(min(arr) for arr in nums)
    gmax = max(max(arr) for arr in nums)
    if not (gmax > gmin):
        gmin, gmax = gmin - 0.5, gmax + 0.5
    step = (gmax - gmin) / max(1, int(bins))
    edges = [gmin + i * step for i in range(int(bins) + 1)]

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(nums, bins=edges, histtype=histtype, stacked=stacked, density=density, label=labels)
    ax.set_ylabel("Number of experiments" if not density else "Density")
    ttl = "Histogram of experiments (overlay)"
    if start or end:
        ttl += f" ({start or '…'} to {end or '…'})"
    ax.set_title(ttl)
    ax.set_xlabel("Start time (local)")
    ax.legend()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.set_xlim(min(edges), max(edges))
    ax.xaxis_date()
    fig.autofmt_xdate()
    plt.tight_layout()
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Saved overlay histogram to: {out_path}")
    plt.close(fig)
    return fig, ax

def plot_experiments_by_proc(
    totals: pl.DataFrame,
    out_path: Path,
    title: str = "Experiments by procedure (totals)",
    top_n: int | None = None,
    horizontal: bool = True,
    collapse_others: bool = False,
    others_label: str = "other",
    table_layout: bool = True,
    label_col_ratio: float = 1.6,
    exclude: set[str] | None = None,
    rename_map: dict[str, str] | None = None,
    color_by: str = "order",
):
    df = totals.select(["proc", "n_runs"])
    if exclude:
        df = df.filter(~pl.col("proc").is_in(exclude))
    if rename_map:
        df = df.with_columns(
            pl.col("proc").map_elements(lambda x: rename_map.get(x, x), return_dtype=pl.Utf8).alias("proc")
        )
    df = df.group_by("proc").agg(pl.col("n_runs").sum().alias("n_runs")).sort("n_runs", descending=True)

    if top_n is not None and df.height > top_n:
        if collapse_others:
            head = df.head(top_n)
            tail_sum = int(df.slice(top_n)["n_runs"].sum())
            df = pl.DataFrame({
                "proc": head["proc"].to_list() + [others_label],
                "n_runs": head["n_runs"].to_list() + [tail_sum],
            })
        else:
            df = df.head(top_n)

    if df.is_empty():
        raise ValueError("No data to plot.")

    procs  = df.get_column("proc").to_list()
    counts = df.get_column("n_runs").cast(int).to_list()
    total  = max(sum(counts), 1)
    colors = _rc_cycle_colors(procs, by=color_by)

    PROC_LABEL_WEIGHT = 900
    PROC_LABEL_SIZE   = 15
    ANNO_SIZE         = 15
    ANNO_WEIGHT       = 700

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Horizontal, two-column "table" layout
    fig_h = max(4, 0.5 * len(procs) + 5)
    fig_w = 12
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(nrows=1, ncols=2, width_ratios=[label_col_ratio, 5], figure=fig)

    ax = fig.add_subplot(gs[1])
    y = np.arange(len(procs))
    ax.barh(y, counts, color=colors)
    ax.set_xlabel("Runs")
    ax.set_title(title)
    ax.set_yticks(y); ax.set_yticklabels([])
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.invert_yaxis()
    xmax = max(counts); ax.set_xlim(0, xmax * 1.12)

    for yi, v in enumerate(counts):
        pct = 100 * v / total
        ax.annotate(f"{v:,}  ({pct:.1f}%)", xy=(v, yi), xytext=(6, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=ANNO_SIZE, fontweight=ANNO_WEIGHT, clip_on=False)

    ax_lbl = fig.add_subplot(gs[0], sharey=ax)
    ax_lbl.set_xlim(0, 1); ax_lbl.set_xticks([]); ax_lbl.set_yticks([])
    for s in ("top", "right", "bottom", "left"):
        ax_lbl.spines[s].set_visible(False)
    ax_lbl.grid(False); ax_lbl.set_facecolor("none")

    for yi, name in enumerate(procs):
        ax_lbl.text(0.98, yi, name, transform=ax_lbl.get_yaxis_transform(),
                    ha="right", va="center", fontsize=PROC_LABEL_SIZE, fontweight=PROC_LABEL_WEIGHT)

    ax_lbl.text(0.99, 1.02, "Procedure", transform=ax_lbl.transAxes,
                ha="right", va="bottom", fontsize=PROC_LABEL_SIZE + 4, fontweight=PROC_LABEL_WEIGHT)

    fig.subplots_adjust(wspace=0.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to: {out_path}")

def plot_monthly(
    monthly_wide: pl.DataFrame,
    out_path: Path,
    mode: str = "stacked",
    top_k: int = 8,
    include_others: bool = True,
    normalize_pct: bool = False,
    title: str = "Experiments per month by procedure",
):
    assert "month" in monthly_wide.columns, "`monthly_wide` must have a 'month' column"
    df = monthly_wide.sort("month")
    months = df.get_column("month").to_list()
    procs  = [c for c in df.columns if c != "month"]
    if not procs:
        raise ValueError("No procedure columns found.")

    totals_row = df.select([pl.sum(c).alias(c) for c in procs]).row(0)
    totals = list(zip(procs, totals_row))
    totals.sort(key=lambda x: x[1], reverse=True)
    keep = [p for p, _ in totals[:top_k]]
    rest = [p for p, _ in totals[top_k:]]

    M_keep = df.select(keep).to_numpy()
    if include_others and rest:
        M_other = df.select(rest).to_numpy().sum(axis=1, keepdims=True)
        M = np.concatenate([M_keep, M_other], axis=1)
        series = keep + ["other"]
    else:
        M = M_keep
        series = keep

    colors = _rc_cycle_colors(series, by="name")
    x = np.arange(len(months))

    if mode == "stacked":
        plt.figure(figsize=(max(10, len(months) * 0.6), 6))
        ax = plt.gca()
        Y = M.copy()
        if normalize_pct:
            denom = Y.sum(axis=1, keepdims=True)
            denom[denom == 0] = 1.0
            Y = (Y / denom) * 100.0
        bottom = np.zeros(len(months))
        for i, (name, col) in enumerate(zip(series, Y.T)):
            ax.bar(x, col, bottom=bottom, label=name, color=colors[i])
            bottom += col
        ax.set_title(title); ax.set_xlim(-0.5, len(months) - 0.5)
        ax.set_xlabel("Month"); ax.set_ylabel("Share (%)" if normalize_pct else "Runs")
        ax.grid(axis="y", linestyle=":", alpha=0.6)
        step = max(1, len(months) // 12)
        ax.set_xticks(x[::step])
        ax.set_xticklabels([months[i] for i in range(0, len(months), step)], rotation=45, ha="right")
        if normalize_pct:
            ax.set_ylim(0, 100); ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}%"))
        else:
            ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
        ax.legend(title="Procedure", ncols=3, fontsize=9, title_fontsize=10, frameon=False, loc="upper left",
                  bbox_to_anchor=(1.01, 1.0))
        plt.tight_layout(); out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200, bbox_inches="tight"); plt.close()
        print(f"Saved stacked plot to: {out_path}")
        return

    if mode == "lines":
        plt.figure(figsize=(max(10, len(months) * 0.6), 6))
        ax = plt.gca()
        for i, name in enumerate(keep):
            ax.plot(x, df.get_column(name).to_list(), marker="o", label=name, color=colors[i])
        ax.set_title(title); ax.set_xlim(-0.5, len(months) - 0.5)
        ax.set_xlabel("Month"); ax.set_ylabel("Runs")
        ax.grid(True, linestyle=":", alpha=0.6)
        step = max(1, len(months) // 12)
        ax.set_xticks(x[::step])
        ax.set_xticklabels([months[i] for i in range(0, len(months), step)], rotation=45, ha="right")
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
        ax.legend(title="Procedure", ncols=3, fontsize=9, title_fontsize=10, frameon=False, loc="upper left",
                  bbox_to_anchor=(1.01, 1.0))
        plt.tight_layout(); out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200, bbox_inches="tight"); plt.close()
        print(f"Saved lines plot to: {out_path}")
        return

    raise ValueError("mode must be one of: 'stacked', 'lines'")

# ---------------------------
# Core logic
# ---------------------------

def ensure_non_empty(paths: list[str], where: Path) -> None:
    if not paths:
        raise FileNotFoundError(
            f"No files found under {where/'runs_metadata'}. Run your curated builder first."
        )

def build_frames(cfg: Config):
    import glob as _glob
    path_glob = str(cfg.warehouse / "runs_metadata" / "date=*" / "runs-*.parquet")
    paths = _glob.glob(path_glob)
    ensure_non_empty(paths, cfg.warehouse)
    lf_md = pl.scan_parquet(paths)

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

    # Local-day key for daily counts and histograms
    lf_md_local = lf_md.with_columns(to_local(pl.col("start_dt"), cfg.local_tz).alias("start_local"))                            .with_columns(pl.col("start_local").dt.date().alias("start_day_local"))

    # 1) Counts per local day
    per_day = (
        lf_md_local.group_by("start_day_local")
                   .agg(pl.len().alias("n"))
                   .sort("start_day_local")
                   .collect()
    )

    # 2) All-time totals per proc
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
        ).collect()
    )

    # 4) Monthly counts per proc (UTC months)
    monthly = (
        lf_md_local.with_columns(
            pl.col("start_dt").cast(pl.Datetime(time_zone="UTC")).dt.truncate("1mo").alias("month")
        ).group_by(["month", "proc"]).agg(pl.len().alias("n_runs"))
         .sort(["month", "proc"]).collect()
    )

    # 5) Monthly wide
    monthly_wide = (
        monthly.with_columns(pl.col("month").dt.strftime("%Y-%m").alias("month"))
               .pivot(index="month", on="proc", values="n_runs", aggregate_function="sum")
               .fill_null(0)
               .sort("month")
    )

    return lf_md_local, per_day, totals, summary, monthly, monthly_wide

def save_tables(cfg: Config, per_day, totals, summary, monthly, monthly_wide):
    cfg.out_stats.mkdir(parents=True, exist_ok=True)
    cfg.docs_dir.mkdir(parents=True, exist_ok=True)

    per_day.write_parquet(cfg.out_stats / "experiments_per_day_local.parquet")
    totals.write_parquet(cfg.out_stats / "experiments_by_proc.parquet")
    summary.write_parquet(cfg.out_stats / "experiments_summary.parquet")
    monthly.write_parquet(cfg.out_stats / "experiments_by_proc_monthly.parquet")
    monthly_wide.write_parquet(cfg.out_stats / "experiments_by_proc_monthly_wide.parquet")

    per_day.write_csv(cfg.docs_dir / "experiments_per_day_local.csv")
    totals.write_csv(cfg.docs_dir / "experiments_by_proc_totals.csv")
    summary.write_csv(cfg.docs_dir / "experiments_summary.csv")
    monthly.write_csv(cfg.docs_dir / "experiments_by_proc_monthly.csv")
    monthly_wide.write_csv(cfg.docs_dir / "experiments_by_proc_monthly_wide.csv")

# ---------------------------
# Requested plots
# ---------------------------

def produce_requested_plots(cfg: Config, lf_md_local, totals, monthly_wide):
    # 2b) hist_compare_procs(lf_md, procs=("IVg","It"), bins=120) — overlay IVg vs It
    # (will be placed after other plots below if anchors fail)

    cfg.plots_dir.mkdir(parents=True, exist_ok=True)

    # 1) hist_runs(lf_md, bins=100)
    hist_runs(lf_md_local, tz=cfg.local_tz, bins=100,
              out_path=cfg.plots_dir / "hist_runs_bins100.png")

    # 2) hist_compare_procs(lf_md, procs=("IVg","It","IV"), bins=120)
    procs = ("IVg", "It", "IV")
    hist_compare_procs(lf_md_local, tz=cfg.local_tz, procs=procs, bins=120,
                       out_path=cfg.plots_dir / "hist_compare_IVg_It_IV_bins120.png")

            # 2b) hist_compare_procs(lf_md, procs=("IVg","It"), bins=120) — overlay IVg vs It
    hist_compare_procs(
        lf_md_local, tz=cfg.local_tz, procs=("IVg","It"), bins=120,
        out_path=cfg.plots_dir / "hist_compare_IVg_It_bins120.png",
    )

# 3) hist_compare_procs with date range, bar + stacked
    hist_compare_procs(
        lf_md_local, tz=cfg.local_tz, procs=procs,
        start="2025-06-01", end="2025-09-01",
        bins=100, histtype="bar", stacked=True,
        out_path=cfg.plots_dir / "hist_compare_IVg_It_IV_2025-06-01_2025-09-01_bar_stacked.png",
    )


    
    # Common cleaning for totals
    EXCLUDE = {"FakeProcedure"}
    RENAME  = {"LaserCalibration": "PVl"}

    def clean_totals(df: pl.DataFrame, min_runs: int) -> pl.DataFrame:
        return (
            df
            .filter(~pl.col("proc").is_in(EXCLUDE))
            .with_columns(pl.col("proc").map_elements(lambda x: RENAME.get(x, x), return_dtype=pl.Utf8))
            .group_by("proc").agg(pl.col("n_runs").sum().alias("n_runs"))
            .filter(pl.col("n_runs") > min_runs)
            .sort("n_runs", descending=True)
        )

    # 4) Bar with MIN_RUNS=1000
    totals_1000 = clean_totals(totals, min_runs=1000)
    plot_experiments_by_proc(
        totals=totals_1000,
        out_path=cfg.plots_dir / "experiments_by_proc_totals_min1000.png",
        title="Experiments by procedure (n_runs > 1000)",
        top_n=None,
        horizontal=True,
        collapse_others=False,
    )

    # 5) Bar with MIN_RUNS=300
    totals_300 = clean_totals(totals, min_runs=300)
    plot_experiments_by_proc(
        totals=totals_300,
        out_path=cfg.plots_dir / "experiments_by_proc_totals_min300.png",
        title="Experiments by procedure (n_runs > 300)",
        top_n=None,
        horizontal=True,
        collapse_others=False,
    )

        # Totals bar with ALL procedures
    plot_experiments_by_proc(
        totals.filter(~pl.col("proc").is_in(EXCLUDE))
            .with_columns(pl.col("proc").map_elements(lambda x: RENAME.get(x, x), return_dtype=pl.Utf8))
            .group_by("proc").agg(pl.col("n_runs").sum().alias("n_runs"))
            .sort("n_runs", descending=True),
        out_path=cfg.plots_dir / "totals_all.png",
        title="Experiments by procedure (all)",
        top_n=None,              # <- keep ALL
        collapse_others=False,   # <- don't collapse tail
        horizontal=True
    )
    
    # 6) Stacked bars (top 8 + 'other')
    plot_monthly(monthly_wide, cfg.plots_dir / "monthly_stacked.png",
                 mode="stacked", top_k=8, include_others=True)

    # 7) Stacked percentages (share by month)
    plot_monthly(monthly_wide, cfg.plots_dir / "monthly_stacked_pct.png",
                 mode="stacked", top_k=8, include_others=True, normalize_pct=True)

    # 8) Lines for top 6 procs
    plot_monthly(monthly_wide, cfg.plots_dir / "monthly_lines.png",
                 mode="lines", top_k=6)

# ---------------------------
# Main
# ---------------------------

def main(argv=None) -> int:
    cfg = parse_args(argv)
    lf_md_local, per_day, totals, summary, monthly, monthly_wide = build_frames(cfg)
    save_tables(cfg, per_day, totals, summary, monthly, monthly_wide)

    if cfg.do_print:
        print("\n=== Summary ==="); print(summary)
        print("\n=== Totals by proc ==="); print(totals)
        print("\n=== Per day (tail) ==="); print(per_day.tail(10))
        print("\n=== Monthly wide (head) ==="); print(monthly_wide.head(12))

    if cfg.make_plots:
        produce_requested_plots(cfg, lf_md_local, totals, monthly_wide)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
