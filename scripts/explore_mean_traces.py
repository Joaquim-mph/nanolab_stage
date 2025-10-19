#!/usr/bin/env python3
"""
Interactive script to load and explore mean IV traces.

This script loads all mean forward and backward traces for specified dates
and makes them easily accessible for manipulation, plotting, and analysis.

Usage:
    python explore_mean_traces.py

Then in the interactive session, you have access to:
    - traces: dict with structure {date: {vmax: DataFrame}}
    - Each DataFrame has columns: V, I_forward, I_return, I_forward_std, I_return_std
    - Helper functions: plot_trace(), plot_comparison(), subtract_traces()
"""

import polars as pl
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


# Configuration
DATES = ["2025-09-11", "2025-09-29", "2025-09-30"]
HYSTERESIS_ROOT = Path("data/04_analysis/hysteresis")
VOLTAGE_RANGES = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]  # Common ranges across all dates


def load_mean_traces(date: str, vmax: float) -> pl.DataFrame:
    """Load mean forward and backward traces for a specific date and voltage range."""
    vmax_str = str(vmax).replace(".", "p")
    hyst_file = HYSTERESIS_ROOT / date / f"hysteresis_vmax{vmax_str}V.csv"

    if not hyst_file.exists():
        print(f"Warning: {hyst_file} not found")
        return None

    df = pl.read_csv(hyst_file)

    # Extract relevant columns and rename for clarity
    trace_df = df.select([
        pl.col("V (V)").alias("V"),
        pl.col("I_forward"),
        pl.col("I_return"),
        pl.col("I_forward_std"),
        pl.col("I_return_std")
    ])

    return trace_df


def load_all_traces():
    """Load all mean traces for all dates and voltage ranges."""
    traces = {}

    for date in DATES:
        print(f"Loading traces for {date}...")
        traces[date] = {}

        for vmax in VOLTAGE_RANGES:
            trace_df = load_mean_traces(date, vmax)
            if trace_df is not None:
                traces[date][vmax] = trace_df
                print(f"  ✓ {vmax}V: {len(trace_df)} voltage points")

    return traces


# Helper functions for common operations
def plot_trace(trace_df: pl.DataFrame, title: str = "IV Trace", show_std: bool = True):
    """Plot a single trace with forward and backward sweeps."""
    fig, ax = plt.subplots(figsize=(10, 6))

    v = trace_df["V"].to_numpy()
    i_fwd = trace_df["I_forward"].to_numpy()
    i_ret = trace_df["I_return"].to_numpy()

    ax.plot(v, i_fwd, 'o-', label='Forward', alpha=0.7, markersize=3)
    ax.plot(v, i_ret, 's-', label='Backward', alpha=0.7, markersize=3)

    if show_std and "I_forward_std" in trace_df.columns:
        i_fwd_std = trace_df["I_forward_std"].to_numpy()
        i_ret_std = trace_df["I_return_std"].to_numpy()
        ax.fill_between(v, i_fwd - i_fwd_std, i_fwd + i_fwd_std, alpha=0.2)
        ax.fill_between(v, i_ret - i_ret_std, i_ret + i_ret_std, alpha=0.2)

    ax.set_xlabel("Voltage (V)", fontsize=12)
    ax.set_ylabel("Current (A)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, ax


def plot_comparison(traces_dict: dict, direction: str = "forward", title: str = "Comparison"):
    """
    Plot multiple traces for comparison.

    Args:
        traces_dict: Dict like {label: DataFrame, ...}
        direction: "forward" or "backward"
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    col_name = "I_forward" if direction == "forward" else "I_return"

    for label, trace_df in traces_dict.items():
        v = trace_df["V"].to_numpy()
        i = trace_df[col_name].to_numpy()
        ax.plot(v, i, 'o-', label=label, alpha=0.7, markersize=3)

    ax.set_xlabel("Voltage (V)", fontsize=12)
    ax.set_ylabel("Current (A)", fontsize=12)
    ax.set_title(f"{title} ({direction.capitalize()})", fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, ax


def subtract_traces(trace1: pl.DataFrame, trace2: pl.DataFrame, direction: str = "forward") -> pl.DataFrame:
    """
    Subtract trace2 from trace1 (trace1 - trace2).

    Returns DataFrame with V and I_diff columns.
    """
    col_name = "I_forward" if direction == "forward" else "I_return"

    # Join on voltage and compute difference
    result = trace1.join(trace2, on="V", how="inner", suffix="_trace2")

    diff_df = result.select([
        pl.col("V"),
        (pl.col(col_name) - pl.col(f"{col_name}_trace2")).alias("I_diff")
    ])

    return diff_df


def get_trace(traces: dict, date: str, vmax: float) -> pl.DataFrame:
    """Convenience function to get a specific trace."""
    return traces.get(date, {}).get(vmax, None)


# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Loading Mean IV Traces for Interactive Exploration")
    print("="*70)
    print()

    # Load all traces
    traces = load_all_traces()

    print()
    print("="*70)
    print("Data loaded successfully!")
    print("="*70)
    print()
    print("Available data structure:")
    print("  traces[date][vmax] -> DataFrame with columns:")
    print("    - V: Voltage")
    print("    - I_forward: Mean forward current")
    print("    - I_return: Mean backward current")
    print("    - I_forward_std: Forward standard deviation")
    print("    - I_return_std: Backward standard deviation")
    print()
    print("Available dates:", DATES)
    print("Available voltage ranges:", VOLTAGE_RANGES)
    print()
    print("Helper functions:")
    print("  - get_trace(traces, date, vmax)")
    print("  - plot_trace(df, title='...', show_std=True)")
    print("  - plot_comparison({label: df, ...}, direction='forward')")
    print("  - subtract_traces(df1, df2, direction='forward')")
    print()
    print("="*70)
    print()
    print("Examples:")
    print("-" * 70)
    print()
    print("# Get a specific trace")
    print("df = get_trace(traces, '2025-09-11', 8.0)")
    print()
    print("# Plot it")
    print("fig, ax = plot_trace(df, title='Sept 11 - 8V')")
    print("plt.show()")
    print()
    print("# Compare all voltage ranges for one date")
    print("comparison = {f'{v}V': traces['2025-09-11'][v] for v in [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]}")
    print("fig, ax = plot_comparison(comparison, direction='forward', title='Sept 11 - All Ranges')")
    print("plt.show()")
    print()
    print("# Compare same voltage range across dates")
    print("comparison = {date: traces[date][8.0] for date in DATES}")
    print("fig, ax = plot_comparison(comparison, direction='forward', title='8V - All Dates')")
    print("plt.show()")
    print()
    print("# Subtract traces")
    print("diff = subtract_traces(traces['2025-09-11'][8.0], traces['2025-09-11'][3.0], direction='forward')")
    print("plt.plot(diff['V'], diff['I_diff'])")
    print("plt.show()")
    print()
    print("# Access raw data")
    print("df = traces['2025-09-11'][8.0]")
    print("voltages = df['V'].to_numpy()")
    print("currents_fwd = df['I_forward'].to_numpy()")
    print()
    print("="*70)
    print()
    print("Starting interactive Python session...")
    print("All traces loaded in 'traces' variable")
    print("="*70)
    print()

    # Start interactive session
    import code
    code.interact(local=locals())
