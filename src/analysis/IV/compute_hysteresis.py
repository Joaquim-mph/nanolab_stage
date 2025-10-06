#!/usr/bin/env python3
"""
Compute hysteresis current as the difference between forward and backward (return) traces.

Hysteresis current: I_hysteresis = I_forward - I_backward

Computes for:
- Raw data (mean values)
- Polynomial fits (orders 1, 3, 5, 7)

Input: IV statistics directory from aggregate_iv_stats.py
Output: Hysteresis data files and summary
"""

import polars as pl
import numpy as np
from pathlib import Path
import argparse


def compute_hysteresis_for_vmax(
    stats_dir: Path,
    v_max: float,
    output_dir: Path
):
    """
    Compute hysteresis current for a specific V_max range.

    Args:
        stats_dir: Directory containing aggregated IV statistics
        v_max: Maximum voltage value
        output_dir: Output directory for hysteresis data
    """

    # Load forward and return data
    v_max_clean = str(v_max).replace(".", "p")

    forward_file = stats_dir / f"forward_vmax{v_max_clean}V.csv"
    return_file = stats_dir / f"return_vmax{v_max_clean}V.csv"
    return_fit_file = stats_dir / f"return_with_fit_vmax{v_max_clean}V.csv"

    if not forward_file.exists():
        print(f"Warning: {forward_file} not found, skipping V_max={v_max}V")
        return None

    if not return_file.exists():
        print(f"Warning: {return_file} not found, skipping V_max={v_max}V")
        return None

    # Load data
    forward_df = pl.read_csv(forward_file)
    return_df = pl.read_csv(return_file)

    # Round voltages to avoid floating point precision issues
    # This ensures -7.9005 and -7.9 are treated as the same
    forward_df = forward_df.with_columns(
        pl.col("V (V)").round(2).alias("V_rounded")
    )
    return_df = return_df.with_columns(
        pl.col("V (V)").round(2).alias("V_rounded")
    )

    # Join on rounded voltage to align data points
    joined = forward_df.join(
        return_df,
        on="V_rounded",
        suffix="_return"
    )

    if len(joined) == 0:
        print(f"Warning: No matching voltage points for V_max={v_max}V")
        return None

    # Use the forward voltage column as the primary voltage
    joined = joined.rename({"V (V)": "V (V)"}).drop("V (V)_return", "V_rounded")

    # Compute raw hysteresis (forward - return)
    hysteresis_df = joined.with_columns([
        (pl.col("I_mean") - pl.col("I_mean_return")).alias("I_hysteresis_raw"),
        # Propagate uncertainties: σ_diff = sqrt(σ_fwd² + σ_ret²)
        (pl.col("I_std")**2 + pl.col("I_std_return")**2).sqrt().alias("I_hysteresis_std"),
    ])

    # Select relevant columns for raw hysteresis
    hysteresis_raw = hysteresis_df.select([
        "V (V)",
        "I_hysteresis_raw",
        "I_hysteresis_std",
        pl.col("I_mean").alias("I_forward"),
        pl.col("I_mean_return").alias("I_return"),
        pl.col("I_std").alias("I_forward_std"),
        pl.col("I_std_return").alias("I_return_std"),
    ])

    # If polynomial fits exist, compute hysteresis for fits
    if return_fit_file.exists():
        return_fit_df = pl.read_csv(return_fit_file)

        # Round voltages for fitting data
        return_fit_df = return_fit_df.with_columns(
            pl.col("V (V)").round(2).alias("V_rounded")
        )

        # Join forward data with return fit data
        # This gives us: I_hyst_poly = I_forward_raw - I_return_fit_poly
        joined_fit = forward_df.join(
            return_fit_df,
            on="V_rounded",
            suffix="_return"
        )

        # Compute hysteresis for each polynomial order
        poly_orders = [1, 3, 5, 7]

        # Create a dataframe with polynomial hysteresis values
        poly_hyst_data = {"V (V)": joined_fit["V (V)"].to_list()}

        for order in poly_orders:
            fit_col = f"I_fit_poly{order}"

            if fit_col in joined_fit.columns:
                # Hysteresis = Forward (raw mean) - Return (fitted polynomial)
                # joined_fit has "I_mean" from forward and "I_fit_polyN" from return
                poly_hyst_data[f"I_hysteresis_poly{order}"] = (
                    joined_fit["I_mean"] - joined_fit[fit_col]
                ).to_list()

        # Create dataframe and join with hysteresis_raw
        if len(poly_hyst_data) > 1:  # Check if we have any polynomial columns
            poly_hyst_df = pl.DataFrame(poly_hyst_data)
            hysteresis_raw = hysteresis_raw.join(
                poly_hyst_df,
                on="V (V)",
                how="left"
            )

    # Save hysteresis data
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"hysteresis_vmax{v_max_clean}V.csv"
    hysteresis_raw.write_csv(output_file)

    print(f"  Saved: {output_file}")

    # Compute summary statistics
    summary = {
        "v_max": v_max,
        "n_points": len(hysteresis_raw),
        "hyst_mean_raw": hysteresis_raw["I_hysteresis_raw"].mean(),
        "hyst_std_raw": hysteresis_raw["I_hysteresis_raw"].std(),
        "hyst_max_raw": hysteresis_raw["I_hysteresis_raw"].max(),
        "hyst_min_raw": hysteresis_raw["I_hysteresis_raw"].min(),
    }

    # Add polynomial hysteresis statistics
    for order in [1, 3, 5, 7]:
        col = f"I_hysteresis_poly{order}"
        if col in hysteresis_raw.columns:
            summary[f"hyst_mean_poly{order}"] = hysteresis_raw[col].mean()
            summary[f"hyst_std_poly{order}"] = hysteresis_raw[col].std()
            summary[f"hyst_max_poly{order}"] = hysteresis_raw[col].max()
            summary[f"hyst_min_poly{order}"] = hysteresis_raw[col].min()

    return summary


def compute_hysteresis(stats_dir: Path, output_dir: Path):
    """
    Compute hysteresis current for all V_max ranges in the stats directory.

    Args:
        stats_dir: Directory containing aggregated IV statistics
        output_dir: Output directory for hysteresis results
    """

    # Find all V_max values from forward files
    forward_files = sorted(stats_dir.glob("forward_vmax*.csv"))

    if not forward_files:
        print("No forward data files found")
        return

    print(f"Found {len(forward_files)} voltage ranges\n")

    summaries = []

    for fwd_file in forward_files:
        # Extract V_max from filename
        v_max_str = fwd_file.stem.replace("forward_vmax", "").replace("V", "")
        v_max = float(v_max_str.replace("p", "."))

        print(f"Processing V_max = {v_max}V")

        summary = compute_hysteresis_for_vmax(stats_dir, v_max, output_dir)

        if summary:
            summaries.append(summary)

    # Save summary of all hysteresis results
    if summaries:
        summary_df = pl.DataFrame(summaries)
        summary_file = output_dir / "hysteresis_summary.csv"
        summary_df.write_csv(summary_file)

        print(f"\n✓ Hysteresis computation complete!")
        print(f"  Results saved to: {output_dir}")
        print(f"  Summary: {summary_file}")
        print(f"\nHysteresis Summary:")
        print(summary_df)


def main():
    parser = argparse.ArgumentParser(
        description="Compute hysteresis current from IV statistics"
    )
    parser.add_argument("--stats-dir", type=Path, required=True,
                       help="Directory containing IV statistics")
    parser.add_argument("--output-dir", type=Path, required=True,
                       help="Output directory for hysteresis data")

    args = parser.parse_args()

    compute_hysteresis(args.stats_dir, args.output_dir)


if __name__ == "__main__":
    main()
