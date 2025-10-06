#!/usr/bin/env python3
"""
Compute "backward-subtracted" hysteresis by subtracting the max-range return fit
from all forward and backward mean traces.

For each day:
1. Identify the maximum voltage range sweep
2. Extract polynomial fits (orders 1,3,5,7) of the return trace from max range
3. For each voltage range:
   - Subtract max-range return fit from forward mean raw data
   - Subtract max-range return fit from backward mean raw data
4. Save results per voltage range and create visualization plots
"""

import argparse
import polars as pl
from pathlib import Path
import sys


def load_return_fits(hysteresis_dir: Path, vmax: float, poly_orders: list[int]) -> dict[int, pl.DataFrame]:
    """
    Load polynomial fits of return trace for a given voltage range.

    Reconstructs return fits from hysteresis data using:
    return_fit = forward - hysteresis_poly{n}
    """
    fits = {}

    vmax_str = str(vmax).replace(".", "p")
    hyst_file = hysteresis_dir / f"hysteresis_vmax{vmax_str}V.csv"

    if not hyst_file.exists():
        raise FileNotFoundError(f"Hysteresis file not found: {hyst_file}")

    df = pl.read_csv(hyst_file)

    for order in poly_orders:
        hyst_col = f"I_hysteresis_poly{order}"
        if hyst_col not in df.columns or "I_forward" not in df.columns:
            raise ValueError(f"Missing required columns in {hyst_file}")

        # Reconstruct return fit: forward - hysteresis = return_fit
        fit_df = df.select([
            pl.col("V (V)").alias("V_rounded"),
            (pl.col("I_forward") - pl.col(hyst_col)).alias(f"I_fit_poly{order}")
        ])

        fits[order] = fit_df

    return fits


def load_mean_traces(hysteresis_dir: Path, vmax: float) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load forward and backward mean traces for a given voltage range from hysteresis file."""
    vmax_str = str(vmax).replace(".", "p")

    hyst_file = hysteresis_dir / f"hysteresis_vmax{vmax_str}V.csv"

    if not hyst_file.exists():
        raise FileNotFoundError(f"Hysteresis file not found: {hyst_file}")

    df = pl.read_csv(hyst_file)

    if "I_forward" not in df.columns or "I_return" not in df.columns:
        raise ValueError(f"Missing I_forward or I_return columns in {hyst_file}")

    forward_df = df.select([
        pl.col("V (V)").alias("V_rounded"),
        pl.col("I_forward").alias("I_forward_mean")
    ])

    return_df = df.select([
        pl.col("V (V)").alias("V_rounded"),
        pl.col("I_return").alias("I_return_mean")
    ])

    return forward_df, return_df


def compute_subtracted_traces(
    forward_mean: pl.DataFrame,
    backward_mean: pl.DataFrame,
    max_range_fits: dict[int, pl.DataFrame]
) -> pl.DataFrame:
    """
    Subtract max-range return fits from forward and backward mean traces.

    Returns DataFrame with columns:
    - V_rounded
    - I_forward_sub_poly{n} (forward mean - max return fit)
    - I_backward_sub_poly{n} (backward mean - max return fit)
    """
    result = forward_mean.join(backward_mean, on="V_rounded", how="inner")

    for order, fit_df in max_range_fits.items():
        # Join the max-range return fit
        fit_col = f"I_fit_poly{order}"
        result = result.join(
            fit_df.rename({fit_col: f"max_fit_poly{order}"}),
            on="V_rounded",
            how="left"
        )

        # Compute subtracted traces
        result = result.with_columns([
            (pl.col("I_forward_mean") - pl.col(f"max_fit_poly{order}")).alias(f"I_forward_sub_poly{order}"),
            (pl.col("I_return_mean") - pl.col(f"max_fit_poly{order}")).alias(f"I_backward_sub_poly{order}")
        ])

    # Select only relevant columns
    cols_to_keep = ["V_rounded"]
    for order in max_range_fits.keys():
        cols_to_keep.extend([f"I_forward_sub_poly{order}", f"I_backward_sub_poly{order}"])

    return result.select(cols_to_keep)


def process_day(
    hysteresis_dir: Path,
    output_dir: Path,
    voltage_ranges: list[float],
    poly_orders: list[int],
    exclude_ranges: list[float] = None
) -> dict[str, pl.DataFrame]:
    """
    Process a single day's data.

    Returns dict mapping vmax -> subtracted traces DataFrame
    """
    if exclude_ranges:
        voltage_ranges = [v for v in voltage_ranges if v not in exclude_ranges]

    # Identify max voltage range
    max_vmax = max(voltage_ranges)
    print(f"  Max voltage range: {max_vmax}V")

    # Load max-range return fits
    print(f"  Loading return fits from max range ({max_vmax}V)...")
    max_range_fits = load_return_fits(hysteresis_dir, max_vmax, poly_orders)

    # Process each voltage range
    results = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    for vmax in voltage_ranges:
        print(f"  Processing vmax={vmax}V...")

        # Load mean traces
        forward_mean, backward_mean = load_mean_traces(hysteresis_dir, vmax)

        # Compute subtracted traces
        subtracted = compute_subtracted_traces(forward_mean, backward_mean, max_range_fits)

        # Save to CSV
        vmax_str = str(vmax).replace(".", "p")
        output_file = output_dir / f"backward_sub_vmax{vmax_str}V.csv"
        subtracted.write_csv(output_file)
        print(f"    Saved: {output_file}")

        results[f"vmax{vmax_str}V"] = subtracted

    return results


def create_plots(
    all_results: dict[str, dict[str, pl.DataFrame]],
    plots_dir: Path,
    poly_orders: list[int]
):
    """
    Create one plot per polynomial order showing all voltage ranges.

    all_results structure: {date: {vmax_label: subtracted_df}}
    """
    import matplotlib.pyplot as plt
    import numpy as np

    plots_dir.mkdir(parents=True, exist_ok=True)

    for order in poly_orders:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"Backward-Subtracted Hysteresis (Polynomial Order {order})", fontsize=14, fontweight='bold')

        for idx, (date, date_results) in enumerate(sorted(all_results.items())):
            ax = axes[idx]
            ax.set_title(f"Date: {date}", fontsize=12)
            ax.set_xlabel("Voltage (V)", fontsize=11)
            ax.set_ylabel("Current (A)", fontsize=11)
            ax.grid(True, alpha=0.3)

            # Plot each voltage range
            for vmax_label in sorted(date_results.keys()):
                df = date_results[vmax_label]

                # Extract voltage value for legend
                vmax_val = vmax_label.replace("vmax", "").replace("V", "").replace("p", ".")

                # Plot forward and backward subtracted traces
                forward_col = f"I_forward_sub_poly{order}"
                backward_col = f"I_backward_sub_poly{order}"

                if forward_col in df.columns and backward_col in df.columns:
                    v = df["V_rounded"].to_numpy()
                    i_fwd = df[forward_col].to_numpy()
                    i_bwd = df[backward_col].to_numpy()

                    ax.plot(v, i_fwd, '-', label=f"{vmax_val}V fwd", alpha=0.7, linewidth=1.5)
                    ax.plot(v, i_bwd, '--', label=f"{vmax_val}V bwd", alpha=0.7, linewidth=1.5)

            ax.legend(fontsize=8, loc='best', ncol=2)

        plt.tight_layout()
        output_file = plots_dir / f"backward_sub_poly{order}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved plot: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute backward-subtracted hysteresis by subtracting max-range return fit"
    )
    parser.add_argument(
        "--hysteresis-dirs",
        nargs="+",
        required=True,
        help="Directories containing hysteresis data (one per date)"
    )
    parser.add_argument(
        "--dates",
        nargs="+",
        required=True,
        help="Date labels corresponding to hysteresis-dirs (e.g., 2025-09-11)"
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="data/04_analysis/backward_substracted",
        help="Root directory for output files"
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default="plots/backward_substracted",
        help="Directory for output plots"
    )
    parser.add_argument(
        "--voltage-ranges",
        type=float,
        nargs="+",
        default=[2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
        help="Voltage ranges to process"
    )
    parser.add_argument(
        "--poly-orders",
        type=int,
        nargs="+",
        default=[1, 3, 5, 7],
        help="Polynomial orders for fits"
    )
    parser.add_argument(
        "--exclude-ranges",
        type=str,
        default=None,
        help="Comma-separated date:vmax pairs to exclude (e.g., '2025-09-11:2.0,2025-09-29:4.0')"
    )

    args = parser.parse_args()

    if len(args.hysteresis_dirs) != len(args.dates):
        print("Error: Number of --hysteresis-dirs must match number of --dates")
        sys.exit(1)

    # Parse exclusions
    exclusions = {}
    if args.exclude_ranges:
        for pair in args.exclude_ranges.split(","):
            date, vmax = pair.split(":")
            if date not in exclusions:
                exclusions[date] = []
            exclusions[date].append(float(vmax))

    output_root = Path(args.output_root)
    plots_dir = Path(args.plots_dir)

    all_results = {}

    # Process each day
    for hyst_dir_str, date in zip(args.hysteresis_dirs, args.dates):
        hyst_dir = Path(hyst_dir_str)
        print(f"\nProcessing date: {date}")
        print(f"  Hysteresis directory: {hyst_dir}")

        if not hyst_dir.exists():
            print(f"  Warning: Directory not found, skipping")
            continue

        # Get exclusions for this date
        exclude_for_date = exclusions.get(date, [])
        if exclude_for_date:
            print(f"  Excluding voltage ranges: {exclude_for_date}")

        # Create output directory for this date
        output_dir = output_root / date

        # Process
        results = process_day(
            hyst_dir,
            output_dir,
            args.voltage_ranges,
            args.poly_orders,
            exclude_ranges=exclude_for_date
        )

        all_results[date] = results

    # Create plots
    print("\nCreating plots...")
    create_plots(all_results, plots_dir, args.poly_orders)

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
