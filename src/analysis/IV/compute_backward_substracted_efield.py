#!/usr/bin/env python3
"""
Compute "backward-subtracted" hysteresis by subtracting the max-range return fit
from all forward and backward mean traces.

WITH ELECTRIC FIELD AND CURRENT DENSITY UNITS:
- Device dimensions: 100 μm × 50 μm
- Electric field: E = V / (100 μm) = V × 100 V/cm
- Current density: J = I / (100 μm × 50 μm) = I / 5×10⁻⁵ cm²

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


# Device dimensions
CHANNEL_LENGTH_UM = 100.0  # micrometers
CHANNEL_WIDTH_UM = 50.0    # micrometers
CHANNEL_LENGTH_CM = CHANNEL_LENGTH_UM * 1e-4  # convert to cm
CHANNEL_AREA_CM2 = (CHANNEL_LENGTH_UM * CHANNEL_WIDTH_UM) * 1e-8  # convert μm² to cm²


def voltage_to_efield(V_volts):
    """Convert voltage to electric field E = V / L in V/cm."""
    return V_volts / CHANNEL_LENGTH_CM  # V/cm


def current_to_density(I_amps):
    """Convert current to current density J = I / A in A/cm²."""
    return I_amps / CHANNEL_AREA_CM2  # A/cm²


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
        # Add E-field and current density columns
        fit_df = df.select([
            pl.col("V (V)").alias("V_rounded"),
            (pl.col("V (V)") / CHANNEL_LENGTH_CM).alias("E (V/cm)"),
            (pl.col("I_forward") - pl.col(hyst_col)).alias(f"I_fit_poly{order}"),
            ((pl.col("I_forward") - pl.col(hyst_col)) / CHANNEL_AREA_CM2).alias(f"J_fit_poly{order} (A/cm2)")
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
        (pl.col("V (V)") / CHANNEL_LENGTH_CM).alias("E (V/cm)"),
        pl.col("I_forward").alias("I_forward_mean"),
        (pl.col("I_forward") / CHANNEL_AREA_CM2).alias("J_forward_mean (A/cm2)")
    ])

    return_df = df.select([
        pl.col("V (V)").alias("V_rounded"),
        (pl.col("V (V)") / CHANNEL_LENGTH_CM).alias("E (V/cm)"),
        pl.col("I_return").alias("I_return_mean"),
        (pl.col("I_return") / CHANNEL_AREA_CM2).alias("J_return_mean (A/cm2)")
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
    - E (V/cm)
    - I_forward_sub_poly{n} (forward mean - max return fit)
    - I_backward_sub_poly{n} (backward mean - max return fit)
    - J_forward_sub_poly{n} (A/cm2) (current density)
    - J_backward_sub_poly{n} (A/cm2) (current density)
    """
    result = forward_mean.join(backward_mean, on=["V_rounded", "E (V/cm)"], how="inner")

    for order, fit_df in max_range_fits.items():
        # Join the max-range return fit
        fit_col = f"I_fit_poly{order}"
        j_fit_col = f"J_fit_poly{order} (A/cm2)"

        result = result.join(
            fit_df.rename({
                fit_col: f"max_fit_poly{order}",
                j_fit_col: f"max_jfit_poly{order}"
            }),
            on=["V_rounded", "E (V/cm)"],
            how="left"
        )

        # Compute subtracted traces (current)
        result = result.with_columns([
            (pl.col("I_forward_mean") - pl.col(f"max_fit_poly{order}")).alias(f"I_forward_sub_poly{order}"),
            (pl.col("I_return_mean") - pl.col(f"max_fit_poly{order}")).alias(f"I_backward_sub_poly{order}")
        ])

        # Compute subtracted traces (current density)
        result = result.with_columns([
            (pl.col("J_forward_mean (A/cm2)") - pl.col(f"max_jfit_poly{order}")).alias(f"J_forward_sub_poly{order} (A/cm2)"),
            (pl.col("J_return_mean (A/cm2)") - pl.col(f"max_jfit_poly{order}")).alias(f"J_backward_sub_poly{order} (A/cm2)")
        ])

    # Select only relevant columns
    cols_to_keep = ["V_rounded", "E (V/cm)"]
    for order in max_range_fits.keys():
        cols_to_keep.extend([
            f"I_forward_sub_poly{order}",
            f"I_backward_sub_poly{order}",
            f"J_forward_sub_poly{order} (A/cm2)",
            f"J_backward_sub_poly{order} (A/cm2)"
        ])

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
    Uses E-field (V/cm) and current density (A/cm²) units.

    all_results structure: {date: {vmax_label: subtracted_df}}
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import sys
    from pathlib import Path as P
    sys.path.insert(0, str(P(__file__).parent.parent.parent / 'ploting'))
    from plotting_config import setup_publication_style, save_figure, get_color_cycle

    # Setup publication style with scienceplots
    setup_publication_style('prism_rain')

    plots_dir.mkdir(parents=True, exist_ok=True)
    colors = get_color_cycle('prism_rain', n_colors=12)

    for order in poly_orders:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"Backward-Subtracted Hysteresis (Polynomial Order {order})\n" +
                     f"Device: {CHANNEL_LENGTH_UM}μm × {CHANNEL_WIDTH_UM}μm",
                     fontsize=13, fontweight='bold', y=0.998)

        for idx, (date, date_results) in enumerate(sorted(all_results.items())):
            ax = axes[idx]
            ax.set_title(f"Date: {date}", fontsize=12)
            ax.set_xlabel("Electric Field E (V/cm)", fontsize=11)
            ax.set_ylabel("Current Density J (A/cm²)", fontsize=11)
            ax.grid(True, alpha=0.3)

            # Plot each voltage range
            color_idx = 0
            for vmax_label in sorted(date_results.keys()):
                df = date_results[vmax_label]

                # Extract voltage value for legend
                vmax_val = vmax_label.replace("vmax", "").replace("V", "").replace("p", ".")

                # Plot forward and backward subtracted traces (current density)
                j_forward_col = f"J_forward_sub_poly{order} (A/cm2)"
                j_backward_col = f"J_backward_sub_poly{order} (A/cm2)"

                if j_forward_col in df.columns and j_backward_col in df.columns:
                    E = df["E (V/cm)"].to_numpy()
                    J_fwd = df[j_forward_col].to_numpy()
                    J_bwd = df[j_backward_col].to_numpy()

                    ax.plot(E, J_fwd, '-', label=f"{vmax_val}V fwd", alpha=0.8, linewidth=1.8, color=colors[color_idx])
                    ax.plot(E, J_bwd, '--', label=f"{vmax_val}V bwd", alpha=0.8, linewidth=1.8, color=colors[color_idx])
                    color_idx += 1

            ax.legend(fontsize=8, loc='best', ncol=2, frameon=True, framealpha=0.9)
            ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
            ax.tick_params(labelsize=10)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        output_file = plots_dir / f"backward_sub_poly{order}_efield.png"
        save_figure(fig, str(output_file.with_suffix('')), formats=['png', 'pdf'])
        print(f"  Saved plot: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute backward-subtracted hysteresis with E-field and current density units"
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
        default="data/04_analysis/backward_substracted_efield",
        help="Root directory for output files"
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default="plots/backward_substracted_efield",
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

    print("="*70)
    print("BACKWARD-SUBTRACTED HYSTERESIS WITH E-FIELD AND CURRENT DENSITY")
    print("="*70)
    print(f"\nDevice dimensions:")
    print(f"  Length: {CHANNEL_LENGTH_UM} μm = {CHANNEL_LENGTH_CM} cm")
    print(f"  Width: {CHANNEL_WIDTH_UM} μm")
    print(f"  Area: {CHANNEL_AREA_CM2} cm²")
    print(f"\nConversions:")
    print(f"  E-field: E = V / {CHANNEL_LENGTH_CM} cm = V × {1/CHANNEL_LENGTH_CM:.1f} V/cm")
    print(f"  Current density: J = I / {CHANNEL_AREA_CM2} cm² = I × {1/CHANNEL_AREA_CM2:.2e} A/cm²")

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
    print(f"\nOutput saved to:")
    print(f"  Data: {output_root}")
    print(f"  Plots: {plots_dir}")


if __name__ == "__main__":
    main()
