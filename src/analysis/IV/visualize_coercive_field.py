#!/usr/bin/env python3
"""
Visualize coercive field analysis results with laser power dependence.

Creates plots showing:
1. Coercive field vs V_max for each date
2. Coercive field vs effective power
3. Max hysteresis current density vs power
"""

import argparse
import polars as pl
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sys


def create_coercive_field_plots(df: pl.DataFrame, output_dir: Path, poly_order: int = 3):
    """
    Create comprehensive coercive field visualization.

    Args:
        df: DataFrame with coercive field analysis results
        output_dir: Directory to save plots
        poly_order: Polynomial order to visualize (default: 3)
    """
    from pathlib import Path as P
    sys.path.insert(0, str(P(__file__).parent.parent.parent / 'ploting'))
    from plotting_config import setup_publication_style, save_figure, get_color_cycle

    setup_publication_style('prism_rain')

    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter for specific polynomial order
    df_plot = df.filter(pl.col("poly_order") == poly_order)

    dates = df_plot["date"].unique().sort().to_list()
    colors = get_color_cycle('prism_rain', n_colors=len(dates))

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Coercive Field Analysis (Polynomial Order {poly_order})\n" +
                 "Device: 100μm × 50μm, Beam-to-sample ratio: 5.7",
                 fontsize=13, fontweight='bold', y=0.998)

    # Plot 1: Coercive field vs V_max
    ax1 = axes[0]
    for idx, date in enumerate(dates):
        df_date = df_plot.filter(pl.col("date") == date)

        # Positive field
        ax1.plot(df_date["V_max"], df_date["E_coercive_pos"],
                 'o-', label=f'{date} (E$_c^+$)', color=colors[idx],
                 linewidth=1.8, markersize=6, alpha=0.8)

        # Negative field (plot absolute value)
        ax1.plot(df_date["V_max"], df_date["E_coercive_neg"].abs(),
                 's--', label=f'{date} (|E$_c^-$|)', color=colors[idx],
                 linewidth=1.8, markersize=6, alpha=0.8)

    ax1.set_xlabel("Maximum Voltage V$_{max}$ (V)", fontsize=11)
    ax1.set_ylabel("Coercive Field |E$_c$| (V/cm)", fontsize=11)
    ax1.set_title("(a) Coercive Field vs Applied Voltage", fontsize=12)
    ax1.legend(fontsize=8, loc='best', ncol=1, frameon=True, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=10)

    # Plot 2: Coercive field vs effective power
    ax2 = axes[1]

    # Filter out dark measurements (power = 0)
    df_light = df_plot.filter(pl.col("effective_power_W") > 0)

    if len(df_light) > 0:
        dates_light = df_light["date"].unique().sort().to_list()

        for idx, date in enumerate(dates_light):
            df_date = df_light.filter(pl.col("date") == date)

            # Convert to μW for better readability
            power_uw = df_date["effective_power_W"] * 1e6

            # Positive field
            ax2.plot(power_uw, df_date["E_coercive_pos"],
                     'o-', label=f'{date} (E$_c^+$)', color=colors[dates.index(date)],
                     linewidth=1.8, markersize=6, alpha=0.8)

            # Negative field
            ax2.plot(power_uw, df_date["E_coercive_neg"].abs(),
                     's--', label=f'{date} (|E$_c^-$|)', color=colors[dates.index(date)],
                     linewidth=1.8, markersize=6, alpha=0.8)

    ax2.set_xlabel("Effective Power on Sample (μW)", fontsize=11)
    ax2.set_ylabel("Coercive Field |E$_c$| (V/cm)", fontsize=11)
    ax2.set_title("(b) Coercive Field vs Laser Power", fontsize=12)
    ax2.legend(fontsize=8, loc='best', ncol=1, frameon=True, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=10)

    # Plot 3: Max hysteresis current density vs power
    ax3 = axes[2]

    for idx, date in enumerate(dates):
        df_date = df_plot.filter(pl.col("date") == date)

        # Get power (0 for dark, actual value for light)
        power_uw = df_date["effective_power_W"] * 1e6

        # Positive field
        ax3.semilogy(df_date["V_max"], df_date["J_max_pos"],
                     'o-', label=f'{date} (J$_{{max}}^+$)', color=colors[idx],
                     linewidth=1.8, markersize=6, alpha=0.8)

        # Negative field
        ax3.semilogy(df_date["V_max"], df_date["J_max_neg"],
                     's--', label=f'{date} (J$_{{max}}^-$)', color=colors[idx],
                     linewidth=1.8, markersize=6, alpha=0.8)

    ax3.set_xlabel("Maximum Voltage V$_{max}$ (V)", fontsize=11)
    ax3.set_ylabel("Max Hysteresis |J| (A/cm²)", fontsize=11)
    ax3.set_title("(c) Max Hysteresis Current Density", fontsize=12)
    ax3.legend(fontsize=8, loc='best', ncol=1, frameon=True, framealpha=0.9)
    ax3.grid(True, alpha=0.3, which='both')
    ax3.tick_params(labelsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    output_file = output_dir / f"coercive_field_analysis_poly{poly_order}"
    save_figure(fig, str(output_file), formats=['png', 'pdf'])
    print(f"✓ Saved: {output_file}.png")


def create_power_dependence_plot(df: pl.DataFrame, output_dir: Path):
    """
    Create focused plot showing power dependence of coercive field.

    Shows all polynomial orders for comparison.
    """
    from pathlib import Path as P
    sys.path.insert(0, str(P(__file__).parent.parent.parent / 'ploting'))
    from plotting_config import setup_publication_style, save_figure, get_color_cycle

    setup_publication_style('prism_rain')

    # Filter only measurements with light
    df_light = df.filter(pl.col("effective_power_W") > 0)

    if len(df_light) == 0:
        print("Warning: No measurements with laser power found")
        return

    poly_orders = df_light["poly_order"].unique().sort().to_list()
    colors_poly = get_color_cycle('prism_rain', n_colors=len(poly_orders))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Laser Power Dependence of Coercive Field\n" +
                 "Device: 100μm × 50μm, Beam-to-sample ratio: 5.7",
                 fontsize=14, fontweight='bold')

    # Plot for each V_max
    v_max_values = df_light["V_max"].unique().sort().to_list()

    for ax_idx, v_max in enumerate(v_max_values[:4]):  # First 4 voltage ranges
        ax = axes.flatten()[ax_idx]
        df_vmax = df_light.filter(pl.col("V_max") == v_max)

        for poly_idx, poly_order in enumerate(poly_orders):
            df_poly = df_vmax.filter(pl.col("poly_order") == poly_order)

            if len(df_poly) == 0:
                continue

            # Convert to μW
            power_uw = df_poly["effective_power_W"] * 1e6

            # Positive field
            ax.plot(power_uw, df_poly["E_coercive_pos"],
                    'o-', label=f'poly-{poly_order} (E$_c^+$)',
                    color=colors_poly[poly_idx], linewidth=1.8,
                    markersize=6, alpha=0.8)

            # Negative field
            ax.plot(power_uw, df_poly["E_coercive_neg"].abs(),
                    's--', label=f'poly-{poly_order} (|E$_c^-$|)',
                    color=colors_poly[poly_idx], linewidth=1.8,
                    markersize=6, alpha=0.8)

        ax.set_xlabel("Effective Power (μW)", fontsize=10)
        ax.set_ylabel("Coercive Field (V/cm)", fontsize=10)
        ax.set_title(f"V$_{{max}}$ = {v_max}V (E$_{{max}}$ = {v_max*100:.0f} V/cm)",
                     fontsize=11)
        ax.legend(fontsize=7, loc='best', ncol=2, frameon=True, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    output_file = output_dir / "coercive_field_power_dependence"
    save_figure(fig, str(output_file), formats=['png', 'pdf'])
    print(f"✓ Saved: {output_file}.png")


def create_summary_table(df: pl.DataFrame, output_dir: Path):
    """
    Create formatted summary table of coercive field results.
    """
    # Filter for polynomial order 3 (recommended)
    df_summary = df.filter(pl.col("poly_order") == 3).select([
        "date",
        "V_max",
        "E_coercive_pos",
        "E_coercive_neg",
        "J_max_pos",
        "J_max_neg",
        "laser_voltage_V",
        "effective_power_W"
    ]).sort(["date", "V_max"])

    # Save as CSV
    output_file = output_dir / "coercive_field_summary_poly3.csv"
    df_summary.write_csv(output_file)
    print(f"✓ Saved: {output_file}")

    # Print formatted table
    print("\n" + "="*80)
    print("COERCIVE FIELD SUMMARY (Polynomial Order 3)")
    print("="*80)
    print(df_summary)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize coercive field analysis with laser power dependence"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default="data/04_analysis/coercive_field/coercive_field_analysis.csv",
        help="Input CSV file with coercive field analysis"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots/coercive_field",
        help="Output directory for plots"
    )
    parser.add_argument(
        "--poly-orders",
        type=int,
        nargs="+",
        default=[3],
        help="Polynomial orders to visualize (default: 3)"
    )

    args = parser.parse_args()

    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("COERCIVE FIELD VISUALIZATION")
    print("="*80)
    print(f"\nInput: {input_file}")
    print(f"Output: {output_dir}")
    print(f"Polynomial orders: {args.poly_orders}")

    # Load data
    df = pl.read_csv(input_file)

    print(f"\nLoaded {len(df)} records")
    print(f"Dates: {df['date'].unique().sort().to_list()}")
    print(f"Voltage ranges: {df['V_max'].unique().sort().to_list()}")

    # Create plots for each polynomial order
    for poly_order in args.poly_orders:
        print(f"\nCreating plots for polynomial order {poly_order}...")
        create_coercive_field_plots(df, output_dir, poly_order)

    # Create power dependence plot
    print("\nCreating power dependence plot...")
    create_power_dependence_plot(df, output_dir)

    # Create summary table
    print("\nCreating summary table...")
    create_summary_table(df, output_dir)

    print("\n" + "="*80)
    print("✓ VISUALIZATION COMPLETE!")
    print("="*80)
    print(f"\nPlots saved to: {output_dir}")


if __name__ == "__main__":
    main()
