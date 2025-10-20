#!/usr/bin/env python3
"""
Create comprehensive comparison plots for backward-subtracted hysteresis data.

Three plot types:
1. Per-day overlay: All voltage ranges for each illumination condition
2. Per-day staggered: Same but vertically offset for clarity
3. Cross-day comparison: Same voltage range across different illumination powers
"""

import argparse
import polars as pl
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sys


# Device dimensions
CHANNEL_LENGTH_UM = 100.0
CHANNEL_WIDTH_UM = 50.0
BEAM_TO_SAMPLE_RATIO = 5.7


def load_power_metadata(coercive_field_file: Path) -> dict:
    """
    Load effective power for each date from coercive field analysis.

    Returns dict: {date: effective_power_uW}
    """
    if not coercive_field_file.exists():
        print(f"Warning: Coercive field file not found: {coercive_field_file}")
        return {}

    df = pl.read_csv(coercive_field_file)

    # Get unique date -> effective power mapping
    power_map = {}
    for date in df["date"].unique().sort().to_list():
        power_w = df.filter(pl.col("date") == date)["effective_power_W"][0]
        power_uw = power_w * 1e6  # Convert to μW
        power_map[date] = power_uw

    return power_map


def load_backward_subtracted_data(
    data_dir: Path,
    date: str,
    vmax: float,
    poly_order: int = 5
) -> pl.DataFrame:
    """
    Load backward-subtracted data for a specific date and voltage range.

    Returns DataFrame with E (V/cm) and J_forward_sub (A/cm²) columns.
    """
    vmax_str = str(vmax).replace(".", "p")
    file_path = data_dir / date / f"backward_sub_vmax{vmax_str}V.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    df = pl.read_csv(file_path)

    # Select relevant columns
    j_col = f"J_forward_sub_poly{poly_order} (A/cm2)"

    if j_col not in df.columns:
        raise ValueError(f"Column {j_col} not found in {file_path}")

    return df.select([
        pl.col("E (V/cm)").alias("E"),
        pl.col(j_col).alias("J_forward")
    ])


def create_per_day_overlay_plots(
    data_dir: Path,
    dates: list[str],
    voltage_ranges: list[float],
    power_map: dict,
    output_dir: Path,
    poly_order: int = 5
):
    """
    Create overlay plots: all voltage ranges for each date.

    One figure per date, showing all voltage ranges overlapped.
    """
    from pathlib import Path as P
    sys.path.insert(0, str(P(__file__).parent.parent.parent / 'ploting'))
    from plotting_config import setup_publication_style, save_figure, get_color_cycle

    setup_publication_style('prism_rain')

    colors = get_color_cycle('prism_rain', n_colors=len(voltage_ranges))

    for date in dates:
        effective_power_uw = power_map.get(date, 0.0)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Title with effective power
        title = f"Backward-Subtracted Hysteresis (Polynomial Order {poly_order})\n"
        if effective_power_uw > 0:
            title += f"Effective Power: {effective_power_uw:.1f} μW"
        else:
            title += "Dark (No Illumination)"

        fig.suptitle(title, fontsize=13, fontweight='bold')

        # Plot each voltage range
        for idx, vmax in enumerate(voltage_ranges):
            try:
                df = load_backward_subtracted_data(data_dir, date, vmax, poly_order)

                E = df["E"].to_numpy()
                J = df["J_forward"].to_numpy()

                ax.plot(E, J, '-', label=f'V$_{{max}}$ = {vmax}V',
                       color=colors[idx], linewidth=1.8, alpha=0.8)

            except FileNotFoundError:
                print(f"  Warning: No data for {date}, vmax={vmax}V")
                continue

        ax.set_xlabel("Electric Field E (V/cm)", fontsize=11)
        ax.set_ylabel("Current Density J (A/cm²)", fontsize=11)
        ax.legend(fontsize=9, loc='best', frameon=True, framealpha=0.9)
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
        ax.tick_params(labelsize=10)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save with descriptive filename
        power_str = f"{effective_power_uw:.1f}uW" if effective_power_uw > 0 else "0uW_dark"
        output_file = output_dir / f"overlay_{date}_power{power_str}_poly{poly_order}"
        save_figure(fig, str(output_file), formats=['png', 'pdf'])
        print(f"✓ Saved overlay plot: {output_file}.png")
        plt.close(fig)


def create_per_day_staggered_plots(
    data_dir: Path,
    dates: list[str],
    voltage_ranges: list[float],
    power_map: dict,
    output_dir: Path,
    poly_order: int = 5,
    offset_factor: float = 0.01
):
    """
    Create staggered plots: all voltage ranges vertically offset.

    One figure per date, curves offset to avoid overlap.
    Shows both forward and backward traces.
    """
    from pathlib import Path as P
    sys.path.insert(0, str(P(__file__).parent.parent.parent / 'ploting'))
    from plotting_config import setup_publication_style, save_figure, get_color_cycle

    setup_publication_style('prism_rain')

    colors = get_color_cycle('prism_rain', n_colors=len(voltage_ranges))

    for date in dates:
        effective_power_uw = power_map.get(date, 0.0)

        fig, ax = plt.subplots(figsize=(10, 8))

        # Title with effective power
        title = f"Backward-Subtracted Hysteresis - Staggered (Polynomial Order {poly_order})\n"
        if effective_power_uw > 0:
            title += f"Effective Power: {effective_power_uw:.1f} μW"
        else:
            title += "Dark (No Illumination)"

        fig.suptitle(title, fontsize=13, fontweight='bold')

        # Plot each voltage range with vertical offset
        for idx, vmax in enumerate(voltage_ranges):
            try:
                vmax_str = str(vmax).replace(".", "p")
                file_path = data_dir / date / f"backward_sub_vmax{vmax_str}V.csv"

                if not file_path.exists():
                    raise FileNotFoundError(f"Data file not found: {file_path}")

                df = pl.read_csv(file_path)

                # Get both forward and backward columns
                j_fwd_col = f"J_forward_sub_poly{poly_order} (A/cm2)"
                j_bwd_col = f"J_backward_sub_poly{poly_order} (A/cm2)"

                if j_fwd_col not in df.columns or j_bwd_col not in df.columns:
                    raise ValueError(f"Columns not found in {file_path}")

                E = df["E (V/cm)"].to_numpy()
                J_fwd = df[j_fwd_col].to_numpy()
                J_bwd = df[j_bwd_col].to_numpy()

                # Apply vertical offset
                offset = idx * offset_factor
                J_fwd_offset = J_fwd + offset
                J_bwd_offset = J_bwd + offset

                # Plot forward and backward with same color
                # Only label once in legend
                ax.plot(E, J_fwd_offset, '-', label=f'V$_{{max}}$ = {vmax}V',
                       color=colors[idx], linewidth=1.8, alpha=0.8)
                ax.plot(E, J_bwd_offset, '--',  # Dashed for backward
                       color=colors[idx], linewidth=1.8, alpha=0.8)

            except FileNotFoundError:
                print(f"  Warning: No data for {date}, vmax={vmax}V")
                continue

        ax.set_xlabel("Electric Field E (V/cm)", fontsize=11)
        ax.set_ylabel("Current Density J [a.u.] (offset for clarity)", fontsize=11)
        ax.legend(fontsize=9, loc='best', frameon=True, framealpha=0.9)
        ax.tick_params(labelsize=10)

        # Remove y-axis tick labels since scale is arbitrary
        ax.set_yticklabels([])

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save with descriptive filename
        power_str = f"{effective_power_uw:.1f}uW" if effective_power_uw > 0 else "0uW_dark"
        output_file = output_dir / f"staggered_{date}_power{power_str}_poly{poly_order}"
        save_figure(fig, str(output_file), formats=['png', 'pdf'])
        print(f"✓ Saved staggered plot: {output_file}.png")
        plt.close(fig)


def create_cross_day_comparison_plots(
    data_dir: Path,
    dates: list[str],
    voltage_ranges: list[float],
    power_map: dict,
    output_dir: Path,
    poly_order: int = 5
):
    """
    Create cross-day comparison plots: same voltage range across different powers.

    One figure per voltage range, showing all dates (illumination conditions).
    """
    from pathlib import Path as P
    sys.path.insert(0, str(P(__file__).parent.parent.parent / 'ploting'))
    from plotting_config import setup_publication_style, save_figure, get_color_cycle

    setup_publication_style('prism_rain')

    colors = get_color_cycle('prism_rain', n_colors=len(dates))

    for vmax in voltage_ranges:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Title
        e_max = vmax * 100.0
        title = f"Photo-Response Comparison (Polynomial Order {poly_order})\n"
        title += f"V$_{{max}}$ = {vmax}V (E$_{{max}}$ = {e_max:.0f} V/cm)"

        fig.suptitle(title, fontsize=13, fontweight='bold')

        # Plot each date
        for idx, date in enumerate(dates):
            effective_power_uw = power_map.get(date, 0.0)

            try:
                df = load_backward_subtracted_data(data_dir, date, vmax, poly_order)

                E = df["E"].to_numpy()
                J = df["J_forward"].to_numpy()

                # Label with effective power
                if effective_power_uw > 0:
                    label = f'{effective_power_uw:.1f} μW ({date})'
                else:
                    label = f'0 μW - Dark ({date})'

                ax.plot(E, J, '-', label=label,
                       color=colors[idx], linewidth=2.0, alpha=0.8)

            except FileNotFoundError:
                print(f"  Warning: No data for {date}, vmax={vmax}V")
                continue

        ax.set_xlabel("Electric Field E (V/cm)", fontsize=11)
        ax.set_ylabel("Current Density J (A/cm²)", fontsize=11)
        ax.legend(fontsize=9, loc='best', frameon=True, framealpha=0.9)
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
        ax.tick_params(labelsize=10)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save
        vmax_str = str(vmax).replace(".", "p")
        output_file = output_dir / f"crossday_vmax{vmax_str}V_poly{poly_order}"
        save_figure(fig, str(output_file), formats=['png', 'pdf'])
        print(f"✓ Saved cross-day plot: {output_file}.png")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Create comprehensive comparison plots for backward-subtracted hysteresis"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/04_analysis/backward_substracted_efield_sept",
        help="Root directory with backward-subtracted data"
    )
    parser.add_argument(
        "--dates",
        nargs="+",
        default=["2025-09-11", "2025-09-29", "2025-09-30"],
        help="Dates to process"
    )
    parser.add_argument(
        "--voltage-ranges",
        type=float,
        nargs="+",
        default=[3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        help="Voltage ranges to plot"
    )
    parser.add_argument(
        "--poly-order",
        type=int,
        default=1,
        help="Polynomial order to plot (default: 5)"
    )
    parser.add_argument(
        "--coercive-field-file",
        type=str,
        default="data/04_analysis/coercive_field/coercive_field_analysis.csv",
        help="Coercive field analysis file (for power metadata)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots/backward_subtracted_comparison",
        help="Output directory for plots"
    )
    parser.add_argument(
        "--offset-factor",
        type=float,
        default=0.008,
        help="Vertical offset factor for staggered plots (default: 0.003)"
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    coercive_field_file = Path(args.coercive_field_file)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("BACKWARD-SUBTRACTED HYSTERESIS COMPARISON PLOTS")
    print("=" * 80)
    print(f"\nData directory: {data_dir}")
    print(f"Dates: {args.dates}")
    print(f"Voltage ranges: {args.voltage_ranges}")
    print(f"Polynomial order: {args.poly_order}")
    print(f"Output directory: {output_dir}")

    # Load power metadata
    print("\nLoading effective power metadata...")
    power_map = load_power_metadata(coercive_field_file)

    print("\nEffective power by date:")
    for date in sorted(power_map.keys()):
        power_uw = power_map[date]
        if power_uw > 0:
            print(f"  {date}: {power_uw:.1f} μW")
        else:
            print(f"  {date}: 0 μW (dark)")

    # Create plots
    print("\n" + "=" * 80)
    print("CREATING PLOTS")
    print("=" * 80)

    print("\n1. Per-day overlay plots (all voltage ranges per illumination)...")
    create_per_day_overlay_plots(
        data_dir, args.dates, args.voltage_ranges, power_map,
        output_dir, args.poly_order
    )

    print("\n2. Per-day staggered plots (vertically offset for clarity)...")
    create_per_day_staggered_plots(
        data_dir, args.dates, args.voltage_ranges, power_map,
        output_dir, args.poly_order, args.offset_factor
    )

    print("\n3. Cross-day comparison plots (same V_max, different powers)...")
    create_cross_day_comparison_plots(
        data_dir, args.dates, args.voltage_ranges, power_map,
        output_dir, args.poly_order
    )

    print("\n" + "=" * 80)
    print("✓ ALL PLOTS CREATED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}")
    print(f"\nGenerated files:")
    print(f"  - Per-day overlay: overlay_{{date}}_power{{X}}uW_poly{args.poly_order}.png/pdf")
    print(f"  - Per-day staggered: staggered_{{date}}_power{{X}}uW_poly{args.poly_order}.png/pdf")
    print(f"  - Cross-day comparison: crossday_vmax{{X}}V_poly{args.poly_order}.png/pdf")


if __name__ == "__main__":
    main()
