#!/usr/bin/env python3
"""
Comprehensive 3-day IV analysis comparison with ELECTRIC FIELD axis.

Channel dimensions:
- Length: 96 μm
- Width: 50 μm

Electric field E = V / L
"""

import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


# Channel dimensions
CHANNEL_LENGTH_UM = 96.0  # micrometers
CHANNEL_LENGTH_CM = CHANNEL_LENGTH_UM * 1e-4  # convert to cm
CHANNEL_WIDTH_UM = 50.0   # micrometers

def voltage_to_efield(V_volts):
    """
    Convert voltage to electric field.

    E = V / L

    Args:
        V_volts: Voltage in Volts

    Returns:
        E in V/cm
    """
    return V_volts / CHANNEL_LENGTH_CM  # V/cm


def load_day_data(date: str, base_dir: Path = Path("data/04_analysis"), exclude_v_max=None):
    """Load hysteresis data for a given date."""
    hyst_dir = base_dir / "hysteresis" / date

    hyst_files = sorted(hyst_dir.glob("hysteresis_vmax*.csv"))
    hyst_detailed = {}

    for f in hyst_files:
        v_max = f.stem.replace("hysteresis_vmax", "").replace("p", ".")
        v_max_float = float(v_max.rstrip("V"))

        # Skip excluded V_max values
        if exclude_v_max and v_max_float in exclude_v_max:
            continue

        df = pl.read_csv(f)
        # Add electric field column in V/cm
        df = df.with_columns(
            (pl.col("V (V)") / CHANNEL_LENGTH_CM).alias("E (V/cm)")
        )
        hyst_detailed[v_max_float] = df

    return {
        "hyst_detailed": hyst_detailed,
        "date": date
    }


def create_comprehensive_plots_efield(days_data: list, output_dir: Path):
    """Create comprehensive comparison plots with electric field axis."""
    output_dir.mkdir(parents=True, exist_ok=True)

    dates = [d["date"] for d in days_data]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green

    # Find common V_max values
    common_v_max = set(days_data[0]["hyst_detailed"].keys())
    for day in days_data[1:]:
        common_v_max = common_v_max.intersection(day["hyst_detailed"].keys())
    v_max_values = sorted(common_v_max)

    print(f"\nCommon V_max ranges: {v_max_values}")
    print(f"Channel length: {CHANNEL_LENGTH_UM} μm")
    print(f"Channel width: {CHANNEL_WIDTH_UM} μm")

    # Create plots for each polynomial order + raw
    poly_orders = ["raw", 1, 3, 5, 7]

    for poly in poly_orders:
        print(f"\nCreating plots for {'Raw data' if poly == 'raw' else f'Polynomial n={poly}'}...")

        # Create figure with subplots for each V_max
        n_ranges = len(v_max_values)
        n_cols = 3
        n_rows = (n_ranges + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
        if n_ranges == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for v_idx, v_max in enumerate(v_max_values):
            ax = axes[v_idx]

            for day_idx, day in enumerate(days_data):
                if v_max in day["hyst_detailed"]:
                    hyst_df = day["hyst_detailed"][v_max]
                    E = hyst_df["E (V/cm)"].to_numpy()  # Electric field

                    if poly == "raw":
                        I_hyst = hyst_df["I_hysteresis_raw"].to_numpy() * 1e9
                        I_std = None
                        if "I_hysteresis_raw_std" in hyst_df.columns:
                            I_std = hyst_df["I_hysteresis_raw_std"].to_numpy() * 1e9
                    else:
                        col_name = f"I_hysteresis_poly{poly}"
                        I_hyst = hyst_df[col_name].to_numpy() * 1e9
                        I_std = None

                    if I_std is not None and np.any(I_std > 0):
                        ax.errorbar(E, I_hyst, yerr=I_std, label=day["date"],
                                   color=colors[day_idx], linewidth=2, alpha=0.7,
                                   capsize=3, errorevery=5)
                    else:
                        ax.plot(E, I_hyst, label=day["date"],
                               color=colors[day_idx], linewidth=2, alpha=0.8)

            ax.set_xlabel("Electric Field (V/cm)", fontsize=12)
            ax.set_ylabel("Hysteresis Current (nA)", fontsize=12)
            e_max = v_max / CHANNEL_LENGTH_CM
            ax.set_title(f"E_max = {e_max:.1f} V/cm (V_max = {v_max}V)",
                        fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)

        # Hide unused subplots
        for idx in range(n_ranges, len(axes)):
            axes[idx].set_visible(False)

        title_str = "Raw Data" if poly == "raw" else f"Polynomial Fit (n={poly})"
        plt.suptitle(f"Hysteresis vs Electric Field - {title_str}\n" +
                     f"Channel: L={CHANNEL_LENGTH_UM}μm, W={CHANNEL_WIDTH_UM}μm",
                     fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        filename = f"hysteresis_efield_{'raw' if poly == 'raw' else f'poly{poly}'}.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Saved: {output_dir / filename}")


def create_combined_view_efield(days_data: list, output_dir: Path):
    """Create a single mega-plot showing all polynomials for one E_max."""
    output_dir.mkdir(parents=True, exist_ok=True)

    dates = [d["date"] for d in days_data]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # Find common V_max values
    common_v_max = set(days_data[0]["hyst_detailed"].keys())
    for day in days_data[1:]:
        common_v_max = common_v_max.intersection(day["hyst_detailed"].keys())
    v_max_values = sorted(common_v_max)

    # Choose middle V_max
    target_v_max = 5.0 if 5.0 in v_max_values else v_max_values[len(v_max_values)//2]
    target_e_max = target_v_max / CHANNEL_LENGTH_CM

    poly_orders = [1, 3, 5, 7]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for poly_idx, poly in enumerate(poly_orders):
        ax = axes[poly_idx]

        for day_idx, day in enumerate(days_data):
            if target_v_max in day["hyst_detailed"]:
                hyst_df = day["hyst_detailed"][target_v_max]
                E = hyst_df["E (V/cm)"].to_numpy()
                col_name = f"I_hysteresis_poly{poly}"
                I_hyst = hyst_df[col_name].to_numpy() * 1e9

                ax.plot(E, I_hyst, label=day["date"],
                       color=colors[day_idx], linewidth=2.5, alpha=0.8)

        ax.set_xlabel("Electric Field (V/cm)", fontsize=12)
        ax.set_ylabel("Hysteresis Current (nA)", fontsize=12)
        ax.set_title(f"Polynomial n={poly}", fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)

    plt.suptitle(f"All Polynomial Orders - E_max = {target_e_max:.0f} V/cm (V_max = {target_v_max}V)\n" +
                 f"Channel: L={CHANNEL_LENGTH_UM}μm, W={CHANNEL_WIDTH_UM}μm",
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    filename = f"combined_all_polynomials_efield_emax{target_e_max:.0f}.png"
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nSaved combined view: {output_dir / filename}")


def create_overlay_efield(days_data: list, output_dir: Path, poly_order=3):
    """Create overlay plot with electric field axis."""
    output_dir.mkdir(parents=True, exist_ok=True)

    dates = [d["date"] for d in days_data]

    # Find common V_max values
    common_v_max = set(days_data[0]["hyst_detailed"].keys())
    for day in days_data[1:]:
        common_v_max = common_v_max.intersection(day["hyst_detailed"].keys())
    v_max_values = sorted(common_v_max)

    day_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    for day_idx, day in enumerate(days_data):
        for v_max in v_max_values:
            if v_max in day["hyst_detailed"]:
                hyst_df = day["hyst_detailed"][v_max]
                E = hyst_df["E (V/cm)"].to_numpy()
                col_name = f"I_hysteresis_poly{poly_order}"
                I_hyst = hyst_df[col_name].to_numpy() * 1e9

                e_max = v_max / CHANNEL_LENGTH_CM
                label = f"{day['date']} - {e_max:.0f} V/cm"
                alpha = 0.6 + 0.1 * (v_max / max(v_max_values))

                ax.plot(E, I_hyst, label=label, color=day_colors[day_idx],
                       linewidth=1.5, alpha=alpha)

    ax.set_xlabel("Electric Field (V/cm)", fontsize=14)
    ax.set_ylabel("Hysteresis Current (nA)", fontsize=14)
    ax.set_title(f"All Days, All E_max Ranges (Polynomial n={poly_order})\n" +
                f"Channel: L={CHANNEL_LENGTH_UM}μm, W={CHANNEL_WIDTH_UM}μm",
                fontsize=16, fontweight='bold')
    ax.legend(fontsize=9, ncol=2, loc='best')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)

    plt.tight_layout()

    filename = f"overlay_all_days_all_efield_poly{poly_order}.png"
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved overlay: {output_dir / filename}")


def print_efield_summary(v_max_values):
    """Print voltage to E-field conversion table."""
    print("\n" + "="*70)
    print("VOLTAGE TO ELECTRIC FIELD CONVERSION")
    print("="*70)
    print(f"Channel Length: {CHANNEL_LENGTH_UM} μm = {CHANNEL_LENGTH_CM} cm")
    print(f"Channel Width: {CHANNEL_WIDTH_UM} μm")
    print("\nConversion: E (V/cm) = V (V) / L (cm)")
    print("\n" + "-"*70)
    print(f"{'V_max (V)':<12} {'E_max (V/cm)':<15} {'E_max (kV/cm)':<15}")
    print("-"*70)
    for v_max in v_max_values:
        e_field_vcm = v_max / CHANNEL_LENGTH_CM
        e_field_kvcm = e_field_vcm / 1000  # Convert V/cm to kV/cm
        print(f"{v_max:<12.1f} {e_field_vcm:<15.1f} {e_field_kvcm:<15.3f}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive 3-day IV comparison with ELECTRIC FIELD axis",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--dates", type=str, nargs='+', required=True,
                       help="Dates to compare (YYYY-MM-DD format)")
    parser.add_argument("--output-dir", type=Path,
                       default=Path("plots/comprehensive_comparison_efield"),
                       help="Output directory for plots")
    parser.add_argument("--exclude-vmax", type=float, nargs='+',
                       help="V_max values to exclude from first date")
    parser.add_argument("--base-dir", type=Path,
                       default=Path("data/04_analysis"),
                       help="Base directory containing analysis results")

    args = parser.parse_args()

    print("="*70)
    print("COMPREHENSIVE IV COMPARISON - ELECTRIC FIELD ANALYSIS")
    print("="*70)
    print(f"Dates: {', '.join(args.dates)}")
    if args.exclude_vmax:
        print(f"Excluding V_max from {args.dates[0]}: {args.exclude_vmax}")
    print(f"Output: {args.output_dir}")
    print("="*70)

    # Load data for all dates
    days_data = []
    for idx, date in enumerate(args.dates):
        print(f"\nLoading data for {date}...")
        exclude = args.exclude_vmax if idx == 0 else None
        data = load_day_data(date, args.base_dir, exclude_v_max=exclude)
        days_data.append(data)
        print(f"  ✓ Loaded {len(data['hyst_detailed'])} voltage ranges")

    # Get V_max values for conversion table
    common_v_max = set(days_data[0]["hyst_detailed"].keys())
    for day in days_data[1:]:
        common_v_max = common_v_max.intersection(day["hyst_detailed"].keys())
    v_max_values = sorted(common_v_max)

    # Print conversion table
    print_efield_summary(v_max_values)

    # Create comprehensive plots
    print("\n" + "="*70)
    print("CREATING COMPREHENSIVE PLOTS (ELECTRIC FIELD)")
    print("="*70)
    create_comprehensive_plots_efield(days_data, args.output_dir)

    # Create combined view
    print("\n" + "="*70)
    print("CREATING COMBINED VIEW (ELECTRIC FIELD)")
    print("="*70)
    create_combined_view_efield(days_data, args.output_dir)

    # Create overlays
    print("\n" + "="*70)
    print("CREATING OVERLAY PLOTS (ELECTRIC FIELD)")
    print("="*70)
    create_overlay_efield(days_data, args.output_dir, poly_order=3)
    create_overlay_efield(days_data, args.output_dir, poly_order=7)

    print("\n" + "="*70)
    print("✓ ELECTRIC FIELD ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nAll outputs saved to: {args.output_dir}")
    print("\nGenerated files:")
    print("  - hysteresis_efield_raw.png")
    print("  - hysteresis_efield_poly1.png")
    print("  - hysteresis_efield_poly3.png")
    print("  - hysteresis_efield_poly5.png")
    print("  - hysteresis_efield_poly7.png")
    print("  - combined_all_polynomials_efield_*.png")
    print("  - overlay_all_days_all_efield_poly3.png")
    print("  - overlay_all_days_all_efield_poly7.png")


if __name__ == "__main__":
    main()
