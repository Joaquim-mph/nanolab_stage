#!/usr/bin/env python3
"""
Comprehensive 3-day IV analysis comparison.

Creates detailed plots for each polynomial order (1, 3, 5, 7) + raw data,
comparing hysteresis across all days.
"""

import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


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

        hyst_detailed[v_max_float] = pl.read_csv(f)

    return {
        "hyst_detailed": hyst_detailed,
        "date": date
    }


def create_comprehensive_plots(days_data: list, output_dir: Path):
    """Create comprehensive comparison plots for all polynomial orders + raw."""
    output_dir.mkdir(parents=True, exist_ok=True)

    dates = [d["date"] for d in days_data]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green

    # Find common V_max values
    common_v_max = set(days_data[0]["hyst_detailed"].keys())
    for day in days_data[1:]:
        common_v_max = common_v_max.intersection(day["hyst_detailed"].keys())
    v_max_values = sorted(common_v_max)

    print(f"\nCommon V_max ranges: {v_max_values}")

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
                    V = hyst_df["V (V)"].to_numpy()

                    if poly == "raw":
                        I_hyst = hyst_df["I_hysteresis_raw"].to_numpy() * 1e9
                        # Check if std column exists
                        if "I_hysteresis_raw_std" in hyst_df.columns:
                            I_std = hyst_df["I_hysteresis_raw_std"].to_numpy() * 1e9
                        else:
                            I_std = None
                    else:
                        col_name = f"I_hysteresis_poly{poly}"
                        I_hyst = hyst_df[col_name].to_numpy() * 1e9
                        I_std = None  # Polynomial fits don't have std

                    if I_std is not None and np.any(I_std > 0):
                        ax.errorbar(V, I_hyst, yerr=I_std, label=day["date"],
                                   color=colors[day_idx], linewidth=2, alpha=0.7,
                                   capsize=3, errorevery=5)
                    else:
                        ax.plot(V, I_hyst, label=day["date"],
                               color=colors[day_idx], linewidth=2, alpha=0.8)

            ax.set_xlabel("V (V)", fontsize=12)
            ax.set_ylabel("Hysteresis Current (nA)", fontsize=12)
            ax.set_title(f"V_max = {v_max}V", fontsize=13, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)

        # Hide unused subplots
        for idx in range(n_ranges, len(axes)):
            axes[idx].set_visible(False)

        title_str = "Raw Data" if poly == "raw" else f"Polynomial Fit (n={poly})"
        plt.suptitle(f"Hysteresis Comparison - {title_str}",
                     fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        filename = f"hysteresis_{'raw' if poly == 'raw' else f'poly{poly}'}.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Saved: {output_dir / filename}")


def create_combined_view(days_data: list, output_dir: Path):
    """Create a single mega-plot showing all polynomials for one V_max."""
    output_dir.mkdir(parents=True, exist_ok=True)

    dates = [d["date"] for d in days_data]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # Find common V_max values
    common_v_max = set(days_data[0]["hyst_detailed"].keys())
    for day in days_data[1:]:
        common_v_max = common_v_max.intersection(day["hyst_detailed"].keys())
    v_max_values = sorted(common_v_max)

    # Create one plot for a representative V_max (e.g., 5V)
    target_v_max = 5.0 if 5.0 in v_max_values else v_max_values[len(v_max_values)//2]

    poly_orders = [1, 3, 5, 7]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for poly_idx, poly in enumerate(poly_orders):
        ax = axes[poly_idx]

        for day_idx, day in enumerate(days_data):
            if target_v_max in day["hyst_detailed"]:
                hyst_df = day["hyst_detailed"][target_v_max]
                V = hyst_df["V (V)"].to_numpy()
                col_name = f"I_hysteresis_poly{poly}"
                I_hyst = hyst_df[col_name].to_numpy() * 1e9

                ax.plot(V, I_hyst, label=day["date"],
                       color=colors[day_idx], linewidth=2.5, alpha=0.8)

        ax.set_xlabel("V (V)", fontsize=12)
        ax.set_ylabel("Hysteresis Current (nA)", fontsize=12)
        ax.set_title(f"Polynomial n={poly}", fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)

    plt.suptitle(f"All Polynomial Orders - V_max = {target_v_max}V",
                 fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()

    filename = f"combined_all_polynomials_vmax{target_v_max}V.png"
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nSaved combined view: {output_dir / filename}")


def create_overlay_all_days_all_vmax(days_data: list, output_dir: Path, poly_order=3):
    """Create overlay plot showing all days and all V_max on one figure."""
    output_dir.mkdir(parents=True, exist_ok=True)

    dates = [d["date"] for d in days_data]

    # Find common V_max values
    common_v_max = set(days_data[0]["hyst_detailed"].keys())
    for day in days_data[1:]:
        common_v_max = common_v_max.intersection(day["hyst_detailed"].keys())
    v_max_values = sorted(common_v_max)

    # Color schemes
    day_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    for day_idx, day in enumerate(days_data):
        for v_max in v_max_values:
            if v_max in day["hyst_detailed"]:
                hyst_df = day["hyst_detailed"][v_max]
                V = hyst_df["V (V)"].to_numpy()
                col_name = f"I_hysteresis_poly{poly_order}"
                I_hyst = hyst_df[col_name].to_numpy() * 1e9

                label = f"{day['date']} - {v_max}V"
                alpha = 0.6 + 0.1 * (v_max / max(v_max_values))

                ax.plot(V, I_hyst, label=label, color=day_colors[day_idx],
                       linewidth=1.5, alpha=alpha)

    ax.set_xlabel("V (V)", fontsize=14)
    ax.set_ylabel("Hysteresis Current (nA)", fontsize=14)
    ax.set_title(f"All Days, All V_max Ranges (Polynomial n={poly_order})",
                fontsize=16, fontweight='bold')
    ax.legend(fontsize=9, ncol=2, loc='best')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)

    plt.tight_layout()

    filename = f"overlay_all_days_all_vmax_poly{poly_order}.png"
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved overlay: {output_dir / filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive 3-day IV comparison with all polynomial orders",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--dates", type=str, nargs='+', required=True,
                       help="Dates to compare (YYYY-MM-DD format)")
    parser.add_argument("--output-dir", type=Path,
                       default=Path("plots/comprehensive_comparison"),
                       help="Output directory for plots")
    parser.add_argument("--exclude-vmax", type=float, nargs='+',
                       help="V_max values to exclude from first date")
    parser.add_argument("--base-dir", type=Path,
                       default=Path("data/04_analysis"),
                       help="Base directory containing analysis results")

    args = parser.parse_args()

    print("="*70)
    print("COMPREHENSIVE IV COMPARISON")
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
        # Only apply exclusion to first date
        exclude = args.exclude_vmax if idx == 0 else None
        data = load_day_data(date, args.base_dir, exclude_v_max=exclude)
        days_data.append(data)
        print(f"  ✓ Loaded {len(data['hyst_detailed'])} voltage ranges")

    # Create comprehensive plots
    print("\n" + "="*70)
    print("CREATING COMPREHENSIVE PLOTS")
    print("="*70)
    create_comprehensive_plots(days_data, args.output_dir)

    # Create combined view
    print("\n" + "="*70)
    print("CREATING COMBINED VIEW")
    print("="*70)
    create_combined_view(days_data, args.output_dir)

    # Create overlay
    print("\n" + "="*70)
    print("CREATING OVERLAY PLOTS")
    print("="*70)
    create_overlay_all_days_all_vmax(days_data, args.output_dir, poly_order=3)
    create_overlay_all_days_all_vmax(days_data, args.output_dir, poly_order=7)

    print("\n" + "="*70)
    print("✓ COMPREHENSIVE ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nAll outputs saved to: {args.output_dir}")
    print("\nGenerated files:")
    print("  - hysteresis_raw.png")
    print("  - hysteresis_poly1.png")
    print("  - hysteresis_poly3.png")
    print("  - hysteresis_poly5.png")
    print("  - hysteresis_poly7.png")
    print("  - combined_all_polynomials_vmax*.png")
    print("  - overlay_all_days_all_vmax_poly3.png")
    print("  - overlay_all_days_all_vmax_poly7.png")


if __name__ == "__main__":
    main()
