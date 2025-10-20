#!/usr/bin/env python3
"""
Comprehensive 3-day IV analysis comparison with ELECTRIC FIELD and CURRENT DENSITY.

Device dimensions:
- Length: 100 μm
- Width: 50 μm
- Area: 5000 μm² = 5×10⁻⁵ cm²

Electric field: E = V / L = V / (100 μm) = V × 100 V/cm
Current density: J = I / A = I / (5×10⁻⁵ cm²)
"""

import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


# Channel dimensions
CHANNEL_LENGTH_UM = 100.0  # micrometers
CHANNEL_LENGTH_CM = CHANNEL_LENGTH_UM * 1e-4  # convert to cm
CHANNEL_WIDTH_UM = 50.0   # micrometers
CHANNEL_AREA_CM2 = (CHANNEL_LENGTH_UM * CHANNEL_WIDTH_UM) * 1e-8  # convert μm² to cm²

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


def current_to_density(I_amps):
    """
    Convert current to current density.

    J = I / A

    Args:
        I_amps: Current in Amperes

    Returns:
        J in A/cm²
    """
    return I_amps / CHANNEL_AREA_CM2  # A/cm²


def load_day_data(date: str, base_dir: Path = Path("data/04_analysis"), exclude_v_max=None):
    """Load hysteresis data for a given date and add E-field and current density columns."""
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

        # Add current density columns for all hysteresis measurements
        # Convert from A to A/cm²
        if "I_hysteresis_raw" in df.columns:
            df = df.with_columns(
                (pl.col("I_hysteresis_raw") / CHANNEL_AREA_CM2).alias("J_hysteresis_raw (A/cm2)")
            )

        for poly in [1, 3, 5, 7]:
            col_name = f"I_hysteresis_poly{poly}"
            if col_name in df.columns:
                df = df.with_columns(
                    (pl.col(col_name) / CHANNEL_AREA_CM2).alias(f"J_hysteresis_poly{poly} (A/cm2)")
                )

        hyst_detailed[v_max_float] = df

    return {
        "hyst_detailed": hyst_detailed,
        "date": date
    }


def create_comprehensive_plots_efield_density(days_data: list, output_dir: Path):
    """Create comprehensive comparison plots with electric field and current density axes."""
    import sys
    from pathlib import Path as P
    sys.path.insert(0, str(P(__file__).parent.parent.parent / 'ploting'))
    from plotting_config import setup_publication_style, save_figure, get_color_cycle

    # Setup publication style with scienceplots
    setup_publication_style('prism_rain')

    output_dir.mkdir(parents=True, exist_ok=True)

    dates = [d["date"] for d in days_data]
    # Use prism_rain palette colors
    colors = get_color_cycle('prism_rain', n_colors=3)

    # Find common V_max values
    common_v_max = set(days_data[0]["hyst_detailed"].keys())
    for day in days_data[1:]:
        common_v_max = common_v_max.intersection(day["hyst_detailed"].keys())
    v_max_values = sorted(common_v_max)

    print(f"\nCommon V_max ranges: {v_max_values}")
    print(f"Channel dimensions: {CHANNEL_LENGTH_UM} μm × {CHANNEL_WIDTH_UM} μm")
    print(f"Channel area: {CHANNEL_AREA_CM2} cm²")

    # Create plots for each polynomial order + raw
    poly_orders = ["raw", 1, 3, 5, 7]

    for poly in poly_orders:
        print(f"\nCreating plots for {'Raw data' if poly == 'raw' else f'Polynomial n={poly}'}...")

        # Create figure with subplots for each V_max
        n_ranges = len(v_max_values)
        n_cols = 3
        n_rows = (n_ranges + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
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
                        J_col = "J_hysteresis_raw (A/cm2)"
                        if J_col in hyst_df.columns:
                            J_hyst = hyst_df[J_col].to_numpy()
                        else:
                            continue
                    else:
                        J_col = f"J_hysteresis_poly{poly} (A/cm2)"
                        if J_col in hyst_df.columns:
                            J_hyst = hyst_df[J_col].to_numpy()
                        else:
                            continue

                    ax.plot(E, J_hyst, label=day["date"],
                           color=colors[day_idx], linewidth=2, alpha=0.8)

            ax.set_xlabel("Electric Field E (V/cm)", fontsize=11)
            ax.set_ylabel("Current Density J (A/cm²)", fontsize=11)
            e_max = v_max / CHANNEL_LENGTH_CM
            ax.set_title(f"E_max = {e_max:.0f} V/cm (V_max = {v_max}V)",
                        fontsize=12, fontweight='bold')
            ax.legend(fontsize=9, loc='best', frameon=True, framealpha=0.9)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
            ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
            ax.tick_params(labelsize=10)

        # Hide unused subplots
        for idx in range(n_ranges, len(axes)):
            axes[idx].set_visible(False)

        title_str = "Raw Data" if poly == "raw" else f"Polynomial Fit (n={poly})"
        plt.suptitle(f"Hysteresis Current Density vs Electric Field - {title_str}\n" +
                     f"Device: L={CHANNEL_LENGTH_UM}μm, W={CHANNEL_WIDTH_UM}μm, A={CHANNEL_AREA_CM2:.2e}cm²",
                     fontsize=14, fontweight='bold', y=0.998)
        plt.tight_layout(rect=[0, 0, 1, 0.99])

        filename = f"hysteresis_efield_density_{'raw' if poly == 'raw' else f'poly{poly}'}"
        save_figure(fig, str(output_dir / filename), formats=['png', 'pdf'])

        print(f"  Saved: {output_dir / filename}.png")


def create_combined_view_efield_density(days_data: list, output_dir: Path):
    """Create a single mega-plot showing all polynomials for one E_max."""
    import sys
    from pathlib import Path as P
    sys.path.insert(0, str(P(__file__).parent.parent.parent / 'ploting'))
    from plotting_config import setup_publication_style, save_figure, get_color_cycle

    setup_publication_style('prism_rain')

    output_dir.mkdir(parents=True, exist_ok=True)

    dates = [d["date"] for d in days_data]
    colors = get_color_cycle('prism_rain', n_colors=3)

    # Find common V_max values
    common_v_max = set(days_data[0]["hyst_detailed"].keys())
    for day in days_data[1:]:
        common_v_max = common_v_max.intersection(day["hyst_detailed"].keys())
    v_max_values = sorted(common_v_max)

    # Choose middle V_max
    target_v_max = 8.0 if 8.0 in v_max_values else v_max_values[len(v_max_values)//2]
    target_e_max = target_v_max / CHANNEL_LENGTH_CM

    poly_orders = [1, 3, 5, 7]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for poly_idx, poly in enumerate(poly_orders):
        ax = axes[poly_idx]

        for day_idx, day in enumerate(days_data):
            if target_v_max in day["hyst_detailed"]:
                hyst_df = day["hyst_detailed"][target_v_max]
                E = hyst_df["E (V/cm)"].to_numpy()
                J_col = f"J_hysteresis_poly{poly} (A/cm2)"

                if J_col in hyst_df.columns:
                    J_hyst = hyst_df[J_col].to_numpy()

                    ax.plot(E, J_hyst, label=day["date"],
                           color=colors[day_idx], linewidth=2, alpha=0.8)

        ax.set_xlabel("Electric Field E (V/cm)", fontsize=11)
        ax.set_ylabel("Current Density J (A/cm²)", fontsize=11)
        ax.set_title(f"Polynomial n={poly}", fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='best', frameon=True, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        ax.tick_params(labelsize=10)

    plt.suptitle(f"All Polynomial Orders - E_max = {target_e_max:.0f} V/cm (V_max = {target_v_max}V)\n" +
                 f"Device: L={CHANNEL_LENGTH_UM}μm, W={CHANNEL_WIDTH_UM}μm, A={CHANNEL_AREA_CM2:.2e}cm²",
                 fontsize=13, fontweight='bold', y=0.998)
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    filename = f"combined_all_polynomials_efield_density_emax{target_e_max:.0f}"
    save_figure(fig, str(output_dir / filename), formats=['png', 'pdf'])

    print(f"\nSaved combined view: {output_dir / filename}.png")


def create_overlay_efield_density(days_data: list, output_dir: Path, poly_order=3):
    """Create overlay plot with electric field and current density axes."""
    import sys
    from pathlib import Path as P
    sys.path.insert(0, str(P(__file__).parent.parent.parent / 'ploting'))
    from plotting_config import setup_publication_style, save_figure, get_color_cycle

    setup_publication_style('prism_rain')

    output_dir.mkdir(parents=True, exist_ok=True)

    dates = [d["date"] for d in days_data]

    # Find common V_max values
    common_v_max = set(days_data[0]["hyst_detailed"].keys())
    for day in days_data[1:]:
        common_v_max = common_v_max.intersection(day["hyst_detailed"].keys())
    v_max_values = sorted(common_v_max)

    day_colors = get_color_cycle('prism_rain', n_colors=3)

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    for day_idx, day in enumerate(days_data):
        for v_max in v_max_values:
            if v_max in day["hyst_detailed"]:
                hyst_df = day["hyst_detailed"][v_max]
                E = hyst_df["E (V/cm)"].to_numpy()
                J_col = f"J_hysteresis_poly{poly_order} (A/cm2)"

                if J_col in hyst_df.columns:
                    J_hyst = hyst_df[J_col].to_numpy()

                    e_max = v_max / CHANNEL_LENGTH_CM
                    label = f"{day['date']} - {e_max:.0f} V/cm"
                    alpha = 0.6 + 0.1 * (v_max / max(v_max_values))

                    ax.plot(E, J_hyst, label=label, color=day_colors[day_idx],
                           linewidth=1.8, alpha=alpha)

    ax.set_xlabel("Electric Field E (V/cm)", fontsize=12)
    ax.set_ylabel("Current Density J (A/cm²)", fontsize=12)
    ax.set_title(f"All Days, All E_max Ranges (Polynomial n={poly_order})\n" +
                f"Device: L={CHANNEL_LENGTH_UM}μm, W={CHANNEL_WIDTH_UM}μm, A={CHANNEL_AREA_CM2:.2e}cm²",
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, ncol=3, loc='best', frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    ax.tick_params(labelsize=10)

    plt.tight_layout()

    filename = f"overlay_all_days_all_efield_density_poly{poly_order}"
    save_figure(fig, str(output_dir / filename), formats=['png', 'pdf'])

    print(f"Saved overlay: {output_dir / filename}.png")


def print_conversion_summary(v_max_values):
    """Print voltage to E-field and current density conversion table."""
    print("\n" + "="*70)
    print("UNIT CONVERSION SUMMARY")
    print("="*70)
    print(f"Device dimensions:")
    print(f"  Length (L): {CHANNEL_LENGTH_UM} μm = {CHANNEL_LENGTH_CM} cm")
    print(f"  Width (W): {CHANNEL_WIDTH_UM} μm")
    print(f"  Area (A): {CHANNEL_AREA_CM2:.2e} cm²")
    print(f"\nConversions:")
    print(f"  E-field: E = V / L = V × {1/CHANNEL_LENGTH_CM:.1f} V/cm")
    print(f"  Current density: J = I / A = I × {1/CHANNEL_AREA_CM2:.2e} A/cm²")
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
        description="Comprehensive 3-day IV comparison with ELECTRIC FIELD and CURRENT DENSITY",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--dates", type=str, nargs='+', required=True,
                       help="Dates to compare (YYYY-MM-DD format)")
    parser.add_argument("--output-dir", type=Path,
                       default=Path("plots/comprehensive_comparison_efield_density"),
                       help="Output directory for plots")
    parser.add_argument("--exclude-vmax", type=float, nargs='+',
                       help="V_max values to exclude from first date")
    parser.add_argument("--base-dir", type=Path,
                       default=Path("data/04_analysis"),
                       help="Base directory containing analysis results")

    args = parser.parse_args()

    print("="*70)
    print("COMPREHENSIVE IV COMPARISON - E-FIELD AND CURRENT DENSITY")
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
    print_conversion_summary(v_max_values)

    # Create comprehensive plots
    print("\n" + "="*70)
    print("CREATING COMPREHENSIVE PLOTS (E-FIELD & CURRENT DENSITY)")
    print("="*70)
    create_comprehensive_plots_efield_density(days_data, args.output_dir)

    # Create combined view
    print("\n" + "="*70)
    print("CREATING COMBINED VIEW (E-FIELD & CURRENT DENSITY)")
    print("="*70)
    create_combined_view_efield_density(days_data, args.output_dir)

    # Create overlays
    print("\n" + "="*70)
    print("CREATING OVERLAY PLOTS (E-FIELD & CURRENT DENSITY)")
    print("="*70)
    create_overlay_efield_density(days_data, args.output_dir, poly_order=3)
    create_overlay_efield_density(days_data, args.output_dir, poly_order=7)

    print("\n" + "="*70)
    print("✓ E-FIELD AND CURRENT DENSITY ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nAll outputs saved to: {args.output_dir}")
    print("\nGenerated files:")
    print("  - hysteresis_efield_density_raw.png/pdf")
    print("  - hysteresis_efield_density_poly1.png/pdf")
    print("  - hysteresis_efield_density_poly3.png/pdf")
    print("  - hysteresis_efield_density_poly5.png/pdf")
    print("  - hysteresis_efield_density_poly7.png/pdf")
    print("  - combined_all_polynomials_efield_density_*.png/pdf")
    print("  - overlay_all_days_all_efield_density_poly3.png/pdf")
    print("  - overlay_all_days_all_efield_density_poly7.png/pdf")


if __name__ == "__main__":
    main()
