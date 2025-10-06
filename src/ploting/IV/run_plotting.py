#!/usr/bin/env python3
"""
IV Plotting Runner - Generates all visualization plots for IV analysis.

Creates comprehensive visualizations:
1. Aggregated IV traces (forward/return with polynomial fits)
2. Hysteresis plots (all polynomial orders comparison)
3. Statistical exploration (overlays, distributions, trends)
4. Detailed per-range analysis
5. Peak location analysis

Output: Publication-ready figures
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description, optional=False):
    """Run a command and handle errors."""
    print(f"\n{'='*70}")
    print(f"▶ {description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        if optional:
            print(f"\n⚠ Warning: {description} failed (optional, continuing...)")
            return True
        else:
            print(f"\n❌ Error: {description} failed with exit code {result.returncode}")
            return False

    print(f"\n✓ {description} completed successfully")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate all IV analysis plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all plots
  python src/ploting/IV/run_plotting.py \\
    --stats-dir data/04_analysis/iv_stats/2025-09-11 \\
    --hysteresis-dir data/04_analysis/hysteresis/2025-09-11

  # Specify output directory
  python src/ploting/IV/run_plotting.py \\
    --stats-dir data/04_analysis/iv_stats/2025-09-11 \\
    --hysteresis-dir data/04_analysis/hysteresis/2025-09-11 \\
    --output-dir plots/my_analysis

  # Select specific plot types
  python src/ploting/IV/run_plotting.py \\
    --stats-dir data/04_analysis/iv_stats/2025-09-11 \\
    --hysteresis-dir data/04_analysis/hysteresis/2025-09-11 \\
    --plots comparison exploration

  # Use different polynomial order
  python src/ploting/IV/run_plotting.py \\
    --stats-dir data/04_analysis/iv_stats/2025-09-11 \\
    --hysteresis-dir data/04_analysis/hysteresis/2025-09-11 \\
    --poly-order 5

Output Structure:
  plots/{output_dir}/
    ├── iv_traces/                          # From plot_aggregated_iv.py
    │   ├── iv_aggregated_all_return_fit.png
    │   ├── iv_aggregated_return_fit_vmax*.png
    │   └── resistance_vs_vmax_return_fit.png
    │
    ├── hysteresis_comparison/              # From compare_polynomial_orders.py
    │   ├── all_ranges_all_polynomials.png       ⭐ MAIN FIGURE
    │   ├── all_ranges_all_polynomials_compact.png
    │   └── residuals_all_ranges_all_polynomials.png
    │
    ├── hysteresis_exploration/             # From explore_hysteresis.py
    │   ├── overlay_all_ranges_poly*.png
    │   ├── grid_comparison_poly*.png
    │   ├── statistics_summary.png
    │   ├── normalized_comparison.png
    │   └── distribution_analysis.png
    │
    ├── hysteresis_detailed/                # From visualize_hysteresis.py
    │   ├── comprehensive_hysteresis_vmax*.png
    │   ├── polynomial_comparison_vmax*.png
    │   └── hysteresis_heatmap_poly*.png
    │
    └── hysteresis_basic/                   # From plot_hysteresis.py
        ├── hysteresis_raw_all.png
        ├── hysteresis_polynomial_all.png
        ├── hysteresis_vmax*.png
        └── hysteresis_vs_vmax.png
        """
    )

    # Input parameters
    parser.add_argument("--stats-dir", type=Path, required=True,
                       help="Directory containing IV statistics (from run_analysis.py)")
    parser.add_argument("--hysteresis-dir", type=Path, required=True,
                       help="Directory containing hysteresis data")
    parser.add_argument("--peaks-dir", type=Path,
                       help="Directory containing peak analysis (optional)")

    # Output parameters
    parser.add_argument("--output-dir", type=Path,
                       help="Base output directory for plots (default: plots/{stats_dir_name})")

    # Plot selection
    parser.add_argument("--plots", type=str, nargs='+',
                       choices=['traces', 'comparison', 'exploration', 'detailed', 'basic', 'all'],
                       default=['all'],
                       help="Which plot types to generate (default: all)")

    # Plot options
    parser.add_argument("--poly-order", type=int, default=3,
                       choices=[1, 3, 5, 7],
                       help="Polynomial order for exploration plots (default: 3)")
    parser.add_argument("--compact", action="store_true",
                       help="Generate compact versions (comparison plots)")
    parser.add_argument("--residuals", action="store_true",
                       help="Generate residual plots (comparison plots)")

    args = parser.parse_args()

    # Determine output directory
    if args.output_dir:
        base_output_dir = args.output_dir
    else:
        stats_name = args.stats_dir.name
        base_output_dir = Path(f"plots/{stats_name}")

    # Get script directory
    script_dir = Path(__file__).parent

    # Determine which plots to generate
    plot_types = args.plots
    if 'all' in plot_types:
        plot_types = ['traces', 'comparison', 'exploration', 'detailed', 'basic']

    print(f"\n{'='*70}")
    print(f"IV PLOTTING PIPELINE")
    print(f"{'='*70}")
    print(f"Stats dir:     {args.stats_dir}")
    print(f"Hysteresis:    {args.hysteresis_dir}")
    print(f"Peaks dir:     {args.peaks_dir or 'Not specified'}")
    print(f"Output dir:    {base_output_dir}")
    print(f"Plot types:    {', '.join(plot_types)}")
    print(f"Poly order:    {args.poly_order}")
    print(f"{'='*70}")

    success_count = 0
    total_count = 0

    # Plot 1: IV Traces (forward/return with fits)
    if 'traces' in plot_types:
        total_count += 1
        output_traces = base_output_dir / "iv_traces"
        cmd_traces = [
            sys.executable,
            str(script_dir / "plot_aggregated_iv.py"),
            "--stats-dir", str(args.stats_dir),
            "--output-dir", str(output_traces)
        ]

        if run_command(cmd_traces, f"[{total_count}] Plot IV Traces"):
            success_count += 1

    # Plot 2: Hysteresis Comparison (MAIN FIGURE)
    if 'comparison' in plot_types:
        total_count += 1
        output_comparison = base_output_dir / "hysteresis_comparison"
        cmd_comparison = [
            sys.executable,
            str(script_dir / "compare_polynomial_orders.py"),
            "--hysteresis-dir", str(args.hysteresis_dir),
            "--output-dir", str(output_comparison)
        ]

        if args.compact:
            cmd_comparison.append("--compact")
        if args.residuals:
            cmd_comparison.append("--residuals")

        if run_command(cmd_comparison, f"[{total_count}] Generate Hysteresis Comparison (⭐ MAIN FIGURE)"):
            success_count += 1

    # Plot 3: Statistical Exploration
    if 'exploration' in plot_types:
        total_count += 1
        output_exploration = base_output_dir / "hysteresis_exploration"
        cmd_exploration = [
            sys.executable,
            str(script_dir / "explore_hysteresis.py"),
            "--stats-dir", str(args.stats_dir),
            "--hysteresis-dir", str(args.hysteresis_dir),
            "--output-dir", str(output_exploration),
            "--poly-order", str(args.poly_order)
        ]

        if run_command(cmd_exploration, f"[{total_count}] Generate Statistical Exploration"):
            success_count += 1

    # Plot 4: Detailed Per-Range Analysis
    if 'detailed' in plot_types:
        total_count += 1
        output_detailed = base_output_dir / "hysteresis_detailed"
        cmd_detailed = [
            sys.executable,
            str(script_dir / "visualize_hysteresis.py"),
            "--stats-dir", str(args.stats_dir),
            "--hysteresis-dir", str(args.hysteresis_dir),
            "--output-dir", str(output_detailed),
            "--poly-order", str(args.poly_order)
        ]

        if run_command(cmd_detailed, f"[{total_count}] Generate Detailed Per-Range Plots"):
            success_count += 1

    # Plot 5: Basic Hysteresis Plots
    if 'basic' in plot_types:
        total_count += 1
        output_basic = base_output_dir / "hysteresis_basic"
        cmd_basic = [
            sys.executable,
            str(script_dir / "plot_hysteresis.py"),
            "--hysteresis-dir", str(args.hysteresis_dir),
            "--output-dir", str(output_basic)
        ]

        if run_command(cmd_basic, f"[{total_count}] Generate Basic Hysteresis Plots", optional=True):
            success_count += 1

    # Summary
    print(f"\n{'='*70}")
    if success_count == total_count:
        print(f"✓ COMPLETE! All {total_count} plot types generated successfully")
    else:
        print(f"⚠ PARTIAL SUCCESS: {success_count}/{total_count} plot types completed")
    print(f"{'='*70}")
    print(f"\nOutput location: {base_output_dir}")
    print(f"\nKey figures:")
    if 'comparison' in plot_types:
        print(f"  ⭐ Main figure: {base_output_dir}/hysteresis_comparison/all_ranges_all_polynomials.png")
    if 'exploration' in plot_types:
        print(f"  📊 Statistics:  {base_output_dir}/hysteresis_exploration/")
    if 'traces' in plot_types:
        print(f"  📈 IV traces:   {base_output_dir}/iv_traces/")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
