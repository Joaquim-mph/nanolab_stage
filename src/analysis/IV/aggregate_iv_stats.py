#!/usr/bin/env python3
"""
Aggregate statistics for repeated IV experiments using 4-layer architecture.

This version reads pre-segmented data from the intermediate layer.
Segment detection is handled by the intermediate preprocessing step.

For a given date and voltage range grouping, compute:
- Mean forward and backward traces
- Standard deviation per voltage point
- Polynomial fits to return traces (orders 1, 3, 5, 7)
- Separate results per V_max range
"""

import polars as pl
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
import argparse
from typing import Optional
import sys

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.models.parameters import IVAnalysisParameters
from pydantic import ValidationError


def linear_fit(V, a, b):
    """Linear model: I = a*V + b"""
    return a * V + b


def load_segmented_data(
    intermediate_root: Path,
    date: str,
    procedure: str = "IV",
    chip_number: Optional[str] = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Load pre-segmented data from intermediate layer.

    Returns forward and return segments separately.

    Args:
        intermediate_root: Root of intermediate segmented data
        date: Date in YYYY-MM-DD format
        procedure: Procedure name
        chip_number: Filter by chip number

    Returns:
        (forward_df, return_df) tuple of DataFrames
    """
    # Load all segments for the date
    segment_pattern = str(intermediate_root / f"proc={procedure}" / f"date={date}" / "run_id=*" / "segment=*" / "part-*.parquet")

    print(f"[4-layer mode] Scanning segmented data: {segment_pattern}")

    try:
        lf = pl.scan_parquet(segment_pattern)
        df_all = lf.collect()
    except Exception as e:
        print(f"Error loading segmented data: {e}")
        raise

    print(f"Found {len(df_all)} data points across {df_all['run_id'].n_unique()} runs, {df_all['segment_id'].n_unique()} total segments")

    # Filter by chip number if specified
    if chip_number and "chip_number" in df_all.columns:
        df_all = df_all.filter(pl.col("chip_number") == chip_number)
        print(f"Filtered to chip {chip_number}: {df_all['run_id'].n_unique()} runs")

    # Separate forward and return segments using segment_type metadata
    # Forward segments: forward_negative, forward_positive
    # Return segments: return_negative, return_positive
    forward_df = df_all.filter(pl.col("segment_type").str.contains("forward"))
    return_df = df_all.filter(pl.col("segment_type").str.contains("return"))

    print(f"  Forward segments: {len(forward_df)} points ({forward_df['segment_id'].n_unique()} segments)")
    print(f"  Return segments: {len(return_df)} points ({return_df['segment_id'].n_unique()} segments)")

    return forward_df, return_df


def process_fits_and_save(
    v_max: float,
    n_runs: int,
    forward_stats: pl.DataFrame,
    return_stats: pl.DataFrame,
    output_dir: Path,
    results: list,
    poly_orders: list[int] = [1, 3, 5, 7],
):
    """
    Process polynomial fits and save results for a given V_max group.

    Args:
        v_max: Maximum voltage for this group
        n_runs: Number of runs in this group
        forward_stats: Forward segment statistics
        return_stats: Return segment statistics
        output_dir: Output directory
        results: List to append fit parameters to
        poly_orders: Polynomial orders to fit
    """
    # Get voltage and current arrays for fitting
    v_return = return_stats["V (V)"].to_numpy()
    i_return = return_stats["I_mean"].to_numpy()

    print(f"  Forward: {len(forward_stats)} voltage points")
    print(f"  Return: {len(return_stats)} voltage points")

    # Fit multiple polynomial orders
    poly_fits = {}

    for order in poly_orders:
        try:
            # Use numpy polyfit for polynomial fitting
            coeffs = np.polyfit(v_return, i_return, order)
            i_fit_poly = np.polyval(coeffs, v_return)

            # Compute R-squared
            ss_res = np.sum((i_return - i_fit_poly)**2)
            ss_tot = np.sum((i_return - np.mean(i_return))**2)
            r_squared_poly = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            poly_fits[f"poly{order}"] = {
                "coeffs": coeffs,
                "r_squared": r_squared_poly,
                "order": order
            }

            print(f"  Polynomial order {order}: R² = {r_squared_poly:.6f}")

        except Exception as e:
            print(f"  Polynomial order {order} fit failed: {e}")
            poly_fits[f"poly{order}"] = None

    # Linear fit for backward compatibility
    try:
        popt, pcov = curve_fit(linear_fit, v_return, i_return)
        slope, intercept = popt
        slope_err, intercept_err = np.sqrt(np.diag(pcov))

        # Compute R-squared
        i_fit = linear_fit(v_return, slope, intercept)
        ss_res = np.sum((i_return - i_fit)**2)
        ss_tot = np.sum((i_return - np.mean(i_return))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        print(f"  Linear fit: I = {slope:.3e} * V + {intercept:.3e}")
        print(f"    R² = {r_squared:.4f}")
        print(f"    Resistance = {1/slope:.2e} Ω")

        fit_params = {
            "v_max": v_max,
            "slope": slope,
            "slope_err": slope_err,
            "intercept": intercept,
            "intercept_err": intercept_err,
            "r_squared": r_squared,
            "resistance_ohm": 1/slope if slope != 0 else np.inf,
            "n_runs": n_runs
        }
    except Exception as e:
        print(f"  Linear fit failed: {e}")
        fit_params = {
            "v_max": v_max,
            "slope": None,
            "slope_err": None,
            "intercept": None,
            "intercept_err": None,
            "r_squared": None,
            "resistance_ohm": None,
            "n_runs": n_runs
        }

    results.append(fit_params)

    # Save per-V_max results
    v_max_clean = str(v_max).replace(".", "p")
    forward_stats.write_csv(output_dir / f"forward_vmax{v_max_clean}V.csv")
    return_stats.write_csv(output_dir / f"return_vmax{v_max_clean}V.csv")

    # Add fit columns to return stats (linear + polynomial orders)
    if fit_params["slope"] is not None:
        v_vals = return_stats["V (V)"].to_numpy()

        # Linear fit
        return_with_fit = return_stats.with_columns(
            pl.lit(linear_fit(v_vals, fit_params["slope"], fit_params["intercept"])).alias("I_fit_linear")
        )

        # Add polynomial fits
        for order in poly_orders:
            poly_key = f"poly{order}"
            if poly_fits.get(poly_key) is not None:
                coeffs = poly_fits[poly_key]["coeffs"]
                i_poly = np.polyval(coeffs, v_vals)
                return_with_fit = return_with_fit.with_columns(
                    pl.lit(i_poly).alias(f"I_fit_poly{order}")
                )

        return_with_fit.write_csv(output_dir / f"return_with_fit_vmax{v_max_clean}V.csv")

    # Save polynomial fit coefficients
    poly_summary = {
        "v_max": v_max,
        "n_runs": n_runs
    }
    for order in poly_orders:
        poly_key = f"poly{order}"
        if poly_fits.get(poly_key) is not None:
            poly_summary[f"r2_poly{order}"] = poly_fits[poly_key]["r_squared"]
            # Save coefficients as separate columns
            for i, coeff in enumerate(poly_fits[poly_key]["coeffs"][::-1]):  # Reverse for c0, c1, c2, ...
                poly_summary[f"poly{order}_c{i}"] = coeff
        else:
            poly_summary[f"r2_poly{order}"] = None

    # Store for later saving
    if not hasattr(aggregate_iv_stats, 'poly_results'):
        aggregate_iv_stats.poly_results = []
    aggregate_iv_stats.poly_results.append(poly_summary)


def aggregate_iv_stats(
    intermediate_root: Path,
    date: str,
    output_dir: Path,
    procedure: str = "IV",
    v_max_min: Optional[float] = None,
    chip_number: Optional[str] = None,
    poly_orders: list[int] = [1, 3, 5, 7],
):
    """
    Aggregate IV statistics from pre-segmented intermediate data.

    This is a 4-layer architecture implementation that reads from
    intermediate segmented data instead of detecting segments.

    Args:
        intermediate_root: Root of intermediate segmented data
        date: Date in YYYY-MM-DD format
        output_dir: Where to save results
        procedure: Procedure name (IV, IVg, etc.)
        v_max_min: Filter by V_max parameter (e.g., 1.0, 2.0, ..., 8.0)
        chip_number: Filter by chip number
        poly_orders: Polynomial orders to fit (default: [1, 3, 5, 7])
    """
    print("\n" + "="*80)
    print("4-LAYER ANALYSIS: Using pre-segmented data")
    print("="*80)

    # Load pre-segmented data
    forward_all, return_all = load_segmented_data(
        intermediate_root=intermediate_root,
        date=date,
        procedure=procedure,
        chip_number=chip_number,
    )

    # Detect voltage column
    if "Vg (V)" in forward_all.columns:
        v_col = "Vg (V)"
    elif "Vsd (V)" in forward_all.columns:
        v_col = "Vsd (V)"
    else:
        raise ValueError(f"No voltage column found. Available columns: {forward_all.columns}")

    print(f"Using voltage column: {v_col}")

    # Calculate max voltage for grouping from forward segments
    v_max_per_run = (
        forward_all.group_by("run_id")
        .agg(pl.col(v_col).abs().max().alias("v_max"))
    )

    forward_all = forward_all.join(v_max_per_run, on="run_id")
    return_all = return_all.join(v_max_per_run, on="run_id")

    # Round to nearest integer for grouping
    forward_all = forward_all.with_columns((pl.col("v_max").round(0)).alias("v_max_group"))
    return_all = return_all.with_columns((pl.col("v_max").round(0)).alias("v_max_group"))

    # Filter by v_max if specified
    if v_max_min is not None:
        forward_all = forward_all.filter(pl.col("v_max_group") == v_max_min)
        return_all = return_all.filter(pl.col("v_max_group") == v_max_min)
        groups = [(v_max_min, forward_all, return_all)]
        print(f"Filtering for V_max ≈ {v_max_min}V")
    else:
        # Group by V_max values
        v_max_values = sorted(forward_all["v_max_group"].unique().to_list())
        groups = [
            (
                v_max,
                forward_all.filter(pl.col("v_max_group") == v_max),
                return_all.filter(pl.col("v_max_group") == v_max)
            )
            for v_max in v_max_values
        ]
        print(f"Found V_max ranges: {v_max_values}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each V_max group
    results = []

    for v_max, forward_group, return_group in groups:
        if forward_group.is_empty() or return_group.is_empty():
            print(f"\nSkipping V_max = {v_max}V: no data")
            continue

        n_runs = forward_group["run_id"].n_unique()
        print(f"\nProcessing V_max = {v_max}V ({n_runs} runs)")

        # Compute statistics grouped by voltage
        forward_stats = (
            forward_group
            .group_by(v_col)
            .agg([
                pl.col("I (A)").mean().alias("I_mean"),
                pl.col("I (A)").std().alias("I_std"),
                pl.col("I (A)").count().alias("n_samples")
            ])
            .sort(v_col)
        )

        return_stats = (
            return_group
            .group_by(v_col)
            .agg([
                pl.col("I (A)").mean().alias("I_mean"),
                pl.col("I (A)").std().alias("I_std"),
                pl.col("I (A)").count().alias("n_samples")
            ])
            .sort(v_col)
        )

        # Rename voltage column to standard name
        forward_stats = forward_stats.rename({v_col: "V (V)"})
        return_stats = return_stats.rename({v_col: "V (V)"})

        # Process fits and save
        process_fits_and_save(
            v_max=v_max,
            n_runs=n_runs,
            forward_stats=forward_stats,
            return_stats=return_stats,
            output_dir=output_dir,
            results=results,
            poly_orders=poly_orders,
        )

    # Save summary of all fits
    if results:
        fit_summary = pl.DataFrame(results)
        fit_summary.write_csv(output_dir / "fit_summary.csv")
        print(f"\nSaved fit summary to {output_dir}")
        print(fit_summary)

    # Save polynomial fit summary
    if hasattr(aggregate_iv_stats, 'poly_results') and aggregate_iv_stats.poly_results:
        poly_summary_df = pl.DataFrame(aggregate_iv_stats.poly_results)
        poly_summary_df.write_csv(output_dir / "polynomial_fits_summary.csv")
        print(f"Saved polynomial fits to {output_dir / 'polynomial_fits_summary.csv'}")
        # Clear for next run
        aggregate_iv_stats.poly_results = []


def run_iv_aggregation(params: IVAnalysisParameters) -> None:
    """
    Run IV aggregation with Pydantic-validated parameters.

    Args:
        params: Validated IVAnalysisParameters instance
    """
    # Require intermediate_root in 4-layer mode
    if params.intermediate_root is None:
        raise ValueError(
            "intermediate_root is required for 4-layer analysis. "
            "Please set intermediate_root in your configuration or run preprocessing first."
        )

    output_dir = params.get_stats_dir()

    aggregate_iv_stats(
        intermediate_root=params.intermediate_root,
        date=params.date,
        output_dir=output_dir,
        procedure=params.procedure,
        v_max_min=params.v_max,
        chip_number=params.chip_number,
        poly_orders=params.poly_orders,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate IV statistics from pre-segmented intermediate data (4-layer architecture)",
        epilog="""
Examples:
  # Using JSON config (recommended)
  python aggregate_iv_stats.py --config config/analysis_config.json

  # Using command-line arguments
  python aggregate_iv_stats.py \\
    --intermediate-root data/03_intermediate/iv_segments \\
    --date 2025-10-18 \\
    --output-base-dir data/04_analysis \\
    --procedure IV \\
    --poly-orders 1 3 5 7

Note: This version requires pre-segmented data from the intermediate layer.
      Run preprocessing first: python src/intermediate/IV/iv_preprocessing_script.py
        """
    )

    # Pydantic mode
    parser.add_argument("--config", type=Path, help="Path to JSON configuration file (Pydantic mode)")

    # Direct arguments
    parser.add_argument("--intermediate-root", type=Path, help="Root of intermediate segmented data")
    parser.add_argument("--date", help="Date in YYYY-MM-DD format")
    parser.add_argument("--output-base-dir", type=Path, help="Base output directory for analysis results")
    parser.add_argument("--procedure", type=str, default="IV", help="Procedure name (IV, IVg, etc.)")
    parser.add_argument("--v-max", type=float, help="Filter by specific V_max value")
    parser.add_argument("--chip-number", help="Filter by chip number")
    parser.add_argument("--poly-orders", nargs="+", type=int, default=[1, 3, 5, 7], help="Polynomial orders to fit")

    args = parser.parse_args()

    try:
        # Mode 1: JSON config file (Pydantic)
        if args.config:
            print(f"[info] Loading configuration from {args.config}")
            params = IVAnalysisParameters.model_validate_json(args.config.read_text())
            print("[info] Configuration validated successfully")

        # Mode 2: Command-line arguments
        elif args.intermediate_root and args.date:
            print("[info] Using command-line arguments (creating Pydantic parameters)")

            output_base = args.output_base_dir or Path("data/04_analysis")

            params = IVAnalysisParameters(
                stage_root=Path("data/02_stage/raw_measurements"),  # Not used, but required by schema
                date=args.date,
                output_base_dir=output_base,
                procedure=args.procedure,
                v_max=args.v_max,
                chip_number=args.chip_number,
                poly_orders=args.poly_orders,
                intermediate_root=args.intermediate_root,
                use_segments=True,  # Always true in 4-layer mode
            )

        else:
            parser.print_help()
            print("\n[error] Must provide either --config or (--intermediate-root and --date)")
            sys.exit(1)

        # Run aggregation with validated parameters
        run_iv_aggregation(params)

    except ValidationError as e:
        print(f"\n[error] Parameter validation failed:")
        print(e)
        sys.exit(1)
    except Exception as e:
        print(f"\n[error] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
