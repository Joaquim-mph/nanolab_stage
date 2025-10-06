#!/usr/bin/env python3
"""
Aggregate statistics for repeated IV experiments.

For a given date and voltage range grouping, compute:
- Mean forward and backward traces
- Standard deviation per voltage point
- Fits to backward trace
- Separate results per V_max range
"""

import polars as pl
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
import argparse
from typing import Optional


def linear_fit(V, a, b):
    """Linear model: I = a*V + b"""
    return a * V + b


def polynomial_fit(V, coeffs):
    """Polynomial model: I = sum(coeffs[i] * V^i)"""
    result = np.zeros_like(V)
    for i, coeff in enumerate(coeffs):
        result += coeff * (V ** i)
    return result


def separate_forward_backward(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Separate forward and backward sweeps based on voltage direction.

    Assumes forward sweep goes from low to high voltage, backward goes high to low.
    Detects the turning point where dV/dt changes sign.
    """
    # Sort by voltage to find turning point
    df_sorted = df.sort("VG_V")

    # Find the midpoint index (assumes symmetric sweep)
    n = len(df_sorted)
    mid = n // 2

    forward = df_sorted[:mid]
    backward = df_sorted[mid:].sort("VG_V", descending=True)

    return forward, backward


def aggregate_iv_stats(
    stage_root: Path,
    date: str,
    output_dir: Path,
    procedure: str = "IVg",
    v_max_min: Optional[float] = None,
    chip_number: Optional[str] = None,
):
    """
    Aggregate IV statistics for repeated experiments on a given date.

    Args:
        stage_root: Root of staged data
        date: Date in YYYY-MM-DD format
        output_dir: Where to save results
        procedure: Procedure name (IV, IVg, etc.)
        v_max_min: Filter by V_max parameter (e.g., 1.0, 2.0, ..., 8.0)
        chip_number: Filter by chip number
    """

    # Load staged IV data for the date
    ivg_pattern = str(stage_root / f"proc={procedure}" / f"date={date}" / "run_id=*" / "part-*.parquet")

    print(f"Scanning: {ivg_pattern}")

    try:
        lf = pl.scan_parquet(ivg_pattern)
        df_all = lf.collect()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print(f"Found {len(df_all)} data points across {df_all['run_id'].n_unique()} runs")

    if len(df_all) == 0:
        print("No data found for the specified date and filters")
        return

    # Filter by chip number if specified
    if chip_number and "chip_number" in df_all.columns:
        df_all = df_all.filter(pl.col("chip_number") == chip_number)
        print(f"Filtered to chip {chip_number}: {df_all['run_id'].n_unique()} runs")

    # Detect voltage column name (IVg uses "Vg (V)", IV uses "Vsd (V)")
    if "Vg (V)" in df_all.columns:
        v_col = "Vg (V)"
    elif "Vsd (V)" in df_all.columns:
        v_col = "Vsd (V)"
    else:
        raise ValueError(f"No voltage column found. Available columns: {df_all.columns}")

    print(f"Using voltage column: {v_col}")

    # Calculate max voltage for each run
    v_max_per_run = (
        df_all.group_by("run_id")
        .agg(pl.col(v_col).max().alias("v_max"))
    )

    df_all = df_all.join(v_max_per_run, on="run_id")

    # Round to nearest integer for grouping
    df_all = df_all.with_columns((pl.col("v_max").round(0)).alias("v_max_group"))

    if v_max_min is not None:
        df_all = df_all.filter(pl.col("v_max_group") == v_max_min)
        groups = [(v_max_min, df_all)]
        print(f"Filtering for V_max ≈ {v_max_min}V: {df_all['run_id'].n_unique()} runs")
    else:
        # Group by V_max values
        v_max_values = sorted(df_all["v_max_group"].unique().to_list())
        groups = [(v_max, df_all.filter(pl.col("v_max_group") == v_max)) for v_max in v_max_values]
        print(f"Found V_max ranges: {v_max_values}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each V_max group
    results = []

    for v_max, group_df in groups:
        if group_df.is_empty():
            continue

        n_runs = group_df["run_id"].n_unique()
        print(f"\nProcessing V_max = {v_max}V ({n_runs} runs, {len(group_df)} points)")

        # Pattern observed: 0V → -Vmax → +Vmax → ~0V
        # This gives us 3 segments based on voltage direction:
        # Segment 1: 0 → -Vmax (forward negative)
        # Segment 2: -Vmax → +Vmax (contains return negative AND forward positive)
        # Segment 3: +Vmax → 0 (return positive)
        #
        # We want the RETURN traces:
        # - Return negative: -Vmax → 0 (part of segment 2)
        # - Return positive: +Vmax → 0 (segment 3)

        all_forward_dfs = []    # Forward segments
        all_return_dfs = []     # Return segments (what we fit)

        for idx, run_id in enumerate(group_df["run_id"].unique()):
            run_df = group_df.filter(pl.col("run_id") == run_id)
            run_pd = run_df.select([v_col, "I (A)"]).to_pandas()

            v = run_pd[v_col].values
            i = run_pd["I (A)"].values

            if len(v) < 4:
                continue

            # Find min and max voltage indices
            min_idx = np.argmin(v)
            max_idx = np.argmax(v)

            # Pattern: 0 → min → max → end
            # Segment A: 0 → min (forward negative - skip)
            # Segment B: min → max (includes return negative + forward positive)
            # Segment C: max → end (return positive - fit this)

            # Within segment B, find where it crosses zero (return vs forward)
            # Find zero crossing between min and max
            zero_cross_idx = min_idx
            for j in range(min_idx, max_idx + 1):
                if abs(v[j]) < abs(v[zero_cross_idx]):
                    zero_cross_idx = j

            # Now we have 4 logical segments:
            # 1. Forward negative: 0 → min_idx
            fwd_neg = run_pd.iloc[:min_idx+1].copy()

            # 2. Return negative: min_idx → zero_cross_idx
            ret_neg = run_pd.iloc[min_idx:zero_cross_idx+1].copy()

            # 3. Forward positive: zero_cross_idx → max_idx
            fwd_pos = run_pd.iloc[zero_cross_idx:max_idx+1].copy()

            # 4. Return positive: max_idx → end
            ret_pos = run_pd.iloc[max_idx:].copy()

            # Collect forward segments (skip these)
            if len(fwd_neg) > 1:
                all_forward_dfs.append(pl.from_pandas(fwd_neg))
            if len(fwd_pos) > 1:
                all_forward_dfs.append(pl.from_pandas(fwd_pos))

            # Collect RETURN segments (fit these - the "odd" segments)
            if len(ret_neg) > 1:
                all_return_dfs.append(pl.from_pandas(ret_neg))
            if len(ret_pos) > 1:
                all_return_dfs.append(pl.from_pandas(ret_pos))

        if not all_forward_dfs or not all_return_dfs:
            print(f"  Skipping V_max={v_max}V: insufficient data")
            continue

        # Concatenate segments
        forward_all = pl.concat(all_forward_dfs)
        return_all = pl.concat(all_return_dfs)  # Both return segments (neg + pos)

        # Compute statistics grouped by voltage
        forward_stats = (
            forward_all
            .group_by(v_col)
            .agg([
                pl.col("I (A)").mean().alias("I_mean"),
                pl.col("I (A)").std().alias("I_std"),
                pl.col("I (A)").count().alias("n_samples")
            ])
            .sort(v_col)
        )

        # Compute stats for all return sweeps (both negative and positive returns)
        return_all_stats = (
            return_all
            .group_by(v_col)
            .agg([
                pl.col("I (A)").mean().alias("I_mean"),
                pl.col("I (A)").std().alias("I_std"),
                pl.col("I (A)").count().alias("n_samples")
            ])
            .sort(v_col)
        )

        # Rename voltage column to standard name for output
        forward_stats = forward_stats.rename({v_col: "V (V)"})
        return_all_stats = return_all_stats.rename({v_col: "V (V)"})

        print(f"  Forward segments: {len(forward_stats)} voltage points ({len(all_forward_dfs)} traces)")
        print(f"  Return segments: {len(return_all_stats)} voltage points ({len(all_return_dfs)} traces)")

        # Fit the return traces (segments 2 and 4: -Vmax→0 and +Vmax→0)
        v_return = return_all_stats["V (V)"].to_numpy()
        i_return = return_all_stats["I_mean"].to_numpy()

        # Fit multiple polynomial orders: 1, 3, 5, 7
        poly_fits = {}

        for order in [1, 3, 5, 7]:
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

        # Linear fit (order 1) for backward compatibility
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
        return_all_stats.write_csv(output_dir / f"return_vmax{v_max_clean}V.csv")

        # Add fit columns to return stats (linear + polynomial orders)
        if fit_params["slope"] is not None:
            v_vals = return_all_stats["V (V)"].to_numpy()

            # Linear fit
            return_with_fit = return_all_stats.with_columns(
                pl.lit(linear_fit(v_vals, fit_params["slope"], fit_params["intercept"])).alias("I_fit_linear")
            )

            # Add polynomial fits
            for order in [1, 3, 5, 7]:
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
        for order in [1, 3, 5, 7]:
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

    # Save summary of all fits
    if results:
        fit_summary = pl.DataFrame(results)
        fit_summary.write_csv(output_dir / "fit_summary.csv")
        print(f"\nSaved results to {output_dir}")
        print(fit_summary)

    # Save polynomial fit summary
    if hasattr(aggregate_iv_stats, 'poly_results') and aggregate_iv_stats.poly_results:
        poly_summary_df = pl.DataFrame(aggregate_iv_stats.poly_results)
        poly_summary_df.write_csv(output_dir / "polynomial_fits_summary.csv")
        print(f"\nSaved polynomial fits to {output_dir / 'polynomial_fits_summary.csv'}")
        # Clear for next run
        aggregate_iv_stats.poly_results = []


def main():
    parser = argparse.ArgumentParser(description="Aggregate IV statistics for repeated experiments")
    parser.add_argument("--stage-root", type=Path, default=Path("data/02_stage/raw_measurements"),
                       help="Root of staged data")
    parser.add_argument("--date", required=True, help="Date in YYYY-MM-DD format")
    parser.add_argument("--output-dir", type=Path, default=Path("data/04_analysis/iv_stats"),
                       help="Output directory for results")
    parser.add_argument("--procedure", type=str, default="IVg", help="Procedure name (IV, IVg, etc.)")
    parser.add_argument("--v-max", type=float, help="Filter by specific V_max value")
    parser.add_argument("--chip-number", help="Filter by chip number")

    args = parser.parse_args()

    aggregate_iv_stats(
        stage_root=args.stage_root,
        date=args.date,
        output_dir=args.output_dir,
        procedure=args.procedure,
        v_max_min=args.v_max,
        chip_number=args.chip_number,
    )


if __name__ == "__main__":
    main()
