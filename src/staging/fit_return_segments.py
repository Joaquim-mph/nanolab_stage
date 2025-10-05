"""
Fit Odd Polynomials to Return Segments

Fits odd polynomials (orders 1, 3, 5, 7) to return segments of IV sweep data.
Odd polynomials ensure f(V) = -f(-V) symmetry, which is physically appropriate
for many semiconductor devices.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
from numpy.polynomial import Polynomial


# ----------------------------- Config -----------------------------

DEFAULT_VOLTAGE_COL = "Vsd (V)"
DEFAULT_CURRENT_COL = "I (A)"
RETURN_SEGMENT_TYPES = ["return_negative", "return_positive"]
ODD_POLYNOMIAL_ORDERS = [1, 3, 5, 7]

# Segment combination strategies
SEGMENT_COMBINATIONS = {
    "individual": "Fit each segment separately",
    "all": "Fit all segments together (complete IV curve)",
    "near_zero": "Fit return-to-zero segments (|V| decreasing, e.g., 1+3+5)",
    "forward": "Fit forward segments (|V| increasing, e.g., 0+2+4)",
    "negative": "Fit all negative voltage segments (forward + return)",
    "positive": "Fit all positive voltage segments (forward + return)",
}


# ----------------------------- Polynomial Fitting -----------------------------

def fit_odd_polynomial(
    voltage: np.ndarray,
    current: np.ndarray,
    order: int
) -> Tuple[Polynomial, Dict[str, float]]:
    """
    Fit odd polynomial to IV data.

    Odd polynomials have only odd powers: I = a₁·V + a₃·V³ + a₅·V⁵ + ...
    This ensures the physically appropriate symmetry I(-V) = -I(V).

    Args:
        voltage: Voltage values (V)
        current: Current values (A)
        order: Maximum polynomial order (must be odd: 1, 3, 5, 7, ...)

    Returns:
        Tuple of (fitted_polynomial, metrics_dict)

    Example:
        >>> v = np.array([-1, -0.5, 0, 0.5, 1])
        >>> i = np.array([-0.001, -0.0003, 0, 0.0003, 0.001])
        >>> poly, metrics = fit_odd_polynomial(v, i, order=3)
        >>> metrics['r_squared']
        0.9998...
    """
    if order % 2 == 0:
        raise ValueError(f"Polynomial order must be odd, got {order}")

    if len(voltage) < order + 1:
        raise ValueError(f"Need at least {order + 1} points to fit order {order} polynomial, got {len(voltage)}")

    # Build design matrix with only odd powers
    # For order=5: [V, V³, V⁵]
    odd_powers = list(range(1, order + 1, 2))
    X = np.column_stack([voltage ** p for p in odd_powers])

    # Least squares fit
    coeffs, residuals, rank, s = np.linalg.lstsq(X, current, rcond=None)

    # Build full coefficient array (insert zeros for even powers)
    full_coeffs = np.zeros(order + 1)
    for i, power in enumerate(odd_powers):
        full_coeffs[power] = coeffs[i]

    # Create Polynomial object
    poly = Polynomial(full_coeffs)

    # Calculate metrics
    i_pred = poly(voltage)
    ss_res = np.sum((current - i_pred) ** 2)
    ss_tot = np.sum((current - np.mean(current)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    rmse = np.sqrt(np.mean((current - i_pred) ** 2))
    mae = np.mean(np.abs(current - i_pred))

    metrics = {
        "order": order,
        "r_squared": float(r_squared),
        "rmse": float(rmse),
        "mae": float(mae),
        "n_points": len(voltage),
        "coefficients": {f"a{p}": float(full_coeffs[p]) for p in odd_powers}
    }

    return poly, metrics


def fit_all_orders(
    voltage: np.ndarray,
    current: np.ndarray,
    orders: List[int] = ODD_POLYNOMIAL_ORDERS
) -> Dict[int, Tuple[Polynomial, Dict[str, float]]]:
    """
    Fit multiple polynomial orders to the same data.

    Args:
        voltage: Voltage values
        current: Current values
        orders: List of polynomial orders to fit

    Returns:
        Dictionary mapping order -> (polynomial, metrics)
    """
    results = {}

    for order in orders:
        try:
            poly, metrics = fit_odd_polynomial(voltage, current, order)
            results[order] = (poly, metrics)
        except Exception as e:
            print(f"[warn] Failed to fit order {order}: {e}")
            continue

    return results


# ----------------------------- Data Loading -----------------------------

def load_segment(segment_dir: Path) -> Optional[pl.DataFrame]:
    """
    Load a single segment from parquet file.

    Args:
        segment_dir: Path to segment directory (e.g., .../segment=return_negative/)

    Returns:
        DataFrame with segment data, or None if not found
    """
    parquet_files = list(segment_dir.glob("*.parquet"))

    if not parquet_files:
        return None

    try:
        df = pl.read_parquet(parquet_files[0])
        return df
    except Exception as e:
        print(f"[warn] Failed to read {parquet_files[0]}: {e}")
        return None


def discover_run_directories(
    data_root: Path,
    limit: Optional[int] = None
) -> List[Path]:
    """
    Discover all run directories containing segments.

    Args:
        data_root: Root of segmented data
        limit: Maximum number of runs to return

    Returns:
        List of run directory paths
    """
    run_dirs = sorted(data_root.glob("**/run_id=*"))

    # Filter to only directories that contain segment subdirectories
    run_dirs = [
        d for d in run_dirs
        if d.is_dir() and list(d.glob("segment=*"))
    ]

    if limit:
        run_dirs = run_dirs[:limit]

    return run_dirs


def load_all_segments_for_run(
    run_dir: Path,
    voltage_col: str = DEFAULT_VOLTAGE_COL,
    current_col: str = DEFAULT_CURRENT_COL
) -> Optional[Dict[int, pl.DataFrame]]:
    """
    Load all segments for a given run.

    Args:
        run_dir: Path to run directory (e.g., .../run_id=abc123/)
        voltage_col: Voltage column name
        current_col: Current column name

    Returns:
        Dictionary mapping segment_id -> DataFrame, or None if no segments found
    """
    segments = {}

    segment_dirs = sorted(run_dir.glob("segment=*"))

    for seg_dir in segment_dirs:
        if not seg_dir.is_dir():
            continue

        # Extract segment ID from directory name
        seg_id = int(seg_dir.name.split("=")[1])

        # Load parquet
        parquet_files = list(seg_dir.glob("*.parquet"))
        if not parquet_files:
            continue

        try:
            df = pl.read_parquet(parquet_files[0])

            # Check required columns
            if voltage_col not in df.columns or current_col not in df.columns:
                continue

            segments[seg_id] = df

        except Exception as e:
            print(f"[warn] Failed to load {seg_dir}: {e}")
            continue

    return segments if segments else None


def identify_return_to_zero_segments(
    segments: Dict[int, pl.DataFrame],
    voltage_col: str = DEFAULT_VOLTAGE_COL
) -> List[int]:
    """
    Identify segments where voltage is returning toward V=0 (|V| decreasing).

    For a typical 6-segment IV sweep:
    - Segment 0: V: 0 → -Vmax (|V| increasing, forward sweep)
    - Segment 1: V: -Vmax → 0 (|V| decreasing, RETURN TO ZERO) ✓
    - Segment 2: V: 0 → +Vmax (|V| increasing, forward sweep)
    - Segment 3: V: +Vmax → 0 (|V| decreasing, RETURN TO ZERO) ✓
    - Segment 4: V: 0 → -Vmax (|V| increasing, forward sweep)
    - Segment 5: V: -Vmax → 0 (|V| decreasing, RETURN TO ZERO) ✓

    This identifies segments where |V| is decreasing (approaching zero),
    which are typically the "return" or "backward" sweep segments.

    Args:
        segments: Dictionary of segment_id -> DataFrame
        voltage_col: Voltage column name

    Returns:
        List of segment IDs where voltage is returning toward zero
    """
    return_segs = []

    for seg_id, df in segments.items():
        if df.height < 2:
            continue

        # Get voltage at start and end of segment
        v_start = df[voltage_col][0]
        v_end = df[voltage_col][-1]

        # Calculate absolute voltage change
        abs_v_start = abs(v_start)
        abs_v_end = abs(v_end)

        # Segment returns to zero if |V| decreases
        # Allow small tolerance for numerical errors
        if abs_v_end < abs_v_start - 1e-6:
            return_segs.append(seg_id)

            # Also check segment_type if available (should be "return_*")
            if "segment_type" in df.columns:
                seg_type = df["segment_type"][0]
                if "return" not in seg_type.lower():
                    print(f"[warn] Segment {seg_id} has |V| decreasing but segment_type='{seg_type}'")

    return sorted(return_segs)


def identify_forward_segments(
    segments: Dict[int, pl.DataFrame],
    voltage_col: str = DEFAULT_VOLTAGE_COL
) -> List[int]:
    """
    Identify segments where voltage is moving away from V=0 (|V| increasing).

    For a typical 6-segment IV sweep:
    - Segment 0: V: 0 → -Vmax (|V| increasing, FORWARD) ✓
    - Segment 1: V: -Vmax → 0 (|V| decreasing, return)
    - Segment 2: V: 0 → +Vmax (|V| increasing, FORWARD) ✓
    - Segment 3: V: +Vmax → 0 (|V| decreasing, return)
    - Segment 4: V: 0 → -Vmax (|V| increasing, FORWARD) ✓
    - Segment 5: V: -Vmax → 0 (|V| decreasing, return)

    This identifies segments where |V| is increasing (moving away from zero),
    which are typically the "forward" sweep segments.

    Args:
        segments: Dictionary of segment_id -> DataFrame
        voltage_col: Voltage column name

    Returns:
        List of segment IDs where voltage is moving away from zero
    """
    forward_segs = []

    for seg_id, df in segments.items():
        if df.height < 2:
            continue

        # Get voltage at start and end of segment
        v_start = df[voltage_col][0]
        v_end = df[voltage_col][-1]

        # Calculate absolute voltage change
        abs_v_start = abs(v_start)
        abs_v_end = abs(v_end)

        # Segment goes forward if |V| increases
        # Allow small tolerance for numerical errors
        if abs_v_end > abs_v_start + 1e-6:
            forward_segs.append(seg_id)

            # Also check segment_type if available (should be "forward_*")
            if "segment_type" in df.columns:
                seg_type = df["segment_type"][0]
                if "forward" not in seg_type.lower():
                    print(f"[warn] Segment {seg_id} has |V| increasing but segment_type='{seg_type}'")

    return sorted(forward_segs)


def combine_segments(
    segments: Dict[int, pl.DataFrame],
    segment_ids: List[int],
    voltage_col: str = DEFAULT_VOLTAGE_COL,
    current_col: str = DEFAULT_CURRENT_COL
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Combine multiple segments into single voltage/current arrays.

    Args:
        segments: Dictionary of segment_id -> DataFrame
        segment_ids: List of segment IDs to combine
        voltage_col: Voltage column name
        current_col: Current column name

    Returns:
        Tuple of (voltage_array, current_array)
    """
    v_arrays = []
    i_arrays = []

    for seg_id in sorted(segment_ids):
        if seg_id not in segments:
            continue

        df = segments[seg_id]
        v = df[voltage_col].to_numpy()
        i = df[current_col].to_numpy()

        # Remove NaN/inf
        mask = np.isfinite(v) & np.isfinite(i)
        v = v[mask]
        i = i[mask]

        v_arrays.append(v)
        i_arrays.append(i)

    if not v_arrays:
        raise ValueError("No valid segments to combine")

    voltage = np.concatenate(v_arrays)
    current = np.concatenate(i_arrays)

    return voltage, current


# ----------------------------- Processing -----------------------------

def process_run(
    run_dir: Path,
    voltage_col: str = DEFAULT_VOLTAGE_COL,
    current_col: str = DEFAULT_CURRENT_COL,
    orders: List[int] = ODD_POLYNOMIAL_ORDERS,
    combinations: List[str] = ["individual", "all", "near_zero"]
) -> Optional[Dict]:
    """
    Load all segments for a run and fit polynomials to various combinations.

    Args:
        run_dir: Path to run directory
        voltage_col: Voltage column name
        current_col: Current column name
        orders: Polynomial orders to fit
        combinations: List of segment combination strategies to use

    Returns:
        Dictionary with fit results for all combinations, or None on failure
    """
    # Extract metadata from path
    # Expected: .../proc=IV/date=YYYY-MM-DD/run_id=xxx
    parts = run_dir.parts

    run_id = None
    date_part = None
    proc = None

    for part in parts:
        if part.startswith("run_id="):
            run_id = part.split("=")[1]
        elif part.startswith("date="):
            date_part = part.split("=")[1]
        elif part.startswith("proc="):
            proc = part.split("=")[1]

    # Load all segments
    segments = load_all_segments_for_run(run_dir, voltage_col, current_col)

    if not segments:
        return None

    # Identify return-to-zero and forward segments
    return_to_zero_seg_ids = identify_return_to_zero_segments(segments, voltage_col)
    forward_seg_ids = identify_forward_segments(segments, voltage_col)

    # Build result dictionary
    result = {
        "run_id": run_id,
        "date": date_part,
        "proc": proc,
        "run_path": str(run_dir),
        "n_segments": len(segments),
        "segment_ids": sorted(segments.keys()),
        "return_to_zero_segment_ids": return_to_zero_seg_ids,
        "forward_segment_ids": forward_seg_ids,
        "combinations": {}
    }

    # Process each combination strategy
    for combo in combinations:
        if combo == "individual":
            # Fit each segment separately
            for seg_id, df in segments.items():
                voltage = df[voltage_col].to_numpy()
                current = df[current_col].to_numpy()

                # Remove NaN/inf
                mask = np.isfinite(voltage) & np.isfinite(current)
                voltage = voltage[mask]
                current = current[mask]

                if len(voltage) < max(orders) + 1:
                    continue

                try:
                    fits = fit_all_orders(voltage, current, orders)

                    seg_type = df["segment_type"][0] if "segment_type" in df.columns else "unknown"

                    result["combinations"][f"segment_{seg_id}"] = {
                        "strategy": "individual",
                        "segment_ids": [seg_id],
                        "segment_type": seg_type,
                        "n_points": len(voltage),
                        "voltage_range": {"min": float(voltage.min()), "max": float(voltage.max())},
                        "current_range": {"min": float(current.min()), "max": float(current.max())},
                        "fits": {f"order_{order}": metrics for order, (poly, metrics) in fits.items()}
                    }
                except Exception as e:
                    print(f"[warn] Failed to fit segment {seg_id}: {e}")
                    continue

        elif combo == "all":
            # Fit all segments together
            try:
                voltage, current = combine_segments(segments, list(segments.keys()), voltage_col, current_col)

                if len(voltage) >= max(orders) + 1:
                    fits = fit_all_orders(voltage, current, orders)

                    result["combinations"]["all_segments"] = {
                        "strategy": "all",
                        "segment_ids": sorted(segments.keys()),
                        "n_points": len(voltage),
                        "voltage_range": {"min": float(voltage.min()), "max": float(voltage.max())},
                        "current_range": {"min": float(current.min()), "max": float(current.max())},
                        "fits": {f"order_{order}": metrics for order, (poly, metrics) in fits.items()}
                    }
            except Exception as e:
                print(f"[warn] Failed to fit all segments: {e}")

        elif combo == "near_zero":
            # Fit return-to-zero segments (|V| → 0)
            if len(return_to_zero_seg_ids) >= 1:
                try:
                    voltage, current = combine_segments(segments, return_to_zero_seg_ids, voltage_col, current_col)

                    if len(voltage) >= max(orders) + 1:
                        fits = fit_all_orders(voltage, current, orders)

                        result["combinations"]["near_zero"] = {
                            "strategy": "near_zero",
                            "segment_ids": return_to_zero_seg_ids,
                            "n_points": len(voltage),
                            "voltage_range": {"min": float(voltage.min()), "max": float(voltage.max())},
                            "current_range": {"min": float(current.min()), "max": float(current.max())},
                            "fits": {f"order_{order}": metrics for order, (poly, metrics) in fits.items()}
                        }
                except Exception as e:
                    print(f"[warn] Failed to fit return-to-zero segments: {e}")

        elif combo == "forward":
            # Fit forward segments (|V| increasing)
            if len(forward_seg_ids) >= 1:
                try:
                    voltage, current = combine_segments(segments, forward_seg_ids, voltage_col, current_col)

                    if len(voltage) >= max(orders) + 1:
                        fits = fit_all_orders(voltage, current, orders)

                        result["combinations"]["forward"] = {
                            "strategy": "forward",
                            "segment_ids": forward_seg_ids,
                            "n_points": len(voltage),
                            "voltage_range": {"min": float(voltage.min()), "max": float(voltage.max())},
                            "current_range": {"min": float(current.min()), "max": float(current.max())},
                            "fits": {f"order_{order}": metrics for order, (poly, metrics) in fits.items()}
                        }
                except Exception as e:
                    print(f"[warn] Failed to fit forward segments: {e}")

        elif combo == "negative":
            # Fit all negative voltage segments
            neg_seg_ids = []
            for seg_id, df in segments.items():
                if df[voltage_col].mean() < 0:
                    neg_seg_ids.append(seg_id)

            if neg_seg_ids:
                try:
                    voltage, current = combine_segments(segments, neg_seg_ids, voltage_col, current_col)

                    if len(voltage) >= max(orders) + 1:
                        fits = fit_all_orders(voltage, current, orders)

                        result["combinations"]["negative_segments"] = {
                            "strategy": "negative",
                            "segment_ids": neg_seg_ids,
                            "n_points": len(voltage),
                            "voltage_range": {"min": float(voltage.min()), "max": float(voltage.max())},
                            "current_range": {"min": float(current.min()), "max": float(current.max())},
                            "fits": {f"order_{order}": metrics for order, (poly, metrics) in fits.items()}
                        }
                except Exception as e:
                    print(f"[warn] Failed to fit negative segments: {e}")

        elif combo == "positive":
            # Fit all positive voltage segments
            pos_seg_ids = []
            for seg_id, df in segments.items():
                if df[voltage_col].mean() > 0:
                    pos_seg_ids.append(seg_id)

            if pos_seg_ids:
                try:
                    voltage, current = combine_segments(segments, pos_seg_ids, voltage_col, current_col)

                    if len(voltage) >= max(orders) + 1:
                        fits = fit_all_orders(voltage, current, orders)

                        result["combinations"]["positive_segments"] = {
                            "strategy": "positive",
                            "segment_ids": pos_seg_ids,
                            "n_points": len(voltage),
                            "voltage_range": {"min": float(voltage.min()), "max": float(voltage.max())},
                            "current_range": {"min": float(current.min()), "max": float(current.max())},
                            "fits": {f"order_{order}": metrics for order, (poly, metrics) in fits.items()}
                        }
                except Exception as e:
                    print(f"[warn] Failed to fit positive segments: {e}")

    return result if result["combinations"] else None


def save_fit_results(
    results: List[Dict],
    output_file: Path
) -> None:
    """
    Save fit results to JSON file atomically.

    Args:
        results: List of result dictionaries
        output_file: Output JSON file path
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode='w',
        delete=False,
        dir=output_file.parent,
        suffix='.json'
    ) as tmp:
        json.dump(results, tmp, indent=2, ensure_ascii=False)
        tmp_path = Path(tmp.name)

    tmp_path.replace(output_file)


def generate_fitted_curves(
    result: Dict,
    segments: Dict[int, pl.DataFrame],
    voltage_col: str = DEFAULT_VOLTAGE_COL,
    n_points: int = 200
) -> Dict[str, pl.DataFrame]:
    """
    Generate fitted I(V) curves from polynomial fits.

    For each combination and polynomial order, evaluate the polynomial
    at evenly-spaced voltage points to create smooth fitted curves.

    Args:
        result: Fit result dictionary for one run
        segments: Original segment data (for voltage range)
        voltage_col: Voltage column name
        n_points: Number of points to generate per curve

    Returns:
        Dictionary mapping combination_key -> DataFrame with fitted curves
    """
    fitted_curves = {}

    for combo_key, combo_data in result["combinations"].items():
        v_min = combo_data["voltage_range"]["min"]
        v_max = combo_data["voltage_range"]["max"]

        # Generate voltage array
        v_fit = np.linspace(v_min, v_max, n_points)

        # Build DataFrame with fitted curves for all orders
        curve_data = {"V": v_fit}

        for fit_key, metrics in combo_data["fits"].items():
            order = metrics["order"]
            coeffs_dict = metrics["coefficients"]

            # Reconstruct polynomial from coefficients
            # Build coefficient array with zeros for even powers
            max_power = max(int(k.replace("a", "")) for k in coeffs_dict.keys())
            coeffs = np.zeros(max_power + 1)
            for coeff_name, coeff_val in coeffs_dict.items():
                power = int(coeff_name.replace("a", ""))
                coeffs[power] = coeff_val

            # Evaluate polynomial
            poly = Polynomial(coeffs)
            i_fit = poly(v_fit)

            curve_data[f"I_order_{order}"] = i_fit

        fitted_curves[combo_key] = pl.DataFrame(curve_data)

    return fitted_curves


def save_fit_parquet(
    result: Dict,
    output_root: Path,
    fitted_curves: Optional[Dict[str, pl.DataFrame]] = None
) -> List[Path]:
    """
    Save fit results as parquet files in proc/date/run_id structure.

    Creates one file per combination strategy:
    03_intermediate/iv_polynomial_fits/proc=IV/date=YYYY-MM-DD/run_id=xxx/
      ├── individual_fits.parquet      (fit metrics for individual segments)
      ├── individual_curves.parquet    (fitted I(V) curves for individual segments)
      ├── all_fits.parquet             (fit metrics for combined)
      ├── all_curves.parquet           (fitted I(V) curves for combined)
      ├── near_zero_fits.parquet       (fit metrics)
      ├── near_zero_curves.parquet     (fitted I(V) curves)
      └── ...

    Args:
        result: Fit result dictionary for one run
        output_root: Root output directory
        fitted_curves: Optional dictionary of fitted curve DataFrames

    Returns:
        List of paths to written parquet files
    """
    # Extract path components
    proc = result["proc"]
    date = result["date"]
    run_id = result["run_id"]

    # Build output directory
    output_dir = output_root / f"proc={proc}" / f"date={date}" / f"run_id={run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    written_files = []

    # Group combinations by strategy
    combinations_by_strategy = {}
    for combo_key, combo_data in result["combinations"].items():
        strategy = combo_data["strategy"]
        if strategy not in combinations_by_strategy:
            combinations_by_strategy[strategy] = []
        combinations_by_strategy[strategy].append((combo_key, combo_data))

    # Save each strategy to its own parquet file
    for strategy, combo_list in combinations_by_strategy.items():
        rows = []

        for combo_key, combo_data in combo_list:
            for fit_key, metrics in combo_data["fits"].items():
                row = {
                    "run_id": run_id,
                    "proc": proc,
                    "date": date,
                    "combination_key": combo_key,
                    "strategy": strategy,
                    "segment_ids": str(combo_data["segment_ids"]),  # Store as string
                    "n_segments": len(combo_data["segment_ids"]),
                    "segment_type": combo_data.get("segment_type", "combined"),
                    "order": metrics["order"],
                    "r_squared": metrics["r_squared"],
                    "rmse": metrics["rmse"],
                    "mae": metrics["mae"],
                    "n_points": metrics["n_points"],
                    "v_min": combo_data["voltage_range"]["min"],
                    "v_max": combo_data["voltage_range"]["max"],
                    "i_min": combo_data["current_range"]["min"],
                    "i_max": combo_data["current_range"]["max"],
                }

                # Add coefficients
                for coeff_name, coeff_val in metrics["coefficients"].items():
                    row[coeff_name] = coeff_val

                rows.append(row)

        if rows:
            df = pl.DataFrame(rows)

            # Write fit metrics atomically
            output_file = output_dir / f"{strategy}_fits.parquet"

            with tempfile.NamedTemporaryFile(
                mode='wb',
                delete=False,
                dir=output_dir,
                suffix='.parquet'
            ) as tmp:
                tmp_path = Path(tmp.name)

            df.write_parquet(tmp_path)
            tmp_path.replace(output_file)

            written_files.append(output_file)

    # Save fitted curves if provided
    if fitted_curves:
        for combo_key, curve_df in fitted_curves.items():
            # Find strategy for this combo_key
            if combo_key in result["combinations"]:
                strategy = result["combinations"][combo_key]["strategy"]

                # Write curve data atomically
                curve_file = output_dir / f"{strategy}_curves.parquet"

                with tempfile.NamedTemporaryFile(
                    mode='wb',
                    delete=False,
                    dir=output_dir,
                    suffix='.parquet'
                ) as tmp:
                    tmp_path = Path(tmp.name)

                curve_df.write_parquet(tmp_path)
                tmp_path.replace(curve_file)

                written_files.append(curve_file)

    return written_files


def create_summary_dataframe(results: List[Dict]) -> pl.DataFrame:
    """
    Convert fit results to Polars DataFrame for analysis.

    Args:
        results: List of fit result dictionaries

    Returns:
        DataFrame with one row per combination per polynomial order
    """
    rows = []

    for result in results:
        # Iterate through all combinations for this run
        for combo_key, combo_data in result["combinations"].items():
            base = {
                "run_id": result["run_id"],
                "date": result["date"],
                "proc": result["proc"],
                "combination_key": combo_key,
                "strategy": combo_data["strategy"],
                "segment_ids": str(combo_data["segment_ids"]),
                "n_segments": len(combo_data["segment_ids"]),
                "segment_type": combo_data.get("segment_type", "combined"),
                "n_points": combo_data["n_points"],
                "v_min": combo_data["voltage_range"]["min"],
                "v_max": combo_data["voltage_range"]["max"],
                "i_min": combo_data["current_range"]["min"],
                "i_max": combo_data["current_range"]["max"],
            }

            # Iterate through polynomial orders for this combination
            for fit_key, metrics in combo_data["fits"].items():
                row = base.copy()
                row["order"] = metrics["order"]
                row["r_squared"] = metrics["r_squared"]
                row["rmse"] = metrics["rmse"]
                row["mae"] = metrics["mae"]

                # Add coefficients as separate columns
                for coeff_name, coeff_val in metrics["coefficients"].items():
                    row[coeff_name] = coeff_val

                rows.append(row)

    return pl.DataFrame(rows) if rows else pl.DataFrame()


# ----------------------------- CLI -----------------------------

def main() -> None:
    """
    Fit odd polynomials to return segments of IV sweeps.

    Usage examples:

    1. Process first 10 return segments:
        $ python fit_return_segments.py --limit 10

    2. Specific polynomial orders:
        $ python fit_return_segments.py --orders 1 3 5

    3. Custom output location:
        $ python fit_return_segments.py --output-dir results/fits

    4. Process all return segments:
        $ python fit_return_segments.py
    """
    ap = argparse.ArgumentParser(
        description="Fit odd polynomials to return segments of IV sweeps"
    )
    ap.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/03_intermediate"),
        help="Root of segmented data (default: data/03_intermediate)"
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/03_intermediate/iv_polynomial_fits"),
        help="Output directory for fit results (default: data/03_intermediate/iv_polynomial_fits)"
    )
    ap.add_argument(
        "--voltage-col",
        type=str,
        default=DEFAULT_VOLTAGE_COL,
        help=f"Voltage column name (default: {DEFAULT_VOLTAGE_COL})"
    )
    ap.add_argument(
        "--current-col",
        type=str,
        default=DEFAULT_CURRENT_COL,
        help=f"Current column name (default: {DEFAULT_CURRENT_COL})"
    )
    ap.add_argument(
        "--orders",
        type=int,
        nargs='+',
        default=ODD_POLYNOMIAL_ORDERS,
        help=f"Polynomial orders to fit (default: {ODD_POLYNOMIAL_ORDERS})"
    )
    ap.add_argument(
        "--combinations",
        type=str,
        nargs='+',
        default=["individual", "all", "near_zero", "forward"],
        choices=list(SEGMENT_COMBINATIONS.keys()),
        help=f"Segment combination strategies (default: individual all near_zero forward)"
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of runs to process (for testing)"
    )

    args = ap.parse_args()

    data_root: Path = args.data_root
    output_dir: Path = args.output_dir
    voltage_col: str = args.voltage_col
    current_col: str = args.current_col
    orders: List[int] = args.orders
    combinations: List[str] = args.combinations
    limit: Optional[int] = args.limit

    # Validate orders are odd
    for order in orders:
        if order % 2 == 0:
            raise SystemExit(f"[error] All polynomial orders must be odd, got {order}")

    # Validate data root
    if not data_root.exists():
        raise SystemExit(f"[error] Data root does not exist: {data_root}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover run directories
    print(f"[info] Discovering segmented runs in {data_root}")
    print(f"[info] Combination strategies: {', '.join(combinations)}")
    run_dirs = discover_run_directories(data_root, limit=limit)
    print(f"[info] Found {len(run_dirs)} runs to process")

    if not run_dirs:
        print("[done] No runs to process")
        return

    # Process runs
    results = []
    success = 0
    failed = 0

    for i, run_dir in enumerate(run_dirs, 1):
        try:
            result = process_run(
                run_dir,
                voltage_col,
                current_col,
                orders,
                combinations
            )

            if result is None:
                failed += 1
                continue

            # Generate fitted curves
            try:
                segments = load_all_segments_for_run(run_dir, voltage_col, current_col)
                fitted_curves = generate_fitted_curves(result, segments, voltage_col) if segments else None
            except Exception as e:
                print(f"[warn] Failed to generate fitted curves for {result['run_id']}: {e}")
                fitted_curves = None

            # Save individual result to proc/date/run_id structure
            try:
                output_files = save_fit_parquet(result, output_dir, fitted_curves)
            except Exception as e:
                print(f"[warn] Failed to save parquet for {result['run_id']}: {e}")

            results.append(result)
            success += 1

            # Print progress
            run_id = result["run_id"][:16]
            n_combos = len(result["combinations"])
            return_to_zero_segs = result["return_to_zero_segment_ids"]

            # Find best fit across all combinations
            best_r2 = 0.0
            best_combo = ""
            for combo_key, combo_data in result["combinations"].items():
                for fit_key, metrics in combo_data["fits"].items():
                    if metrics["r_squared"] > best_r2:
                        best_r2 = metrics["r_squared"]
                        best_combo = combo_key

            print(
                f"[{i:04d}] OK {run_id} | {n_combos} combos | "
                f"return_segs={return_to_zero_segs} | best: {best_combo[:20]} R²={best_r2:.6f}"
            )

        except Exception as e:
            print(f"[{i:04d}] FAIL {run_dir} - {e}")
            failed += 1
            continue

    # Save results
    if results:
        # Save detailed JSON in _metadata
        metadata_dir = output_dir / "_metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)

        json_file = metadata_dir / "fit_results.json"
        save_fit_results(results, json_file)
        print(f"\n[info] Saved detailed results to {json_file}")

        # Save summary as Parquet in _metadata
        summary_df = create_summary_dataframe(results)
        parquet_file = metadata_dir / "fit_summary.parquet"
        summary_df.write_parquet(parquet_file)
        print(f"[info] Saved summary to {parquet_file}")

        # Print statistics
        print(f"\n[stats] Polynomial Fit Quality Summary:")
        for order in sorted(orders):
            order_data = summary_df.filter(pl.col("order") == order)
            if order_data.height > 0:
                mean_r2 = order_data["r_squared"].mean()
                median_r2 = order_data["r_squared"].median()
                print(f"  Order {order}: mean R² = {mean_r2:.6f}, median R² = {median_r2:.6f}")

    print(f"\n[done] Processing complete  |  success={success}  failed={failed}  total={len(run_dirs)}")


if __name__ == "__main__":
    main()
