#!/usr/bin/env python3
"""
Analyze coercive field from backward-subtracted hysteresis data.

The coercive field is defined as the E-field where maximum backward-subtracted
hysteresis current occurs, analyzed separately for positive and negative fields.

Incorporates laser power calibration:
- Loads laser calibration curve (VL vs Power)
- Interpolates power for each IV measurement's laser voltage
- Computes effective power on sample: P_effective = P_calibrated / 5.7 (beam-to-sample ratio)

Device: 100 μm × 50 μm
E-field: E = V / (100 μm) = V × 100 V/cm
Current density: J = I / (5×10⁻⁵ cm²)
"""

import argparse
import polars as pl
from pathlib import Path
import numpy as np
from typing import Optional
import sys


# Device dimensions
CHANNEL_LENGTH_UM = 100.0  # micrometers
CHANNEL_WIDTH_UM = 50.0    # micrometers
CHANNEL_LENGTH_CM = CHANNEL_LENGTH_UM * 1e-4  # convert to cm
CHANNEL_AREA_CM2 = (CHANNEL_LENGTH_UM * CHANNEL_WIDTH_UM) * 1e-8  # convert μm² to cm²

# Laser beam to sample ratio
BEAM_TO_SAMPLE_RATIO = 5.7


def load_laser_calibration(date: str, stage_root: Path, fallback_date: Optional[str] = None) -> Optional[pl.DataFrame]:
    """
    Load laser calibration data for a given date.

    If calibration for the specified date is not found, tries fallback_date.

    Returns DataFrame with columns: VL (V), Power (W)
    """
    cal_dir = stage_root / f"proc=LaserCalibration/date={date}"

    if not cal_dir.exists():
        if fallback_date:
            print(f"  Warning: No laser calibration found for {date}, trying fallback {fallback_date}")
            cal_dir = stage_root / f"proc=LaserCalibration/date={fallback_date}"
            if not cal_dir.exists():
                print(f"  Warning: Fallback calibration also not found for {fallback_date}")
                return None
        else:
            print(f"  Warning: No laser calibration found for {date}")
            return None

    # Find parquet file
    parquet_files = list(cal_dir.rglob("*.parquet"))
    if not parquet_files:
        print(f"  Warning: No parquet files in {cal_dir}")
        return None

    # Read first calibration file
    df = pl.read_parquet(parquet_files[0])

    # Check required columns
    if "VL (V)" not in df.columns or "Power (W)" not in df.columns:
        print(f"  Warning: Missing required columns in calibration file")
        return None

    # Sort by voltage for interpolation
    df = df.sort("VL (V)")

    print(f"  Loaded laser calibration: VL range {df['VL (V)'].min():.2f}-{df['VL (V)'].max():.2f} V")
    print(f"  Power range: {df['Power (W)'].min():.2e}-{df['Power (W)'].max():.2e} W")

    return df.select(["VL (V)", "Power (W)"])


def interpolate_laser_power(laser_voltage: float, calibration: pl.DataFrame) -> float:
    """
    Interpolate laser power from calibration curve.

    Returns power in Watts, or 0.0 if calibration is None or voltage out of range.
    """
    if calibration is None or laser_voltage == 0.0:
        return 0.0

    vl_array = calibration["VL (V)"].to_numpy()
    power_array = calibration["Power (W)"].to_numpy()

    # Linear interpolation (extrapolate if needed)
    power = np.interp(laser_voltage, vl_array, power_array)

    return float(power)


def get_laser_voltage_from_iv_data(date: str, vmax: float, stage_root: Path) -> float:
    """
    Get laser voltage from IV measurement metadata.

    Returns laser_voltage_V parameter from IV data for this specific voltage range.
    """
    iv_dir = stage_root / f"proc=IV/date={date}"

    if not iv_dir.exists():
        return 0.0

    # Find all IV files for this date
    parquet_files = list(iv_dir.rglob("*.parquet"))
    if not parquet_files:
        return 0.0

    # Read all files and check for measurements with this voltage range
    # Look for files where max(abs(Vsd)) ≈ vmax
    for pfile in parquet_files:
        try:
            df = pl.read_parquet(pfile)

            if "Vsd (V)" in df.columns and "laser_voltage_V" in df.columns:
                # Check if this file contains measurements for this voltage range
                max_v = df["Vsd (V)"].abs().max()
                if abs(max_v - vmax) < 0.5:  # Within 0.5V tolerance
                    return float(df["laser_voltage_V"][0])
        except Exception:
            continue

    # If no exact match, try to get any laser voltage from this date
    try:
        df = pl.read_parquet(parquet_files[0])
        if "laser_voltage_V" in df.columns:
            return float(df["laser_voltage_V"][0])
    except Exception:
        pass

    return 0.0


def find_coercive_field(
    backward_sub_file: Path,
    poly_order: int
) -> dict:
    """
    Find coercive field (E-field at max |J_hysteresis|) for positive and negative fields.

    Returns dict with:
        - E_coercive_pos: Positive coercive field (V/cm)
        - E_coercive_neg: Negative coercive field (V/cm)
        - J_max_pos: Max current density at positive coercive field (A/cm²)
        - J_max_neg: Max current density at negative coercive field (A/cm²)
        - hysteresis_type: 'forward_sub' or 'backward_sub'
    """
    if not backward_sub_file.exists():
        return {
            "E_coercive_pos": None,
            "E_coercive_neg": None,
            "J_max_pos": None,
            "J_max_neg": None,
            "hysteresis_type": None
        }

    df = pl.read_parquet(backward_sub_file) if backward_sub_file.suffix == ".parquet" else pl.read_csv(backward_sub_file)

    # Column names for this polynomial order
    j_fwd_col = f"J_forward_sub_poly{poly_order} (A/cm2)"
    j_bwd_col = f"J_backward_sub_poly{poly_order} (A/cm2)"

    if j_fwd_col not in df.columns or j_bwd_col not in df.columns:
        print(f"  Warning: Missing columns {j_fwd_col} or {j_bwd_col}")
        return {
            "E_coercive_pos": None,
            "E_coercive_neg": None,
            "J_max_pos": None,
            "J_max_neg": None,
            "hysteresis_type": None
        }

    # Compute hysteresis magnitude: |J_forward - J_backward|
    df = df.with_columns([
        (pl.col(j_fwd_col) - pl.col(j_bwd_col)).abs().alias("J_hyst_abs")
    ])

    # Split into positive and negative fields
    df_pos = df.filter(pl.col("E (V/cm)") >= 0)
    df_neg = df.filter(pl.col("E (V/cm)") < 0)

    result = {}

    # Positive field
    if len(df_pos) > 0:
        idx_max_pos = df_pos["J_hyst_abs"].arg_max()
        result["E_coercive_pos"] = float(df_pos["E (V/cm)"][idx_max_pos])
        result["J_max_pos"] = float(df_pos["J_hyst_abs"][idx_max_pos])
    else:
        result["E_coercive_pos"] = None
        result["J_max_pos"] = None

    # Negative field
    if len(df_neg) > 0:
        idx_max_neg = df_neg["J_hyst_abs"].arg_max()
        result["E_coercive_neg"] = float(df_neg["E (V/cm)"][idx_max_neg])
        result["J_max_neg"] = float(df_neg["J_hyst_abs"][idx_max_neg])
    else:
        result["E_coercive_neg"] = None
        result["J_max_neg"] = None

    result["hysteresis_type"] = "backward_sub"

    return result


def analyze_coercive_field_per_date(
    date: str,
    backward_sub_dir: Path,
    stage_root: Path,
    voltage_ranges: list[float],
    poly_orders: list[int],
    calibration_fallback: Optional[str] = None
) -> pl.DataFrame:
    """
    Analyze coercive field for all voltage ranges and polynomial orders for a given date.

    Returns DataFrame with columns:
        - date
        - V_max
        - E_max
        - poly_order
        - E_coercive_pos
        - E_coercive_neg
        - J_max_pos
        - J_max_neg
        - laser_voltage_V
        - incident_power_W (calibrated power)
        - effective_power_W (power on sample = incident / 5.7)
    """
    print(f"\nAnalyzing {date}...")

    # Load laser calibration
    calibration = load_laser_calibration(date, stage_root, fallback_date=calibration_fallback)

    results = []

    for vmax in voltage_ranges:
        vmax_str = str(vmax).replace(".", "p")
        backward_file = backward_sub_dir / date / f"backward_sub_vmax{vmax_str}V.csv"

        if not backward_file.exists():
            print(f"  Skipping V_max={vmax}V (file not found)")
            continue

        # Get laser voltage for this measurement
        laser_voltage = get_laser_voltage_from_iv_data(date, vmax, stage_root)

        # Interpolate incident power
        incident_power = interpolate_laser_power(laser_voltage, calibration)

        # Compute effective power on sample
        effective_power = incident_power / BEAM_TO_SAMPLE_RATIO if incident_power > 0 else 0.0

        for order in poly_orders:
            # Find coercive fields
            coercive_data = find_coercive_field(backward_file, order)

            results.append({
                "date": date,
                "V_max": vmax,
                "E_max": vmax * 100.0,  # Convert to V/cm
                "poly_order": order,
                "E_coercive_pos": coercive_data["E_coercive_pos"],
                "E_coercive_neg": coercive_data["E_coercive_neg"],
                "J_max_pos": coercive_data["J_max_pos"],
                "J_max_neg": coercive_data["J_max_neg"],
                "laser_voltage_V": laser_voltage,
                "incident_power_W": incident_power,
                "effective_power_W": effective_power
            })

            if coercive_data["E_coercive_pos"] is not None:
                print(f"  V_max={vmax}V, poly={order}: E_c+ = {coercive_data['E_coercive_pos']:.1f} V/cm, "
                      f"J_max+ = {coercive_data['J_max_pos']:.2e} A/cm²")
            if coercive_data["E_coercive_neg"] is not None:
                print(f"  V_max={vmax}V, poly={order}: E_c- = {coercive_data['E_coercive_neg']:.1f} V/cm, "
                      f"J_max- = {coercive_data['J_max_neg']:.2e} A/cm²")

    return pl.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze coercive field from backward-subtracted hysteresis with laser power"
    )
    parser.add_argument(
        "--dates",
        nargs="+",
        required=True,
        help="Dates to analyze (e.g., 2025-09-11 2025-09-29 2025-09-30)"
    )
    parser.add_argument(
        "--backward-sub-root",
        type=str,
        default="data/04_analysis/backward_substracted_efield_sept",
        help="Root directory for backward-subtracted data"
    )
    parser.add_argument(
        "--stage-root",
        type=str,
        default="data/02_stage/raw_measurements",
        help="Root directory for staged raw measurements (for laser calibration)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/04_analysis/coercive_field",
        help="Output directory for coercive field analysis"
    )
    parser.add_argument(
        "--voltage-ranges",
        type=float,
        nargs="+",
        default=[3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        help="Voltage ranges to analyze"
    )
    parser.add_argument(
        "--poly-orders",
        type=int,
        nargs="+",
        default=[1, 3, 5, 7],
        help="Polynomial orders to analyze"
    )

    args = parser.parse_args()

    backward_sub_root = Path(args.backward_sub_root)
    stage_root = Path(args.stage_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("COERCIVE FIELD ANALYSIS WITH LASER POWER")
    print("=" * 80)
    print(f"\nDevice dimensions: {CHANNEL_LENGTH_UM} μm × {CHANNEL_WIDTH_UM} μm")
    print(f"Beam-to-sample ratio: {BEAM_TO_SAMPLE_RATIO}")
    print(f"\nDates: {', '.join(args.dates)}")
    print(f"Voltage ranges: {args.voltage_ranges}")
    print(f"Polynomial orders: {args.poly_orders}")

    # Analyze each date
    all_results = []

    # Define calibration fallbacks (if date has no calibration, use this date's calibration)
    calibration_fallbacks = {
        "2025-09-30": "2025-09-29",  # Use Sept 29 calibration for Sept 30
    }

    for date in args.dates:
        fallback = calibration_fallbacks.get(date, None)
        df_date = analyze_coercive_field_per_date(
            date,
            backward_sub_root,
            stage_root,
            args.voltage_ranges,
            args.poly_orders,
            calibration_fallback=fallback
        )
        all_results.append(df_date)

    # Combine all results
    if all_results:
        df_all = pl.concat(all_results)

        # Save to CSV
        output_file = output_dir / "coercive_field_analysis.csv"
        df_all.write_csv(output_file)
        print(f"\n✓ Saved: {output_file}")

        # Print summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(df_all.select([
            "date", "V_max", "poly_order", "E_coercive_pos", "E_coercive_neg",
            "effective_power_W"
        ]))

        # Save per-date summaries
        for date in args.dates:
            df_date = df_all.filter(pl.col("date") == date)
            output_file_date = output_dir / f"coercive_field_{date}.csv"
            df_date.write_csv(output_file_date)
            print(f"\n✓ Saved: {output_file_date}")

    print("\n" + "=" * 80)
    print("✓ COERCIVE FIELD ANALYSIS COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
