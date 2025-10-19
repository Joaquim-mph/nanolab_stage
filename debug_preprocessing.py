#!/usr/bin/env python3
"""
Debug script to test preprocessing worker function in isolation.

This helps diagnose why workers freeze when called from the pipeline.
"""

import sys
from pathlib import Path

# Add paths (same as wrapper does)
sys.path.insert(0, str(Path(__file__).parent / "src" / "intermediate" / "IV"))
sys.path.insert(0, str(Path(__file__).parent / "src" / "staging"))

from iv_preprocessing_script import process_iv_run, discover_iv_runs
from stage_utils import ensure_dir

def test_single_run():
    """Test processing a single IV run to see if it works."""

    # Discover runs
    stage_root = Path("data/02_stage/raw_measurements")
    iv_runs = discover_iv_runs(stage_root)

    if not iv_runs:
        print("ERROR: No IV runs found!")
        return

    print(f"Found {len(iv_runs)} IV runs")
    print(f"Testing with first run: {iv_runs[0]}")
    print()

    # Set up parameters (same as pipeline uses)
    output_root = Path("data/03_intermediate/iv_segments")
    voltage_col = "Vsd (V)"
    dv_threshold = 0.001
    min_points = 5
    force = False
    events_dir = Path("data/03_intermediate/_events")

    # Create directories
    ensure_dir(output_root)
    ensure_dir(events_dir)

    print("=" * 60)
    print("CALLING process_iv_run()...")
    print("=" * 60)

    try:
        event = process_iv_run(
            iv_runs[0],
            output_root,
            voltage_col,
            dv_threshold,
            min_points,
            force,
            events_dir,
        )

        print()
        print("=" * 60)
        print("SUCCESS!")
        print("=" * 60)
        print(f"Status: {event.get('status')}")
        print(f"Run ID: {event.get('run_id')}")
        print(f"Segments detected: {event.get('segments_detected')}")
        print(f"Segments written: {event.get('segments_written')}")

    except Exception as e:
        print()
        print("=" * 60)
        print("ERROR!")
        print("=" * 60)
        print(f"Exception: {e}")
        import traceback
        traceback.print_exc()


def test_multiprocess():
    """Test with actual ProcessPoolExecutor (1 worker, 1 task)."""
    from concurrent.futures import ProcessPoolExecutor
    import os

    # Discover runs
    stage_root = Path("data/02_stage/raw_measurements")
    iv_runs = discover_iv_runs(stage_root)

    if not iv_runs:
        print("ERROR: No IV runs found!")
        return

    print(f"Found {len(iv_runs)} IV runs")
    print(f"Testing with first run using ProcessPoolExecutor: {iv_runs[0]}")
    print()

    # Set up parameters
    output_root = Path("data/03_intermediate/iv_segments")
    voltage_col = "Vsd (V)"
    dv_threshold = 0.001
    min_points = 5
    force = False
    events_dir = Path("data/03_intermediate/_events")

    # Create directories
    ensure_dir(output_root)
    ensure_dir(events_dir)

    # Set Polars threads
    os.environ["POLARS_MAX_THREADS"] = "2"

    print("=" * 60)
    print("CREATING ProcessPoolExecutor with 1 worker...")
    print("=" * 60)

    with ProcessPoolExecutor(max_workers=1) as executor:
        print("Executor created successfully")
        print("Submitting task...")

        future = executor.submit(
            process_iv_run,
            iv_runs[0],
            output_root,
            voltage_col,
            dv_threshold,
            min_points,
            force,
            events_dir,
        )

        print("Task submitted, waiting for result...")
        print("(If this hangs, the problem is with multiprocessing/pickle)")
        print()

        try:
            # Add timeout to prevent infinite hang
            event = future.result(timeout=30)

            print()
            print("=" * 60)
            print("SUCCESS!")
            print("=" * 60)
            print(f"Status: {event.get('status')}")
            print(f"Run ID: {event.get('run_id')}")
            print(f"Segments detected: {event.get('segments_detected')}")

        except TimeoutError:
            print()
            print("=" * 60)
            print("TIMEOUT!")
            print("=" * 60)
            print("Worker process is frozen/hanging")
            print("This indicates a multiprocessing issue")

        except Exception as e:
            print()
            print("=" * 60)
            print("ERROR!")
            print("=" * 60)
            print(f"Exception: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Debug preprocessing worker function")
    parser.add_argument(
        "--mode",
        choices=["single", "multiprocess"],
        default="single",
        help="Test mode: single (direct call) or multiprocess (via executor)"
    )

    args = parser.parse_args()

    if args.mode == "single":
        print("Testing direct function call (no multiprocessing)...")
        print()
        test_single_run()
    else:
        print("Testing via ProcessPoolExecutor (with multiprocessing)...")
        print()
        test_multiprocess()
