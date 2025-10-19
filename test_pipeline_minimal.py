#!/usr/bin/env python3
"""
Minimal test to reproduce the pipeline freeze issue.

This simulates what the pipeline does but with maximum logging.
"""

import sys
import time
import gc
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "src" / "intermediate" / "IV"))
sys.path.insert(0, str(Path(__file__).parent / "src" / "staging"))

from iv_preprocessing_script import process_iv_run, discover_iv_runs
from stage_utils import ensure_dir

print("=" * 70)
print("SIMULATING PIPELINE FREEZE")
print("=" * 70)
print()

# Discover runs
stage_root = Path("data/02_stage/raw_measurements")
iv_runs = discover_iv_runs(stage_root)

print(f"Found {len(iv_runs)} IV runs")
print()

# Set up parameters
output_root = Path("data/03_intermediate/iv_segments")
voltage_col = "Vsd (V)"
dv_threshold = 0.001
min_points = 5
force = False
events_dir = Path("data/03_intermediate/_events")

ensure_dir(output_root)
ensure_dir(events_dir)

os.environ["POLARS_MAX_THREADS"] = "2"

# Test with just 10 tasks to see if issue persists
test_runs = iv_runs[:10]
print(f"Testing with first {len(test_runs)} runs")
print()

print("Creating ProcessPoolExecutor with 2 workers...")
with ProcessPoolExecutor(max_workers=2) as executor:
    print("✓ Executor created")
    print()

    print("Submitting tasks...")
    futures = {}
    for i, run_path in enumerate(test_runs):
        future = executor.submit(
            process_iv_run,
            run_path,
            output_root,
            voltage_col,
            dv_threshold,
            min_points,
            force,
            events_dir,
        )
        futures[future] = run_path
        print(f"  [{i+1}/{len(test_runs)}] Submitted: {run_path.name}")

    print()
    print(f"✓ All {len(test_runs)} tasks submitted")
    print()

    print("Waiting for results...")
    print("(If this hangs, workers are frozen)")
    print()

    ok = skipped = error = 0

    for i, future in enumerate(as_completed(futures), 1):
        run_path = futures[future]
        print(f"[{i}/{len(test_runs)}] Got result from as_completed(), calling .result()...")

        try:
            event = future.result(timeout=30)
            status = event.get("status")

            if status == "ok":
                ok += 1
                print(f"         → OK (segments={event.get('segments_detected')})")
            elif status == "skipped":
                skipped += 1
                print(f"         → SKIPPED (already exists)")
            else:
                error += 1
                print(f"         → ERROR: {event.get('error')}")

        except TimeoutError:
            error += 1
            print(f"         → TIMEOUT!")

        except Exception as e:
            error += 1
            print(f"         → EXCEPTION: {e}")

print()
print("=" * 70)
print("RESULTS")
print("=" * 70)
print(f"OK: {ok}")
print(f"Skipped: {skipped}")
print(f"Errors: {error}")
print()

if ok + skipped > 0:
    print("✓ Workers are functioning correctly!")
else:
    print("✗ All tasks failed/timed out")
