#!/usr/bin/env python3
"""
Test the enhanced preprocess CLI command with Rich progress bars.

This demonstrates the new Rich UI without actually processing files.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cli.utils.logging import setup_rich_logging
from cli.utils.preprocessing_wrapper import create_preprocessing_results_table
from cli.utils.console import console, print_step, print_success, print_config
from rich.progress import Progress, SpinnerColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, MofNCompleteColumn
import time


def demo_preprocessing():
    """Demonstrate the preprocessing UI."""

    # Step 1: Show configuration
    print_step(
        2,
        "PREPROCESSING",
        "Detecting and segmenting voltage sweeps"
    )

    print_config({
        "stage_root": "data/02_stage/raw_measurements",
        "output_dir": "data/03_intermediate/iv_segments",
        "procedure": "IV",
        "voltage_col": "Vsd (V)",
        "dv_threshold": 0.001,
        "min_segment_points": 5,
        "workers": 8,
        "polars_threads": 2,
        "force": False,
    })

    # Step 2: Set up logging
    logger = setup_rich_logging("INFO")
    logger.info("[bold cyan]Starting preprocessing pipeline...[/bold cyan]")
    console.print()

    # Step 3: Show file discovery
    logger.info("Scanning for IV runs in [path]data/02_stage/raw_measurements[/path]...")
    time.sleep(0.5)
    logger.info(f"Found [bold]48[/bold] IV runs")
    console.print()

    # Step 4: Show progress bar with demo data (minimal - no per-run logging)
    total_runs = 48
    ok_count = 0
    skip_count = 0
    error_count = 0
    total_segments = 0

    progress = Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    )

    start_time = time.time()

    with progress:
        task = progress.add_task(
            "[cyan]Processing IV runs...",
            total=total_runs
        )

        for i in range(1, total_runs + 1):
            # Simulate processing
            time.sleep(0.05)

            # Randomly determine status (just count, don't log)
            if i % 8 == 0:
                skip_count += 1
                total_segments += 4  # Assume 4 segments per run
            elif i % 15 == 0:
                error_count += 1
            else:
                ok_count += 1
                total_segments += 4  # Assume 4 segments per run

            # Update progress bar only
            progress.update(task, advance=1)

    elapsed_time = time.time() - start_time

    # Step 5: Show results table
    console.print()
    results_table = create_preprocessing_results_table(
        runs_processed=total_runs,
        runs_ok=ok_count,
        runs_skipped=skip_count,
        runs_error=error_count,
        total_segments=total_segments,
        elapsed_time=elapsed_time,
    )
    console.print(results_table)
    console.print()

    # Step 6: Success message
    print_success(
        "Preprocessing completed successfully!",
        f"Segmented data saved to: [path]data/03_intermediate/iv_segments[/path]\n"
        f"Processed: {ok_count:,} runs\n"
        f"Total segments: {total_segments:,}"
    )


if __name__ == "__main__":
    demo_preprocessing()
