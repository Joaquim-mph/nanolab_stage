#!/usr/bin/env python3
"""
Test the enhanced stage CLI command with Rich progress bars.

This demonstrates the new Rich UI without actually processing files.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cli.utils.logging import (
    setup_rich_logging,
    create_results_table,
)
from cli.utils.console import console, print_step, print_success, print_config
from rich.progress import Progress, SpinnerColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, MofNCompleteColumn
import time


def demo_staging():
    """Demonstrate the staging UI."""

    # Step 1: Show configuration
    print_step(
        1,
        "STAGING",
        "Converting raw CSV files to Parquet format"
    )

    print_config({
        "raw_root": "data/01_raw",
        "stage_root": "data/02_stage/raw_measurements",
        "workers": 8,
        "polars_threads": 2,
        "force": False,
    })

    # Step 2: Set up logging
    logger = setup_rich_logging("INFO")
    logger.info("[bold cyan]Starting staging pipeline...[/bold cyan]")
    console.print()

    # Step 3: Show file discovery
    logger.info("Scanning for CSV files in [path]data/01_raw[/path]...")
    time.sleep(0.5)
    logger.info(f"Found [bold]1,234[/bold] CSV files")
    console.print()

    # Step 4: Show progress bar with demo data (minimal - no per-file logging)
    total_files = 50
    ok_count = 0
    skip_count = 0
    reject_count = 0

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
            "[cyan]Processing CSV files...",
            total=total_files
        )

        for i in range(1, total_files + 1):
            # Simulate processing
            time.sleep(0.05)

            # Randomly determine status (just count, don't log)
            if i % 10 == 0:
                skip_count += 1
            elif i % 13 == 0:
                reject_count += 1
            else:
                ok_count += 1

            # Update progress bar only
            progress.update(task, advance=1)

    elapsed_time = time.time() - start_time

    # Step 5: Show results table
    console.print()
    results_table = create_results_table(
        files_processed=total_files,
        files_ok=ok_count,
        files_skipped=skip_count,
        files_rejected=reject_count,
        elapsed_time=elapsed_time,
    )
    console.print(results_table)
    console.print()

    # Step 6: Success message
    print_success(
        "Staging completed successfully!",
        f"Staged data saved to: [path]data/02_stage/raw_measurements[/path]\n"
        f"Processed: {ok_count:,} files"
    )


if __name__ == "__main__":
    demo_staging()
