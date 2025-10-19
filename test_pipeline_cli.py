#!/usr/bin/env python3
"""
Test the full pipeline CLI command with Rich visualization.

This demonstrates the complete pipeline UI without actually processing files.
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cli.utils.logging import setup_rich_logging
from cli.utils.staging_wrapper import create_results_table
from cli.utils.preprocessing_wrapper import create_preprocessing_results_table
from cli.utils.console import console, print_success, print_config
from cli.commands.pipeline import create_pipeline_overview, create_timeline_table, create_final_summary
from rich.progress import Progress, SpinnerColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, MofNCompleteColumn
from rich.panel import Panel
from rich.table import Table


def demo_pipeline():
    """Demonstrate the full pipeline UI."""

    # Pipeline overview
    console.print()
    console.print(create_pipeline_overview())
    console.print()

    # Configuration
    print_config({
        "raw_root": "data/01_raw",
        "stage_root": "data/02_stage/raw_measurements",
        "output_root": "data/03_intermediate",
        "procedure": "IV",
        "workers": 8,
        "polars_threads": 2,
        "force": False,
    })

    logger = setup_rich_logging("INFO")

    # ========== STAGE 1: STAGING ==========
    console.print()
    console.rule("[bold cyan]Stage 1: Raw CSV → Staged Parquet[/bold cyan]")
    console.print()

    stage1_start = time.time()

    logger.info("Scanning for CSV files in [path]data/01_raw[/path]...")
    time.sleep(0.3)
    logger.info(f"Found [bold]1,234[/bold] CSV files")
    console.print()

    # Stage 1 progress
    total_csvs = 100
    stage1_ok = 0
    stage1_skip = 0
    stage1_reject = 0

    progress = Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    )

    with progress:
        task = progress.add_task(
            "[cyan]Processing CSV files...",
            total=total_csvs
        )

        for i in range(1, total_csvs + 1):
            time.sleep(0.02)
            if i % 10 == 0:
                stage1_skip += 1
            elif i % 15 == 0:
                stage1_reject += 1
            else:
                stage1_ok += 1
            progress.update(task, advance=1)

    # Stage 1 results
    console.print()
    stage1_table = create_results_table(
        files_processed=total_csvs,
        files_ok=stage1_ok,
        files_skipped=stage1_skip,
        files_rejected=stage1_reject,
        elapsed_time=time.time() - stage1_start,
    )
    console.print(stage1_table)
    console.print()

    stage1_end = time.time()
    logger.info(f"[green]✓[/green] Stage 1 complete: {stage1_ok:,} files staged")
    console.print()

    # ========== STAGE 2: PREPROCESSING ==========
    console.print()
    console.rule("[bold cyan]Stage 2: Staged Parquet → Segments[/bold cyan]")
    console.print()

    stage2_start = time.time()

    logger.info("Scanning for IV runs in [path]data/02_stage/raw_measurements[/path]...")
    time.sleep(0.3)
    logger.info(f"Found [bold]48[/bold] IV runs")
    console.print()

    # Stage 2 progress
    total_runs = 48
    stage2_ok = 0
    stage2_skip = 0
    stage2_error = 0
    total_segments = 0

    with progress:
        task = progress.add_task(
            "[cyan]Processing IV runs...",
            total=total_runs
        )

        for i in range(1, total_runs + 1):
            time.sleep(0.03)
            if i % 8 == 0:
                stage2_skip += 1
                total_segments += 4
            elif i % 12 == 0:
                stage2_error += 1
            else:
                stage2_ok += 1
                total_segments += 4
            progress.update(task, advance=1)

    # Stage 2 results
    console.print()
    stage2_table = create_preprocessing_results_table(
        runs_processed=total_runs,
        runs_ok=stage2_ok,
        runs_skipped=stage2_skip,
        runs_error=stage2_error,
        total_segments=total_segments,
        elapsed_time=time.time() - stage2_start,
    )
    console.print(stage2_table)
    console.print()

    stage2_end = time.time()
    logger.info(
        f"[green]✓[/green] Stage 2 complete: "
        f"{stage2_ok:,} runs processed, "
        f"{total_segments:,} segments created"
    )
    console.print()

    # ========== FINAL SUMMARY ==========
    console.print()
    console.rule("[bold green]Pipeline Complete[/bold green]")
    console.print()

    # Timeline
    timeline = create_timeline_table(
        stage1_start, stage1_end,
        stage2_start, stage2_end,
        total_csvs, total_runs
    )
    console.print(timeline)
    console.print()

    # Summary
    staging_results = {
        "ok": stage1_ok,
        "skipped": stage1_skip,
        "rejected": stage1_reject,
    }
    preprocessing_results = {
        "ok": stage2_ok,
        "skipped": stage2_skip,
        "error": stage2_error,
        "total_segments": total_segments,
    }
    summary = create_final_summary(staging_results, preprocessing_results)
    console.print(summary)
    console.print()

    # Success
    print_success(
        "🎉 Pipeline completed successfully!",
        f"Staged: [path]data/02_stage/raw_measurements[/path]\n"
        f"Segmented: [path]data/03_intermediate/iv_segments[/path]\n"
        f"Total segments created: {total_segments:,}"
    )


if __name__ == "__main__":
    demo_pipeline()
