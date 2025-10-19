"""
Rich Logging Utilities

Provides Rich-based logging for beautiful terminal output.
"""

import logging
from rich.logging import RichHandler
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from typing import Dict, Any, Optional

# Import our custom console
from .console import console


def setup_rich_logging(level: str = "INFO") -> logging.Logger:
    """
    Set up Rich logging handler.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)

    Returns:
        Logger instance configured with Rich handler
    """
    # Create logger
    logger = logging.getLogger("nanolab")
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    logger.handlers.clear()

    # Add Rich handler
    rich_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        tracebacks_show_locals=True,
        markup=True,
        show_time=True,
        show_path=False,
    )
    rich_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(rich_handler)

    return logger


def create_summary_table(
    title: str,
    data: Dict[str, Any],
    show_header: bool = True
) -> Table:
    """
    Create a summary table from a dictionary.

    Args:
        title: Table title
        data: Dictionary of key-value pairs
        show_header: Whether to show column headers

    Returns:
        Rich Table instance
    """
    table = Table(title=title, show_header=show_header)

    if show_header:
        table.add_column("Metric", style="cyan", width=30)
        table.add_column("Value", style="green", justify="right")
    else:
        table.add_column("", style="cyan", width=30)
        table.add_column("", style="green", justify="right")

    for key, value in data.items():
        # Format key
        formatted_key = key.replace("_", " ").title()
        # Format value
        if isinstance(value, float):
            formatted_value = f"{value:.2f}"
        elif isinstance(value, int):
            formatted_value = f"{value:,}"
        else:
            formatted_value = str(value)

        table.add_row(formatted_key, formatted_value)

    return table


def create_results_table(
    files_processed: int,
    files_ok: int,
    files_skipped: int,
    files_rejected: int,
    elapsed_time: Optional[float] = None,
) -> Table:
    """
    Create a results summary table for staging operations.

    Args:
        files_processed: Total files processed
        files_ok: Successfully processed files
        files_skipped: Skipped files (already exist)
        files_rejected: Rejected files (errors)
        elapsed_time: Optional elapsed time in seconds

    Returns:
        Rich Table instance
    """
    table = Table(title="📊 Staging Results", show_header=False, box=None)
    table.add_column("Metric", style="cyan", width=25)
    table.add_column("Count", style="green", justify="right", width=15)
    table.add_column("Percentage", style="dim", justify="right", width=10)

    # Calculate percentages
    total = files_processed
    ok_pct = (files_ok / total * 100) if total > 0 else 0
    skip_pct = (files_skipped / total * 100) if total > 0 else 0
    reject_pct = (files_rejected / total * 100) if total > 0 else 0

    # Color code based on status
    ok_style = "bold green" if files_ok > 0 else "dim"
    skip_style = "yellow" if files_skipped > 0 else "dim"
    reject_style = "bold red" if files_rejected > 0 else "dim"

    table.add_row(
        "Total Files",
        f"[bold]{total:,}[/bold]",
        ""
    )
    table.add_row("", "", "")  # Spacer
    table.add_row(
        "✓ Processed Successfully",
        f"[{ok_style}]{files_ok:,}[/{ok_style}]",
        f"[dim]{ok_pct:.1f}%[/dim]"
    )
    table.add_row(
        "⊝ Skipped (exists)",
        f"[{skip_style}]{files_skipped:,}[/{skip_style}]",
        f"[dim]{skip_pct:.1f}%[/dim]"
    )
    table.add_row(
        "✗ Rejected (errors)",
        f"[{reject_style}]{files_rejected:,}[/{reject_style}]",
        f"[dim]{reject_pct:.1f}%[/dim]"
    )

    if elapsed_time is not None:
        table.add_row("", "", "")  # Spacer
        minutes = int(elapsed_time // 60)
        seconds = elapsed_time % 60
        if minutes > 0:
            time_str = f"{minutes}m {seconds:.1f}s"
        else:
            time_str = f"{seconds:.1f}s"
        table.add_row(
            "Elapsed Time",
            f"[bold magenta]{time_str}[/bold magenta]",
            ""
        )

        # Files per second
        if elapsed_time > 0:
            fps = files_processed / elapsed_time
            table.add_row(
                "Processing Speed",
                f"[dim]{fps:.1f} files/s[/dim]",
                ""
            )

    return table


def log_file_status(
    logger: logging.Logger,
    file_num: int,
    total_files: int,
    status: str,
    file_path: str,
    details: str = "",
    error: Optional[str] = None
):
    """
    Log the status of a single file processing operation.

    Args:
        logger: Logger instance
        file_num: Current file number
        total_files: Total number of files
        status: Status string (OK, SKIP, REJECT)
        file_path: Path to the file
        details: Additional details (procedure, rows, etc.)
        error: Error message if status is REJECT
    """
    # Format file number
    file_id = f"[{file_num:04d}/{total_files:04d}]"

    # Color code status
    if status.upper() == "OK":
        status_styled = f"[green]✓ {status}[/green]"
        logger.info(f"{file_id} {status_styled} {file_path} {details}")
    elif status.upper() == "SKIP":
        status_styled = f"[yellow]⊝ {status}[/yellow]"
        logger.info(f"{file_id} {status_styled} {file_path} {details}")
    elif status.upper() == "REJECT":
        status_styled = f"[red]✗ {status}[/red]"
        if error:
            logger.warning(f"{file_id} {status_styled} {file_path} :: [dim]{error}[/dim]")
        else:
            logger.warning(f"{file_id} {status_styled} {file_path}")
    else:
        logger.info(f"{file_id} {status} {file_path} {details}")
