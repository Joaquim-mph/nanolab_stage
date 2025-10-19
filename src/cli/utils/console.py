"""
Rich Console Singleton

Provides a globally accessible Rich console instance with custom theme
for consistent terminal output throughout the CLI.
"""

from rich.console import Console
from rich.theme import Theme

# Custom theme for nanolab pipeline
custom_theme = Theme({
    "info": "cyan",
    "success": "bold green",
    "warning": "bold yellow",
    "error": "bold red",
    "step": "bold blue",
    "highlight": "bold magenta",
    "path": "italic cyan",
    "value": "green",
    "key": "cyan",
    "dim": "dim",
})

# Global console instance
console = Console(theme=custom_theme)


def print_step(step_num: int, title: str, description: str = ""):
    """
    Print a formatted step header.

    Args:
        step_num: Step number
        title: Step title
        description: Optional step description
    """
    from rich.panel import Panel

    content = f"[bold cyan]STEP {step_num}: {title.upper()}[/bold cyan]"
    if description:
        content += f"\n{description}"

    console.print()
    console.print(Panel.fit(content, border_style="cyan"))


def print_success(message: str, details: str = ""):
    """
    Print a success message in a panel.

    Args:
        message: Success message
        details: Optional additional details
    """
    from rich.panel import Panel

    content = message
    if details:
        content += f"\n\n{details}"

    console.print(Panel(
        content,
        title="[bold green]Success[/bold green]",
        border_style="green"
    ))


def print_error(message: str, solution: str = ""):
    """
    Print an error message in a panel.

    Args:
        message: Error message
        solution: Optional suggested solution
    """
    from rich.panel import Panel

    content = message
    if solution:
        content += f"\n\n[bold]Solution:[/bold]\n{solution}"

    console.print(Panel(
        content,
        title="[bold red]Error[/bold red]",
        border_style="red"
    ))


def print_warning(message: str):
    """
    Print a warning message.

    Args:
        message: Warning message
    """
    console.print(f"[warning]{message}[/warning]")


def print_config(config_dict: dict, title: str = "Configuration"):
    """
    Print a configuration dictionary as a formatted table.

    Args:
        config_dict: Dictionary of configuration parameters
        title: Table title
    """
    from rich.table import Table

    table = Table(title=title, show_header=False, box=None)
    table.add_column("Parameter", style="key", width=25)
    table.add_column("Value", style="value")

    for key, value in config_dict.items():
        # Format the key (convert snake_case to Title Case)
        formatted_key = key.replace("_", " ").title()
        # Format the value
        formatted_value = str(value)
        table.add_row(formatted_key, formatted_value)

    console.print()
    console.print(table)
    console.print()
