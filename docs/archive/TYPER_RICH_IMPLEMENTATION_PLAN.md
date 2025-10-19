# Typer + Rich Implementation Plan

**Date:** 2025-10-19
**Goal:** Modernize CLI with Typer + Rich for better UX
**Status:** Planning Phase

---

## 1. Current State Analysis

### Existing CLI Tools
- **Main entry point:** `run_pipeline.py` (uses `argparse`)
- **Staging:** `src/staging/stage_raw_measurements.py` (uses `argparse`)
- **Intermediate:** `src/intermediate/IV/iv_preprocessing_script.py` (uses `argparse`)
- **Analysis:** `src/analysis/IV/aggregate_iv_stats.py` (uses `argparse`)
- **Hysteresis:** `src/analysis/IV/compute_hysteresis.py` (uses `argparse`)
- **Peaks:** `src/analysis/IV/analyze_hysteresis_peaks.py` (uses `argparse`)

### Current Output Style
- Plain `print()` statements
- Basic separators (`===`, `---`)
- Manual formatting
- Simple error messages
- No progress bars
- No color coding
- No interactive elements

---

## 2. Typer + Rich Benefits

### What We'll Gain

**Typer:**
- ✅ Automatic help generation with better formatting
- ✅ Type hints → automatic validation
- ✅ Subcommands support (e.g., `pipeline stage`, `pipeline analyze`)
- ✅ Auto-completion support
- ✅ Better error messages
- ✅ Option groups and required/optional clarity

**Rich:**
- ✅ Beautiful progress bars with live updates
- ✅ Color-coded output (success=green, error=red, warning=yellow)
- ✅ Formatted tables for summaries
- ✅ Panels for important information
- ✅ Syntax highlighting for code/JSON
- ✅ Live status updates
- ✅ Spinners for long operations
- ✅ Tree views for file structures

---

## 3. Architecture Design

### Option A: Unified CLI with Subcommands (Recommended)

```bash
# Main command with subcommands
nanolab-pipeline <subcommand> [options]

# Subcommands:
nanolab-pipeline stage [options]          # Staging only
nanolab-pipeline preprocess [options]     # Intermediate preprocessing
nanolab-pipeline analyze [options]        # Analysis only
nanolab-pipeline hysteresis [options]     # Hysteresis computation
nanolab-pipeline peaks [options]          # Peak analysis
nanolab-pipeline plot [options]           # Plotting
nanolab-pipeline run [options]            # Full pipeline (all steps)
nanolab-pipeline status                   # Show pipeline status
nanolab-pipeline info                     # Show configuration/paths
```

**Pros:**
- Single entry point
- Clear organization
- Professional CLI experience
- Easy to discover features
- Consistent interface

**Cons:**
- More initial work
- Requires restructuring

### Option B: Enhanced Individual Scripts

Keep current script structure but enhance each with Typer + Rich:
```bash
python src/staging/stage_raw_measurements.py [options]
python src/intermediate/IV/iv_preprocessing_script.py [options]
python src/analysis/IV/aggregate_iv_stats.py [options]
python run_pipeline.py [options]  # Enhanced with Rich
```

**Pros:**
- Minimal disruption
- Gradual migration
- Backward compatible

**Cons:**
- Less cohesive
- Multiple entry points
- Harder to maintain consistency

### Recommendation: **Option A** (Unified CLI)

Create `nanolab_pipeline/cli.py` as main entry point with subcommands.

---

## 4. Implementation Roadmap

### Phase 1: Core Infrastructure (Week 1)
**Goal:** Set up Typer app structure and Rich console

**Tasks:**
1. Create `src/cli/` directory structure:
   ```
   src/cli/
   ├── __init__.py
   ├── main.py           # Main Typer app
   ├── commands/
   │   ├── __init__.py
   │   ├── stage.py      # Staging command
   │   ├── preprocess.py # Intermediate command
   │   ├── analyze.py    # Analysis command
   │   ├── hysteresis.py # Hysteresis command
   │   ├── peaks.py      # Peaks command
   │   ├── plot.py       # Plotting command
   │   └── run.py        # Full pipeline command
   ├── utils/
   │   ├── __init__.py
   │   ├── console.py    # Rich console singleton
   │   ├── progress.py   # Progress bar helpers
   │   └── formatters.py # Output formatting utilities
   └── config.py         # CLI configuration
   ```

2. Create Rich console singleton:
   ```python
   # src/cli/utils/console.py
   from rich.console import Console
   from rich.theme import Theme

   custom_theme = Theme({
       "info": "cyan",
       "success": "bold green",
       "warning": "bold yellow",
       "error": "bold red",
       "step": "bold blue",
   })

   console = Console(theme=custom_theme)
   ```

3. Create main Typer app:
   ```python
   # src/cli/main.py
   import typer
   from .commands import stage, preprocess, analyze, ...

   app = typer.Typer(
       name="nanolab-pipeline",
       help="Nanolab data processing pipeline",
       add_completion=True,
   )

   app.add_typer(stage.app, name="stage")
   app.add_typer(preprocess.app, name="preprocess")
   ...
   ```

4. Create entry point script:
   ```python
   # nanolab-pipeline.py (root directory)
   from src.cli.main import app

   if __name__ == "__main__":
       app()
   ```

**Deliverable:** Working CLI skeleton with `--help` output

---

### Phase 2: Staging Command (Week 1-2)
**Goal:** Migrate staging to Typer + Rich

**Tasks:**
1. Create `src/cli/commands/stage.py`:
   ```python
   import typer
   from pathlib import Path
   from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
   from ..utils.console import console
   from ...staging.stage_raw_measurements import run_staging_pipeline

   app = typer.Typer(help="Stage raw CSV files to Parquet")

   @app.command()
   def stage(
       raw_root: Path = typer.Option(..., help="Raw data directory"),
       stage_root: Path = typer.Option(..., help="Staging output directory"),
       procedures_yaml: Path = typer.Option("config/procedures.yml"),
       workers: int = typer.Option(8, help="Number of parallel workers"),
       force: bool = typer.Option(False, help="Overwrite existing files"),
   ):
       """Stage raw CSV measurements to Parquet format."""
       # Rich panel for configuration
       # Progress bars for processing
       # Color-coded success/error messages
   ```

2. Add Rich enhancements:
   - **Panel** for configuration summary
   - **Progress bar** for file processing
   - **Table** for final statistics (files processed, rejected, etc.)
   - **Tree view** for output directory structure

3. Integration points:
   - Wrap existing `run_staging_pipeline()` with Rich progress
   - Add callback for progress updates
   - Color-code log messages

**Deliverable:** `nanolab-pipeline stage` working with beautiful output

---

### Phase 3: Preprocessing Command (Week 2)
**Goal:** Migrate intermediate preprocessing to Typer + Rich

**Tasks:**
1. Create `src/cli/commands/preprocess.py`
2. Add Rich features:
   - **Live progress** for segment detection
   - **Multi-progress bars** (one per worker)
   - **Stats table** showing segments created per run
   - **Status panel** with running totals

3. Progress tracking:
   ```python
   with Progress(
       SpinnerColumn(),
       *Progress.get_default_columns(),
       TimeElapsedColumn(),
   ) as progress:
       task = progress.add_task("[cyan]Preprocessing runs...", total=total_runs)
       # Update progress as runs complete
   ```

**Deliverable:** `nanolab-pipeline preprocess` with live progress

---

### Phase 4: Analysis Commands (Week 2-3)
**Goal:** Migrate analysis scripts to Typer + Rich

**Tasks:**
1. Create commands:
   - `src/cli/commands/analyze.py` - Main IV analysis
   - `src/cli/commands/hysteresis.py` - Hysteresis computation
   - `src/cli/commands/peaks.py` - Peak analysis

2. Add Rich features:
   - **Progress bars** for polynomial fitting
   - **Tables** for fit quality (R² values)
   - **Panels** for configuration and results
   - **Status messages** with timestamps

3. Result presentation:
   ```python
   from rich.table import Table

   table = Table(title="Polynomial Fit Results")
   table.add_column("Order", style="cyan")
   table.add_column("R²", style="green")
   table.add_column("RMSE", style="yellow")

   for order, r2, rmse in results:
       table.add_row(str(order), f"{r2:.6f}", f"{rmse:.2e}")

   console.print(table)
   ```

**Deliverable:** All analysis commands with beautiful output

---

### Phase 5: Full Pipeline Command (Week 3)
**Goal:** Create unified `run` command for complete pipeline

**Tasks:**
1. Create `src/cli/commands/run.py`:
   ```python
   @app.command()
   def run(
       config: Path = typer.Option(None, help="Config file"),
       date: str = typer.Option(None, help="Analysis date"),
       # ... other options
   ):
       """Run the complete 4-layer pipeline."""
       # Show pipeline stages
       # Execute each step with Rich progress
       # Summary at end
   ```

2. Add Rich features:
   - **Panel** showing pipeline overview
   - **Step-by-step progress** with nested progress bars
   - **Summary table** at end
   - **Timeline visualization**

3. Pipeline visualization:
   ```python
   from rich.panel import Panel
   from rich.tree import Tree

   tree = Tree("🚀 Pipeline Execution")
   tree.add("[green]✓[/green] Staging (completed)")
   tree.add("[yellow]⚙[/yellow] Preprocessing (running...)")
   tree.add("[dim]○[/dim] Analysis (pending)")
   tree.add("[dim]○[/dim] Plotting (pending)")

   console.print(tree)
   ```

**Deliverable:** `nanolab-pipeline run` with full pipeline visualization

---

### Phase 6: Utility Commands (Week 3-4)
**Goal:** Add helpful utility commands

**Tasks:**
1. **Status command** - Show pipeline state:
   ```bash
   nanolab-pipeline status
   ```
   Shows:
   - What dates have been staged
   - What dates have been preprocessed
   - What dates have been analyzed
   - Data sizes
   - Last run timestamps

2. **Info command** - Show configuration:
   ```bash
   nanolab-pipeline info
   ```
   Shows:
   - Data paths
   - Configuration files
   - Installed versions
   - System info

3. **Validate command** - Check pipeline health:
   ```bash
   nanolab-pipeline validate
   ```
   Checks:
   - All directories exist
   - Config files valid
   - Intermediate data consistency
   - Dependencies installed

**Deliverable:** Helpful utility commands

---

### Phase 7: Polish & Features (Week 4)
**Goal:** Add advanced features and polish

**Tasks:**
1. **Auto-completion:**
   ```bash
   nanolab-pipeline --install-completion
   ```

2. **Interactive mode:**
   ```python
   from rich.prompt import Confirm, Prompt

   if Confirm.ask("Run preprocessing?"):
       # Run preprocessing
   ```

3. **Configuration wizard:**
   ```bash
   nanolab-pipeline init
   ```
   Interactive prompts to create config file.

4. **Dry-run mode:**
   ```bash
   nanolab-pipeline run --dry-run
   ```
   Shows what would be done without executing.

5. **Verbose/quiet modes:**
   ```bash
   nanolab-pipeline run --verbose  # Extra details
   nanolab-pipeline run --quiet    # Minimal output
   ```

**Deliverable:** Polished CLI with advanced features

---

## 5. Rich UI Components Plan

### Progress Indicators

**File Processing:**
```python
with Progress() as progress:
    task1 = progress.add_task("[green]Staging files...", total=1000)
    task2 = progress.add_task("[cyan]Validating...", total=1000)
    # Update as files process
```

**Multi-step Pipeline:**
```python
from rich.progress import BarColumn, TextColumn

progress = Progress(
    TextColumn("[bold blue]{task.fields[stage]}"),
    BarColumn(),
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
)

with progress:
    staging_task = progress.add_task(stage="Staging", total=100)
    preprocess_task = progress.add_task(stage="Preprocessing", total=100)
    analysis_task = progress.add_task(stage="Analysis", total=100)
```

### Tables

**Configuration Summary:**
```python
from rich.table import Table

config_table = Table(title="Pipeline Configuration", show_header=False)
config_table.add_column("Parameter", style="cyan")
config_table.add_column("Value", style="green")

config_table.add_row("Date", "2025-10-18")
config_table.add_row("Procedure", "IV")
config_table.add_row("Polynomial Orders", "[1, 3, 5, 7]")
config_table.add_row("Workers", "8")

console.print(config_table)
```

**Results Summary:**
```python
results_table = Table(title="Analysis Results")
results_table.add_column("Date", style="cyan")
results_table.add_column("Runs", justify="right", style="magenta")
results_table.add_column("Segments", justify="right", style="magenta")
results_table.add_column("Time", justify="right", style="green")

results_table.add_row("2025-01-08", "7636", "45816", "10.2s")
results_table.add_row("2025-10-18", "5234", "31404", "7.8s")

console.print(results_table)
```

### Panels

**Step Headers:**
```python
from rich.panel import Panel

console.print(Panel.fit(
    "[bold cyan]STEP 1: STAGING[/bold cyan]\n"
    "Converting raw CSV files to Parquet format",
    border_style="cyan"
))
```

**Success/Error Messages:**
```python
# Success
console.print(Panel(
    "✓ Preprocessing completed successfully\n"
    f"Processed: 7636 runs → 45816 segments\n"
    f"Time: 2m 15s",
    title="[bold green]Success",
    border_style="green"
))

# Error
console.print(Panel(
    "❌ intermediate_root not found\n"
    "Run preprocessing first: nanolab-pipeline preprocess",
    title="[bold red]Error",
    border_style="red"
))
```

### Live Status

**Real-time Updates:**
```python
from rich.live import Live
from rich.table import Table

def generate_status_table():
    table = Table()
    table.add_column("Worker")
    table.add_column("Status")
    table.add_column("Progress")
    # ... populate from worker status
    return table

with Live(generate_status_table(), refresh_per_second=4) as live:
    while processing:
        live.update(generate_status_table())
```

### Trees

**Pipeline Overview:**
```python
from rich.tree import Tree

pipeline = Tree("📊 Nanolab Pipeline")
staging = pipeline.add("Layer 1: Raw CSV")
stage = pipeline.add("Layer 2: Staged Parquet")
intermediate = pipeline.add("Layer 3: Intermediate (Segments)")
intermediate.add("✓ 2025-01-08 (7636 runs)")
intermediate.add("✓ 2025-10-18 (5234 runs)")
analysis = pipeline.add("Layer 4: Analysis")

console.print(pipeline)
```

---

## 6. Backward Compatibility

### Strategy

1. **Keep existing scripts functional** during migration
2. **Create shim layer** that allows both old and new CLI
3. **Deprecation warnings** for old scripts:
   ```python
   import warnings
   warnings.warn(
       "This script is deprecated. Use 'nanolab-pipeline stage' instead.",
       DeprecationWarning
   )
   ```

4. **Migration guide** in documentation

### Transition Period

- **Phase 1-3:** Both CLIs work simultaneously
- **Phase 4:** New CLI becomes default, old CLI shows warnings
- **Phase 5:** Old CLI removed (major version bump)

---

## 7. Configuration Integration

### Typer + Pydantic Integration

```python
from pathlib import Path
import typer
from pydantic import ValidationError
from src.models.parameters import PipelineParameters

@app.command()
def run(
    config: Path = typer.Option(None, help="Config JSON file"),
    date: str = typer.Option(None, help="Analysis date"),
):
    """Run the complete pipeline."""
    try:
        if config:
            params = PipelineParameters.from_json(config)
        else:
            # Build params from CLI args
            params = PipelineParameters(...)

        # Validation happens automatically!

    except ValidationError as e:
        console.print("[red]Configuration validation failed:[/red]")
        console.print(e)
        raise typer.Exit(1)
```

---

## 8. Error Handling & User Experience

### Improved Error Messages

**Before:**
```
Error: No such file or directory: data/03_intermediate/iv_segments
```

**After (with Rich):**
```
╭─ Error ─────────────────────────────────────────────╮
│ ❌ Intermediate data not found                      │
│                                                      │
│ Path: data/03_intermediate/iv_segments              │
│                                                      │
│ Solution:                                            │
│ Run preprocessing first:                            │
│   nanolab-pipeline preprocess --date 2025-10-18     │
│                                                      │
│ Or check your configuration:                        │
│   nanolab-pipeline info                             │
╰──────────────────────────────────────────────────────╯
```

### Validation Feedback

```python
from rich.prompt import Confirm

if not intermediate_root.exists():
    console.print("[yellow]⚠ Intermediate data not found[/yellow]")

    if Confirm.ask("Would you like to run preprocessing now?"):
        # Run preprocessing automatically
    else:
        console.print("[red]Aborted[/red]")
        raise typer.Exit(1)
```

---

## 9. Testing Strategy

### Unit Tests
```python
# tests/cli/test_commands.py
from typer.testing import CliRunner
from src.cli.main import app

runner = CliRunner()

def test_stage_command():
    result = runner.invoke(app, ["stage", "--help"])
    assert result.exit_code == 0
    assert "Stage raw CSV" in result.stdout

def test_stage_with_config():
    result = runner.invoke(app, [
        "stage",
        "--raw-root", "data/01_raw",
        "--stage-root", "data/02_stage/raw_measurements"
    ])
    assert result.exit_code == 0
```

### Integration Tests
- Test full pipeline with test data
- Verify Rich output formatting
- Check progress bars update correctly

---

## 10. Documentation Updates

### Files to Update

1. **QUICK_START.md** - Add new CLI commands
2. **CLAUDE.md** - Update with Typer commands
3. **README.md** - Show new CLI interface
4. **New: CLI_REFERENCE.md** - Complete CLI documentation

### CLI Reference Structure

```markdown
# CLI Reference

## Installation
pip install -e .

## Auto-completion
nanolab-pipeline --install-completion

## Commands

### stage
Stage raw CSV files to Parquet

### preprocess
Detect and segment voltage sweeps

### analyze
Aggregate statistics and fit polynomials

...
```

---

## 11. Implementation Checklist

### Phase 1: Infrastructure ✅
- [ ] Create `src/cli/` directory structure
- [ ] Set up Rich console singleton
- [ ] Create main Typer app
- [ ] Create entry point script
- [ ] Test `--help` output

### Phase 2: Staging ✅
- [ ] Create `stage` command
- [ ] Add Rich progress bars
- [ ] Add configuration panel
- [ ] Add results table
- [ ] Test with real data

### Phase 3: Preprocessing ✅
- [ ] Create `preprocess` command
- [ ] Add live progress tracking
- [ ] Add multi-worker progress
- [ ] Add stats display
- [ ] Test with real data

### Phase 4: Analysis ✅
- [ ] Create `analyze` command
- [ ] Create `hysteresis` command
- [ ] Create `peaks` command
- [ ] Add fit results tables
- [ ] Test with real data

### Phase 5: Full Pipeline ✅
- [ ] Create `run` command
- [ ] Add pipeline visualization
- [ ] Add step-by-step progress
- [ ] Add final summary
- [ ] Test end-to-end

### Phase 6: Utilities ✅
- [ ] Create `status` command
- [ ] Create `info` command
- [ ] Create `validate` command
- [ ] Test utility functions

### Phase 7: Polish ✅
- [ ] Add auto-completion
- [ ] Add interactive mode
- [ ] Add config wizard (`init`)
- [ ] Add dry-run mode
- [ ] Add verbose/quiet modes

### Phase 8: Testing ✅
- [ ] Write unit tests
- [ ] Write integration tests
- [ ] Test edge cases
- [ ] Performance testing

### Phase 9: Documentation ✅
- [ ] Update QUICK_START.md
- [ ] Update CLAUDE.md
- [ ] Create CLI_REFERENCE.md
- [ ] Add migration guide
- [ ] Update README.md

---

## 12. Example Output (Mockup)

### `nanolab-pipeline run --date 2025-10-18`

```
╭─ Nanolab Pipeline ───────────────────────────────────╮
│                                                       │
│  📊 4-Layer Data Processing Pipeline                 │
│  Date: 2025-10-18                                    │
│  Procedure: IV                                       │
│                                                       │
╰───────────────────────────────────────────────────────╯

┏━ Configuration ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Parameter              Value                         ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Stage Root             data/02_stage/...             │
│ Intermediate Root      data/03_intermediate/...      │
│ Output Base            data/04_analysis              │
│ Polynomial Orders      [1, 3, 5, 7]                  │
│ Workers                8                             │
└───────────────────────────────────────────────────────┘

━━━━━━━━━━━ Step 1: Staging ━━━━━━━━━━━
✓ Data already staged

━━━━━━━━━━━ Step 2: Preprocessing ━━━━━━━━━━━
Processing runs... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:02:15
✓ Created 45,816 segments from 7,636 runs

━━━━━━━━━━━ Step 3: Analysis ━━━━━━━━━━━
Computing statistics... ━━━━━━━━━━━━━━━━━━━ 100% 0:00:08
Fitting polynomials... ━━━━━━━━━━━━━━━━━━━━ 100% 0:00:02

┏━ Polynomial Fit Quality ━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Order  R²        RMSE      Status                   ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 1      0.999984  1.23e-11  ✓ Excellent              │
│ 3      0.999992  8.45e-12  ✓ Excellent              │
│ 5      0.999995  6.21e-12  ✓ Excellent              │
│ 7      0.999997  4.87e-12  ✓ Excellent              │
└───────────────────────────────────────────────────────┘

╭─ Success ────────────────────────────────────────────╮
│                                                       │
│  ✓ Pipeline completed successfully!                  │
│                                                       │
│  Total time: 2m 25s                                  │
│                                                       │
│  Results saved to:                                   │
│    • data/04_analysis/iv_stats/2025-10-18_IV/        │
│    • data/04_analysis/hysteresis/2025-10-18_IV/      │
│                                                       │
╰───────────────────────────────────────────────────────╯
```

---

## 13. Priority & Timeline

### High Priority (Must Have)
1. ✅ Core Typer app structure
2. ✅ Main pipeline command (`run`)
3. ✅ Rich progress bars for long operations
4. ✅ Better error messages with solutions
5. ✅ Configuration display

### Medium Priority (Should Have)
6. ✅ Individual subcommands (stage, preprocess, analyze)
7. ✅ Results tables with fit quality
8. ✅ Status/info utility commands
9. ✅ Auto-completion support

### Low Priority (Nice to Have)
10. ⭐ Interactive config wizard
11. ⭐ Dry-run mode
12. ⭐ Live status updates for workers
13. ⭐ Tree visualizations

### Estimated Timeline
- **Minimal viable CLI:** 1 week
- **Full featured CLI:** 3-4 weeks
- **Polished + tested:** 4-5 weeks

---

## 14. Next Steps

1. **Review this plan** - Get feedback
2. **Prototype** - Build basic Typer app with one command
3. **Iterate** - Add Rich features incrementally
4. **Test** - Verify with real pipeline runs
5. **Document** - Update user guides
6. **Deploy** - Roll out to users

---

## Questions to Address

1. **Entry point name:** `nanolab-pipeline` vs `nanolab` vs `nlp`?
2. **Backward compatibility:** How long to keep old scripts?
3. **Interactive features:** How much interactivity do users want?
4. **Verbosity levels:** What detail levels make sense?
5. **Configuration wizard:** Worth the effort?

---

**Ready to start implementation?** 🚀
