# Pydantic Parameter Models - Migration Guide

## Overview

The project now uses **Pydantic v2** for configuration and validation throughout the data pipeline. This provides:

✅ **Type safety** - Automatic type checking and coercion
✅ **Validation** - Field-level constraints (min/max values, patterns, required fields)
✅ **Documentation** - Self-documenting parameter classes with descriptions
✅ **Error prevention** - Catches bad values before they cause runtime errors
✅ **JSON support** - Easy configuration loading/saving
✅ **IDE integration** - Full autocomplete and type hints

## Installation

```bash
pip install -r requirements.txt
```

Key additions:
- `pydantic>=2.10.0` - Core validation library
- `pytest>=8.3.0` - For running validation tests

## Parameter Classes

### 1. `StagingParameters`

Controls CSV-to-Parquet staging pipeline.

**Required fields:**
- `raw_root: Path` - Directory with raw CSV files
- `stage_root: Path` - Output directory for Parquet files
- `procedures_yaml: Path` - YAML schema file

**Optional fields:**
- `workers: int = 6` - Parallel workers (1-32)
- `polars_threads: int = 1` - Polars threads per worker (1-16)
- `local_tz: str = "America/Santiago"` - Timezone for date partitioning
- `force: bool = False` - Overwrite existing files
- `only_yaml_data: bool = False` - Drop non-YAML columns

**Validation rules:**
- `raw_root` and `procedures_yaml` must exist
- `procedures_yaml` must be a file (not directory)
- `workers` between 1 and 32
- `polars_threads` between 1 and 16

**Example:**
```python
from models.parameters import StagingParameters
from pathlib import Path

params = StagingParameters(
    raw_root=Path("data/01_raw"),
    stage_root=Path("data/02_stage/raw_measurements"),
    procedures_yaml=Path("config/procedures.yml"),
    workers=8,
    force=True
)
```

### 2. `IVAnalysisParameters`

Controls IV curve analysis pipeline.

**Required fields:**
- `stage_root: Path` - Root of staged Parquet data
- `date: str` - Date in YYYY-MM-DD format
- `output_base_dir: Path` - Base directory for analysis outputs

**Optional fields:**
- `procedure: str = "IV"` - Procedure name (IV, IVg, etc.)
- `chip_number: str | None = None` - Filter by chip
- `v_max: float | None = None` - Filter by voltage range (0-100V)
- `poly_orders: List[int] = [1, 3, 5, 7]` - Polynomial orders to fit
- `fit_backward: bool = True` - Fit polynomial to backward trace
- `compute_hysteresis: bool = True` - Calculate hysteresis curves
- `voltage_rounding_decimals: int = 2` - Decimal places for voltage alignment
- `analyze_peaks: bool = False` - Analyze peak locations

**Validation rules:**
- `date` must match pattern `YYYY-MM-DD`
- `stage_root` must exist
- `v_max` between 0 and 100 (if specified)
- `poly_orders` must be odd integers between 1 and 15
- Duplicates in `poly_orders` are automatically removed

**Helper methods:**
- `get_stats_dir()` - Returns IV statistics output directory
- `get_hysteresis_dir()` - Returns hysteresis output directory
- `get_peaks_dir()` - Returns peaks output directory

**Example:**
```python
from models.parameters import IVAnalysisParameters

params = IVAnalysisParameters(
    stage_root=Path("data/02_stage/raw_measurements"),
    date="2025-09-11",
    output_base_dir=Path("data/04_analysis"),
    poly_orders=[1, 3, 5, 7],
    compute_hysteresis=True,
    analyze_peaks=True
)

# Use helper methods
print(params.get_stats_dir())  # data/04_analysis/iv_stats/2025-09-11_IV
```

### 3. `PlottingParameters`

Controls visualization output.

**Required fields:**
- `output_dir: Path` - Directory for plot outputs

**Optional fields:**
- `dpi: int = 300` - Resolution (72-1200)
- `format: str = "png"` - Output format (png, pdf, svg, jpg)
- `figure_width: float = 12.0` - Width in inches (4-30)
- `figure_height: float = 8.0` - Height in inches (3-20)
- `style: str = "publication"` - Style preset (publication, presentation, notebook)
- `font_size: int = 10` - Base font size (6-24)
- `line_width: float = 1.5` - Line width (0.5-5.0)
- `show_error_bars: bool = True` - Display error bars
- `show_grid: bool = True` - Display grid
- `grid_alpha: float = 0.3` - Grid transparency (0-1)
- `compact_layout: bool = False` - Use compact subplot layout
- `show_residuals: bool = False` - Include residuals plots

**Validation rules:**
- `dpi` between 72 and 1200
- `format` must be one of: png, pdf, svg, jpg
- `style` must be one of: publication, presentation, notebook
- `grid_alpha` between 0.0 and 1.0

**Helper methods:**
- `get_figsize()` - Returns (width, height) tuple
- `get_style_params()` - Returns matplotlib rcParams dict

**Example:**
```python
from models.parameters import PlottingParameters

params = PlottingParameters(
    output_dir=Path("plots/analysis"),
    dpi=300,
    format="png",
    style="publication",
    compact_layout=True
)

# Apply style to matplotlib
import matplotlib.pyplot as plt
plt.rcParams.update(params.get_style_params())
```

### 4. `PipelineParameters`

Combined parameters for complete pipeline execution.

**Required fields:**
- `staging: StagingParameters` - Staging configuration
- `analysis: IVAnalysisParameters` - Analysis configuration
- `plotting: PlottingParameters` - Plotting configuration

**Optional fields:**
- `run_staging: bool = True` - Execute staging step
- `run_analysis: bool = True` - Execute analysis step
- `run_plotting: bool = True` - Execute plotting step

**Cross-parameter validation:**
- `analysis.stage_root` must match `staging.stage_root`
- If `run_plotting=True` without `run_analysis=True`, analysis outputs must exist

**Methods:**
- `from_json(path)` - Load from JSON config file
- `to_json(path)` - Save to JSON config file
- `validate_all_paths()` - Check inputs exist, create output directories

**Example:**
```python
from models.parameters import PipelineParameters

# Load from JSON
params = PipelineParameters.from_json("config/pipeline_config.json")

# Or create programmatically
params = PipelineParameters(
    staging=StagingParameters(...),
    analysis=IVAnalysisParameters(...),
    plotting=PlottingParameters(...)
)

# Validate and prepare
params.validate_all_paths()

# Save for later
params.to_json("config/my_config.json")
```

## Usage Examples

### Basic Usage

```python
from pathlib import Path
from models.parameters import StagingParameters
from pydantic import ValidationError

try:
    params = StagingParameters(
        raw_root=Path("data/01_raw"),
        stage_root=Path("data/02_stage/raw_measurements"),
        procedures_yaml=Path("config/procedures.yml"),
        workers=8,
    )
    print(f"Workers: {params.workers}")
    print(f"Default rejects dir: {params.rejects_dir}")

except ValidationError as e:
    print(f"Validation failed: {e}")
```

### Loading from JSON

```python
from models.parameters import PipelineParameters

# Load configuration
params = PipelineParameters.from_json("config/pipeline_config.json")

# Access nested parameters
print(f"Staging workers: {params.staging.workers}")
print(f"Analysis date: {params.analysis.date}")
print(f"Plotting DPI: {params.plotting.dpi}")

# Execute pipeline steps
if params.run_staging:
    run_staging(params.staging)

if params.run_analysis:
    run_analysis(params.analysis)

if params.run_plotting:
    run_plotting(params.plotting)
```

### Creating Configuration Programmatically

```python
from models.parameters import (
    StagingParameters,
    IVAnalysisParameters,
    PlottingParameters,
    PipelineParameters
)

# Build components
staging = StagingParameters(
    raw_root=Path("data/01_raw"),
    stage_root=Path("data/02_stage/raw_measurements"),
    procedures_yaml=Path("config/procedures.yml"),
    workers=12,
    force=True
)

analysis = IVAnalysisParameters(
    stage_root=staging.stage_root,  # Reference staging output
    date="2025-09-11",
    output_base_dir=Path("data/04_analysis"),
    poly_orders=[1, 3, 5, 7],
    compute_hysteresis=True
)

plotting = PlottingParameters(
    output_dir=Path("plots") / f"{analysis.date}_{analysis.procedure}",
    dpi=300,
    style="publication"
)

# Combine into pipeline
pipeline = PipelineParameters(
    staging=staging,
    analysis=analysis,
    plotting=plotting
)

# Save for later use
pipeline.to_json("config/my_pipeline.json")
```

## Validation Tests

Comprehensive tests demonstrate that validation catches:

- ❌ Invalid paths (non-existent directories)
- ❌ Out-of-range numeric values (workers, DPI, voltages)
- ❌ Invalid date formats (must be YYYY-MM-DD)
- ❌ Even polynomial orders (must be odd: 1, 3, 5, 7, etc.)
- ❌ Missing required fields
- ❌ Extra unknown fields
- ❌ Inconsistent cross-parameter values

**Run tests:**
```bash
# Run all tests
pytest tests/test_parameters.py -v

# Run specific test class
pytest tests/test_parameters.py::TestStagingParameters -v

# Run demonstration (shows validation errors)
python tests/test_parameters.py
```

**Example test output:**
```
======================================================================
VALIDATION DEMONSTRATION - Testing Bad Values
======================================================================

[TEST 1] Invalid worker count (0):
  ✓ CAUGHT: Input should be greater than or equal to 1

[TEST 2] Invalid date format (2025-9-11):
  ✓ CAUGHT: String should match pattern '^\d{4}-\d{2}-\d{2}$'

[TEST 3] Even polynomial order (2):
  ✓ CAUGHT: Value error, Polynomial order 2 should be odd for symmetric fitting

[TEST 4] DPI too high (2000):
  ✓ CAUGHT: Input should be less than or equal to 1200

[TEST 5] Invalid plot format (gif):
  ✓ CAUGHT: String should match pattern '^(png|pdf|svg|jpg)$'

✓ All validation tests passed - bad values were caught!
```

## Example Configurations

### Example 1: Full Pipeline

`config/examples/pipeline_config.json`:
```json
{
  "staging": {
    "raw_root": "data/01_raw",
    "stage_root": "data/02_stage/raw_measurements",
    "procedures_yaml": "config/procedures.yml",
    "workers": 8,
    "force": false
  },
  "analysis": {
    "stage_root": "data/02_stage/raw_measurements",
    "date": "2025-09-11",
    "output_base_dir": "data/04_analysis",
    "procedure": "IV",
    "poly_orders": [1, 3, 5, 7],
    "compute_hysteresis": true,
    "analyze_peaks": true
  },
  "plotting": {
    "output_dir": "plots/2025-09-11_IV",
    "dpi": 300,
    "format": "png",
    "style": "publication",
    "compact_layout": true,
    "show_residuals": true
  }
}
```

### Example 2: High-Performance Staging

```json
{
  "staging": {
    "raw_root": "data/01_raw",
    "stage_root": "data/02_stage/raw_measurements",
    "procedures_yaml": "config/procedures.yml",
    "workers": 16,
    "polars_threads": 2,
    "force": true,
    "only_yaml_data": true
  }
}
```

### Example 3: Presentation-Quality Plots

```json
{
  "plotting": {
    "output_dir": "plots/presentation",
    "dpi": 600,
    "format": "pdf",
    "style": "presentation",
    "figure_width": 16.0,
    "figure_height": 9.0,
    "font_size": 14,
    "line_width": 2.0,
    "show_grid": false
  }
}
```

## Interactive Examples

Run the interactive example script to see all features in action:

```bash
python examples/use_pydantic_config.py
```

This demonstrates:
1. Creating parameters programmatically
2. Validation error handling
3. Loading/saving JSON configs
4. Using helper methods
5. Pipeline-level validation
6. Building complex configurations step-by-step

## Migration from Old Parameter Dictionaries

### Before (argparse + dicts):
```python
def aggregate_iv_stats(
    stage_root: Path,
    date: str,
    output_dir: Path,
    procedure: str = "IVg",
    v_max_min: Optional[float] = None,
    chip_number: Optional[str] = None,
):
    # No validation until runtime
    # Manual type checking
    # Hard to document
    ...
```

### After (Pydantic):
```python
def aggregate_iv_stats(params: IVAnalysisParameters):
    # Validated before function call
    # Type-safe with autocomplete
    # Self-documenting

    # Access validated parameters
    stage_root = params.stage_root
    date = params.date
    output_dir = params.get_stats_dir()  # Helper method
    ...
```

## Benefits

### 1. Type Safety
```python
# Old way - no type checking
params = {"workers": "8"}  # String instead of int - runtime error later!

# New way - automatic coercion and validation
params = StagingParameters(workers="8")  # Automatically converted to int(8)
```

### 2. Validation Before Execution
```python
# Old way - fails during execution
params = {"workers": 0}  # Invalid, but no error yet
run_pipeline(params)  # Fails here after setup

# New way - fails immediately
params = StagingParameters(workers=0)  # ValidationError raised immediately
```

### 3. Clear Error Messages
```python
# Old way
ValueError: workers must be positive

# New way
ValidationError: 1 validation error for StagingParameters
workers
  Input should be greater than or equal to 1 [type=greater_than_equal, input_value=0, input_type=int]
```

### 4. Self-Documentation
```python
# Just look at the class definition!
class StagingParameters(BaseModel):
    workers: int = Field(
        default=6,
        ge=1,
        le=32,
        description="Number of parallel worker processes"
    )
```

### 5. JSON Serialization
```python
# Save
params.to_json("config.json")

# Load
params = PipelineParameters.from_json("config.json")
```

## Next Steps

1. **Update existing scripts** to use Pydantic models instead of argparse/dicts
2. **Create JSON configs** for common workflows
3. **Add more validation rules** as needed
4. **Use helper methods** throughout codebase

## File Structure

```
nanolab_stage/
├── src/
│   └── models/
│       ├── __init__.py
│       └── parameters.py           # Pydantic model definitions
├── tests/
│   └── test_parameters.py          # Comprehensive validation tests
├── examples/
│   └── use_pydantic_config.py      # Interactive usage examples
├── config/
│   └── examples/
│       ├── pipeline_config.json    # Full pipeline example
│       └── ...
├── requirements.txt                # Updated with pydantic
└── PYDANTIC_MIGRATION.md          # This file
```

## References

- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Pydantic V2 Migration Guide](https://docs.pydantic.dev/latest/migration/)
- Project tests: `tests/test_parameters.py`
- Example usage: `examples/use_pydantic_config.py`
