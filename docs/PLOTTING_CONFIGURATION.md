# Publication-Ready Plotting Configuration

**Professional matplotlib styling for scientific publications**

This document explains how to use the plotting configuration system for creating consistent, publication-quality figures.

---

## Quick Start

```python
from src.ploting.plotting_config import setup_publication_style

# Setup publication style (call once at script start)
setup_publication_style(theme='default')

# Create your plots
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 9])
plt.show()
```

That's it! Your plots now use publication-ready defaults.

---

## Installation

### Required Dependencies

```bash
# Install matplotlib (should already be installed)
pip install matplotlib

# REQUIRED: Install scienceplots for publication-ready styles
pip install scienceplots
```

**Note:** scienceplots is **required** and enabled by default for all themes to ensure publication-quality output.

### Update Your Project

The configuration files are already in `src/ploting/`:
- `matplotlibrc` - Base configuration file
- `plotting_config.py` - Python module with themes and helpers
- `plotting_example.py` - Example scripts
- `styles_legacy.py` - Your original styles (for reference)

---

## Available Themes

### 1. Default (Publication Standard)

**Best for:** Academic papers, technical reports

**Characteristics:**
- Conservative sizing (10-12pt fonts)
- Clean, professional appearance
- Colorblind-friendly palette
- Standard matplotlib colors

**Usage:**
```python
setup_publication_style(theme='default')
```

**When to use:**
- Journal submissions
- Technical documentation
- General scientific figures

---

### 2. Prism Rain (High Contrast)

**Best for:** Presentations, posters, distinguishing many datasets

**Characteristics:**
- Bold, vibrant colors
- Thicker lines (2.0pt)
- Larger markers (size 8)
- Enhanced visibility
- 15-color palette

**Usage:**
```python
setup_publication_style(theme='prism_rain')
```

**When to use:**
- Conference posters
- Multiple overlapping datasets
- When maximum visual distinction is needed
- Dark backgrounds (projectors)

---

### 3. Minimal (Clean & Simple)

**Best for:** Reports, documentation, clean aesthetics

**Characteristics:**
- Minimal styling
- No top/right spines
- Light grid
- Understated colors
- Small fonts (9-10pt)

**Usage:**
```python
setup_publication_style(theme='minimal')
```

**When to use:**
- Internal reports
- Documentation
- Web-based figures
- When less is more

---

### 4. Presentation (Large Text)

**Best for:** Slides, talks, projected figures

**Characteristics:**
- Large fonts (14-18pt)
- Thick lines (3.0pt)
- Large markers (size 10)
- High contrast
- Figure size: 12×9

**Usage:**
```python
setup_publication_style(theme='presentation')
```

**When to use:**
- PowerPoint/Keynote slides
- Conference talks
- Projected presentations
- When viewed from distance

---

## Color Palettes

### Available Palettes

```python
from src.ploting.plotting_config import PALETTES, get_color_cycle

# Get colors from a palette
colors = get_color_cycle('prism_rain', n_colors=5)

# Use in plots
plt.plot(x, y, color=colors[0])
```

**Palettes:**
1. **`default`** - Standard matplotlib Tableau (10 colors, colorblind-safe)
2. **`prism_rain`** - High-contrast vibrant (15 colors)
3. **`prism_rain_vivid`** - Maximum contrast (15 colors, very bright)
4. **`minimal`** - Professional understated (9 colors, Nord-inspired)
5. **`scientific`** - Nature journal inspired (8 colors)

### Color Preview

```python
# Example: Show all palettes
from src.ploting.plotting_config import PALETTES

for name, colors in PALETTES.items():
    print(f"{name}: {len(colors)} colors")
    print(f"  First 3: {colors[:3]}")
```

---

## Helper Functions

### Setup Publication Style

```python
setup_publication_style(theme='default', use_scienceplots=False, dpi=300)
```

**Parameters:**
- `theme` - Theme name ('default', 'prism_rain', 'minimal', 'presentation')
- `use_scienceplots` - Apply scienceplots 'science' style as base (requires scienceplots)
- `dpi` - DPI for saved figures (default: 300)

**Example:**
```python
# Standard publication
setup_publication_style()

# With scienceplots base
setup_publication_style(use_scienceplots=True)

# High-resolution output
setup_publication_style(dpi=600)
```

---

### Save Figure

```python
save_figure(fig, filepath, dpi=300, formats=None, **kwargs)
```

**Saves figure in multiple formats automatically.**

**Parameters:**
- `fig` - Matplotlib figure object
- `filepath` - Output path (without extension)
- `dpi` - Resolution (default: 300)
- `formats` - List of formats (default: ['png', 'pdf'])
- `**kwargs` - Additional arguments for `fig.savefig()`

**Example:**
```python
from src.ploting.plotting_config import save_figure

fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 9])

# Save as PNG and PDF
save_figure(fig, 'my_plot', formats=['png', 'pdf'])

# Save as SVG only, high DPI
save_figure(fig, 'my_plot', formats=['svg'], dpi=600)
```

**Output:**
```
Saved: my_plot.png
Saved: my_plot.pdf
```

---

### Get Color Cycle

```python
colors = get_color_cycle(palette='default', n_colors=None)
```

**Get colors from a palette for custom use.**

**Example:**
```python
# Get 5 colors from prism_rain palette
colors = get_color_cycle('prism_rain', n_colors=5)

# Use in plots
for i, color in enumerate(colors):
    plt.plot(x, y + i, color=color, label=f'Line {i}')
```

---

### Apply Grid

```python
apply_grid(ax, which='both', alpha=0.3, linestyle='--', linewidth=0.5)
```

**Apply subtle grid to axes.**

**Example:**
```python
from src.ploting.plotting_config import apply_grid

fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 9])
apply_grid(ax)  # Add subtle grid
```

---

### Set Spine Visibility

```python
set_spine_visibility(ax, top=False, right=False, left=True, bottom=True)
```

**Control axis spines (box around plot).**

**Example:**
```python
from src.ploting.plotting_config import set_spine_visibility

fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 9])
set_spine_visibility(ax, top=False, right=False)  # Minimal style
```

---

## Context Manager (Advanced)

For temporary style changes:

```python
from src.ploting.plotting_config import PlotStyle

# Normal style
setup_publication_style('default')

# Temporarily use different style
with PlotStyle('prism_rain'):
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])
    plt.savefig('prism_plot.png')

# Back to previous style automatically
```

---

## Complete Examples

### Example 1: Basic Publication Plot

```python
from src.ploting.plotting_config import setup_publication_style, save_figure
import matplotlib.pyplot as plt
import numpy as np

# Setup
setup_publication_style('default')

# Data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Plot
fig, ax = plt.subplots()
ax.plot(x, y, label='sin(x)')
ax.set_xlabel('X (units)')
ax.set_ylabel('Y (units)')
ax.set_title('Publication-Ready Plot')
ax.legend()
ax.grid(True, alpha=0.3)

# Save
save_figure(fig, 'publication_plot', formats=['png', 'pdf'])
```

---

### Example 2: Multi-Panel Figure

```python
setup_publication_style('default')

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle('Multi-Panel Scientific Figure')

# Plot in each panel
x = np.linspace(0, 10, 100)

axes[0, 0].plot(x, x)
axes[0, 0].set_title('(a) Linear')

axes[0, 1].plot(x, x**2)
axes[0, 1].set_title('(b) Quadratic')

axes[1, 0].semilogy(x, np.exp(x/3))
axes[1, 0].set_title('(c) Exponential')

axes[1, 1].scatter(x, x + np.random.randn(100)*0.5)
axes[1, 1].set_title('(d) Scatter')

plt.tight_layout()
save_figure(fig, 'multipanel', formats=['png', 'pdf'])
```

---

### Example 3: Custom Colors

```python
from src.ploting.plotting_config import setup_publication_style, get_color_cycle

setup_publication_style('prism_rain')

# Get custom color cycle
colors = get_color_cycle('prism_rain', n_colors=5)

fig, ax = plt.subplots()

for i, color in enumerate(colors):
    y = np.sin(x + i * np.pi/4)
    ax.plot(x, y, color=color, linewidth=2, label=f'Dataset {i+1}')

ax.legend()
ax.grid(True, alpha=0.3)
save_figure(fig, 'custom_colors')
```

---

### Example 4: IV Curve (Realistic)

```python
setup_publication_style('prism_rain')

# Simulate IV data
V_fwd = np.linspace(-8, 8, 100)
V_ret = np.linspace(8, -8, 100)
I_fwd = 1e-9 * (V_fwd + 0.1 * V_fwd**3)
I_ret = 1e-9 * (V_ret + 0.1 * V_ret**3 - 0.3 * V_ret)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Linear scale
ax1.plot(V_fwd, I_fwd * 1e9, 'o-', label='Forward', markersize=3)
ax1.plot(V_ret, I_ret * 1e9, 's-', label='Return', markersize=3)
ax1.set_xlabel('Voltage V$_{sd}$ (V)')
ax1.set_ylabel('Current I (nA)')
ax1.set_title('(a) IV Curve - Linear')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Log scale
ax2.semilogy(V_fwd, np.abs(I_fwd), 'o-', label='Forward', markersize=3)
ax2.semilogy(V_ret, np.abs(I_ret), 's-', label='Return', markersize=3)
ax2.set_xlabel('Voltage V$_{sd}$ (V)')
ax2.set_ylabel('|Current| (A)')
ax2.set_title('(b) IV Curve - Log Scale')
ax2.legend()
ax2.grid(True, alpha=0.3, which='both')

plt.tight_layout()
save_figure(fig, 'iv_curve', formats=['png', 'pdf'])
```

---

## Integration with Existing Scripts

### Update Your Plotting Scripts

**Before:**
```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(x, y)
plt.savefig('plot.png', dpi=300, bbox_inches='tight')
```

**After:**
```python
import matplotlib.pyplot as plt
from src.ploting.plotting_config import setup_publication_style, save_figure

# Add this once at the start
setup_publication_style('default')

fig, ax = plt.subplots()
ax.plot(x, y)

# Use save_figure helper
save_figure(fig, 'plot', formats=['png', 'pdf'])
```

---

## Configuration Reference

### matplotlibrc Settings

Key settings in `src/ploting/matplotlibrc`:

```ini
# Publication standard
savefig.dpi         : 300
figure.figsize      : 10, 8
font.size           : 11
axes.labelsize      : 11
axes.titlesize      : 12
lines.linewidth     : 1.5
legend.fontsize     : 10

# Tight layout
savefig.bbox        : tight
savefig.pad_inches  : 0.1
```

### Theme Parameters

Each theme overrides specific parameters:

**Default Theme:**
- Fonts: 10-12pt
- Lines: 1.5pt
- Figure: 8×6 inches
- Colors: Tableau (10 colors)

**Prism Rain Theme:**
- Fonts: 10-13pt (bold labels)
- Lines: 2.0pt
- Figure: 10×8 inches
- Colors: Prism Rain (15 colors)

**Minimal Theme:**
- Fonts: 9-11pt
- Lines: 1.2pt
- Figure: 7×5 inches
- No top/right spines
- Grid enabled

**Presentation Theme:**
- Fonts: 14-18pt
- Lines: 3.0pt
- Figure: 12×9 inches
- Large markers

---

## Best Practices

### Journal Submissions

```python
# Use default theme
setup_publication_style('default', dpi=300)

# Save as high-quality formats
save_figure(fig, 'figure1', formats=['pdf', 'png'], dpi=600)
```

### Conference Posters

```python
# Use prism_rain for visibility
setup_publication_style('prism_rain', dpi=300)

# Larger figure sizes
fig, ax = plt.subplots(figsize=(12, 9))
```

### Presentations

```python
# Use presentation theme
setup_publication_style('presentation')

# Large, clear text
# Thick lines for visibility from distance
```

### Web Documentation

```python
# Use minimal theme
setup_publication_style('minimal', dpi=150)

# Save as SVG for web
save_figure(fig, 'web_plot', formats=['svg'])
```

---

## Troubleshooting

### Fonts Not Found

**Problem:** Warning about missing fonts

**Solution:**
```python
# Use default fonts (always available)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
```

### Figures Too Large/Small

**Problem:** Figure doesn't fit in document

**Solution:**
```python
# Customize figure size
fig, ax = plt.subplots(figsize=(width_inches, height_inches))

# Or adjust theme
setup_publication_style('default')
plt.rcParams['figure.figsize'] = (8, 6)
```

### Text Overlapping

**Problem:** Labels overlap with data

**Solution:**
```python
# Use tight_layout
plt.tight_layout()

# Or constrained_layout
plt.rcParams['figure.constrained_layout.use'] = True

# Or adjust padding
save_figure(fig, 'plot', pad_inches=0.2)
```

### scienceplots Not Found

**Problem:** "scienceplots 'science' style not found"

**Solution:**
```python
# Install scienceplots (optional)
pip install scienceplots

# Or don't use it
setup_publication_style(use_scienceplots=False)
```

---

## Examples Gallery

Run the examples script to see all themes and features:

```bash
python src/ploting/plotting_example.py
```

**Generates:**
- `example_1_basic.png` - Basic publication plot
- `example_2_prism_rain.png` - Prism rain theme
- `example_3_minimal.png` - Minimal style
- `example_4_multipanel.png` - Multi-panel figure
- `example_5_palettes.png` - Color palette comparison
- `example_6_*.png` - Context manager usage
- `example_7_presentation.png` - Presentation style
- `example_8_iv_realistic.png` - Realistic IV curve

---

## Migration Guide

### From Legacy styles.py

**Old code:**
```python
from styles import set_plot_style, PRISM_RAIN_PALETTE

set_plot_style("prism_rain")
```

**New code:**
```python
from src.ploting.plotting_config import setup_publication_style, get_color_cycle

setup_publication_style(theme='prism_rain')
colors = get_color_cycle('prism_rain')
```

### Benefits of New System

1. **Multiple formats** - Automatic PNG + PDF output
2. **Consistent theming** - All parameters coordinated
3. **Helper functions** - Grid, spines, colors
4. **Context managers** - Temporary style changes
5. **Documentation** - This comprehensive guide
6. **Examples** - Ready-to-use code snippets

---

## Related Documentation

- **matplotlibrc** - Base configuration file
- **plotting_config.py** - Source code with full API
- **plotting_example.py** - Example scripts
- **Matplotlib docs** - https://matplotlib.org/stable/tutorials/introductory/customizing.html

---

## Support

For issues or questions:
1. Check this documentation
2. Review examples: `python src/ploting/plotting_example.py`
3. Check source: `src/ploting/plotting_config.py`
4. Matplotlib docs: https://matplotlib.org

---

**Your plots are now publication-ready!** 📊✨
