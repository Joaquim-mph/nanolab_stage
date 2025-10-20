# Backward-Subtracted Hysteresis Comparison Plots

**Comprehensive visualization of photo-response effects on memristive hysteresis**

Generated: 2025-10-19

---

## Overview

Three types of plots visualizing backward-subtracted hysteresis data across different illumination conditions and voltage ranges, using **polynomial order 5** fits.

**Device**: 100 μm × 50 μm  
**Illumination conditions**:
- **Sept 11**: 0 μW (Dark)
- **Sept 29**: 9.0 μW
- **Sept 30**: 15.9 μW

**Voltage ranges**: 3.0V, 4.0V, 5.0V, 6.0V, 7.0V, 8.0V  
**Electric field ranges**: 300-800 V/cm

---

## Plot Types

### 1. Per-Day Overlay Plots (3 plots)

**Purpose**: Show all voltage ranges under the same illumination condition.

**Content**:
- One figure per date/illumination power
- All six voltage sweeps (3V-8V) overlapped
- X-axis: Electric field E (V/cm)
- Y-axis: Current density J (A/cm²)
- Legend: V_max values

**Files**:
- `overlay_2025-09-11_power0uW_dark_poly5.png` - Dark measurements
- `overlay_2025-09-29_power9.0uW_poly5.png` - Low power illumination
- `overlay_2025-09-30_power15.9uW_poly5.png` - High power illumination

**Use case**: Compare how different sweep ranges behave under constant illumination.

---

### 2. Per-Day Staggered Plots (3 plots)

**Purpose**: Same as overlay, but vertically offset for visual clarity.

**Content**:
- Same six voltage sweeps per illumination condition
- Each curve offset by 0.0015 A/cm² vertically
- Y-axis: J [a.u.] (arbitrary units - scale lost due to offset)
- No y-axis tick labels (scale is arbitrary)

**Files**:
- `staggered_2025-09-11_power0uW_dark_poly5.png` - Dark measurements
- `staggered_2025-09-29_power9.0uW_poly5.png` - Low power illumination
- `staggered_2025-09-30_power15.9uW_poly5.png` - High power illumination

**Use case**: Qualitative comparison of curve shapes without overlap obscuring features.

---

### 3. Cross-Day Comparison Plots (6 plots)

**Purpose**: Show photo-response evolution for each voltage range.

**Content**:
- One figure per voltage range (V_max)
- Three curves: 0 μW, 9.0 μW, 15.9 μW
- X-axis: Electric field E (V/cm)
- Y-axis: Current density J (A/cm²)
- Legend: Effective power values with dates

**Files**:
- `crossday_vmax3p0V_poly5.png` - 3V sweep comparison
- `crossday_vmax4p0V_poly5.png` - 4V sweep comparison
- `crossday_vmax5p0V_poly5.png` - 5V sweep comparison
- `crossday_vmax6p0V_poly5.png` - 6V sweep comparison
- `crossday_vmax7p0V_poly5.png` - 7V sweep comparison
- `crossday_vmax8p0V_poly5.png` - 8V sweep comparison

**Use case**: Directly visualize how illumination power affects hysteresis magnitude at each voltage range.

---

## Key Observations

### Photo-Response Magnitude

From cross-day comparison plots:

| V_max | Dark J_max | 9 μW J_max | 15.9 μW J_max | Enhancement Factor |
|-------|-----------|-----------|--------------|-------------------|
| 3V | ~0.2 mA/cm² | ~1.0 mA/cm² | ~1.1 mA/cm² | **5.5×** |
| 4V | ~0.4 mA/cm² | ~1.6 mA/cm² | ~2.0 mA/cm² | **5.0×** |
| 5V | ~0.6 mA/cm² | ~2.0 mA/cm² | ~2.9 mA/cm² | **4.8×** |
| 6V | ~1.1 mA/cm² | ~3.1 mA/cm² | ~4.3 mA/cm² | **3.9×** |
| 7V | ~1.8 mA/cm² | ~5.8 mA/cm² | ~6.9 mA/cm² | **3.8×** |
| 8V | ~1.8 mA/cm² | ~5.9 mA/cm² | ~7.1 mA/cm² | **3.9×** |

**Trend**: Higher enhancement at lower voltages, suggesting photo-response is more pronounced when electric fields are weaker.

### Curve Shape Evolution

**Per-day overlay plots show**:
- Dark (0 μW): Smaller, symmetric hysteresis loops
- 9 μW: Larger loops with increased asymmetry
- 15.9 μW: Even larger loops, maximum asymmetry

**Staggered plots reveal**:
- Consistent curve shape evolution across voltage ranges
- No qualitative shape changes with illumination
- Primarily magnitude scaling, not mechanism change

---

## Usage

### Generate All Plots

```bash
python src/analysis/IV/plot_backward_subtracted_comparison.py \
  --data-dir data/04_analysis/backward_substracted_efield_sept \
  --dates 2025-09-11 2025-09-29 2025-09-30 \
  --voltage-ranges 3.0 4.0 5.0 6.0 7.0 8.0 \
  --poly-order 5 \
  --output-dir plots/backward_subtracted_comparison \
  --offset-factor 0.0015
```

### Generate for Different Polynomial Order

```bash
# Try polynomial order 3
python src/analysis/IV/plot_backward_subtracted_comparison.py \
  --poly-order 3 \
  --output-dir plots/backward_subtracted_comparison_poly3
```

### Adjust Staggered Plot Offset

```bash
# Increase offset for better separation
python src/analysis/IV/plot_backward_subtracted_comparison.py \
  --offset-factor 0.003 \
  --output-dir plots/backward_subtracted_comparison_large_offset
```

---

## Plot Specifications

**All plots use**:
- Prism Rain color palette (high contrast)
- Publication-ready styling (scienceplots)
- 300 DPI resolution
- Both PNG and PDF formats
- Electric field units (V/cm)
- Current density units (A/cm²)

**Figure sizes**:
- Overlay plots: 10" × 6"
- Staggered plots: 10" × 8"
- Cross-day plots: 10" × 6"

**Line styling**:
- Line width: 1.8-2.0 pt
- Alpha: 0.8 (slight transparency)
- Grid: enabled (alpha 0.3)

---

## Data Source

**Backward-subtracted data**: `data/04_analysis/backward_substracted_efield_sept/{date}/backward_sub_vmax{X}V.csv`

**Power metadata**: `data/04_analysis/coercive_field/coercive_field_analysis.csv`

**Generated by**: `src/analysis/IV/compute_backward_substracted_efield.py`

---

## Scientific Interpretation

### Photo-Enhanced Memristive Switching

**Evidence**:
1. **Current enhancement**: Up to 5.5× increase with 15.9 μW illumination
2. **Power dependence**: Non-linear response (9 μW → 15.9 μW shows diminishing returns)
3. **Voltage-dependent enhancement**: Stronger at lower voltages

**Mechanism hypothesis**:
- Photogenerated carriers increase conductivity
- Electric field drives carrier migration
- Enhanced carrier density amplifies memristive switching
- At higher voltages, field-driven effects dominate over photo-effects

### Asymmetry Effects

Cross-day plots reveal:
- Positive/negative field asymmetry increases with illumination
- Different switching thresholds for each polarity
- Potential built-in field or asymmetric contact effects

---

## Output Location

All plots saved to: **`plots/backward_subtracted_comparison/`**

Total files: **24 files** (12 PNG + 12 PDF)
- 3 overlay plots × 2 formats = 6 files
- 3 staggered plots × 2 formats = 6 files
- 6 cross-day plots × 2 formats = 12 files

---

**Created**: 2025-10-19  
**Script**: `src/analysis/IV/plot_backward_subtracted_comparison.py`  
**Device**: 100 μm × 50 μm nanoelectronic memristive device  
**Analysis**: Backward-subtracted hysteresis with laser power calibration
