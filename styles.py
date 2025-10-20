"""
Legacy styles.py - Maintained for backward compatibility

This file is kept for existing scripts that import from it directly.
For new scripts, use the improved plotting configuration system:

    from src.ploting.plotting_config import setup_publication_style
    setup_publication_style(theme='prism_rain_large')

This provides the same styling with better organization and more features.
"""

import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Try to import from the new system
try:
    sys.path.insert(0, str(Path(__file__).parent / 'src' / 'ploting'))
    from plotting_config import setup_publication_style as _new_setup
    _USE_NEW_SYSTEM = True
except ImportError:
    _USE_NEW_SYSTEM = False
    print("Warning: Could not import new plotting system, using legacy")

# ============================================================================
# Common settings across all themes
# ============================================================================
COMMON_RC = {
    "text.usetex": False,
    "lines.markersize": 6,
    "legend.fontsize": 12,
    "axes.grid": False,
}

# ============================================================================
# Color Palettes
# ============================================================================
PRISM_RAIN_PALETTE = [
    # Primary vibrant colors (classic, high-contrast)
    "#e41a1c",  # red
    "#377eb8",  # blue
    "#4daf4a",  # green
    "#984ea3",  # purple
    "#ff7f00",  # orange

    # Extended vivid tones (brighter, neon-like accents)
    "#00bfc4",  # cyan-teal
    "#f781bf",  # pink
    "#ffd92f",  # bright yellow
    "#a65628",  # warm brown-orange
    "#8dd3c7",  # aqua-mint

    # Deep complementary accents (maintain contrast)
    "#b2182b",  # crimson
    "#2166ac",  # royal blue
    "#1a9850",  # rich green
    "#762a83",  # deep violet
    "#e08214",  # vivid amber
]

PRISM_RAIN_PALETTE_VIVID = [
    "#ff0054", "#0099ff", "#00cc66", "#cc33ff", "#ffaa00",
    "#00e6e6", "#ff66b2", "#ffe600", "#ff3300", "#00b3b3",
    "#3366ff", "#66ff33", "#9933ff", "#ff9933", "#33ccff",
]

# ============================================================================
# Theme Definitions
# ============================================================================
THEMES = {
    "prism_rain": {
        "base": ["science"],
        "rc": {
            **COMMON_RC,
            
            # Background colors
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#ffffff",
            "savefig.facecolor": "#ffffff",
            
            # Typography - FIXED SIZES FOR BETTER BALANCE
            #"font.family": "serif",  # ← FIXED (was "serif")
            "font.sans-serif": ["Source Sans Pro Black", "Source Sans 3"],
            "font.size": 35,              # ← REDUCED from 35
            
            "axes.labelsize": 55,         # ← REDUCED from 55 (axis labels)
            "axes.titlesize": 55,         # ← REDUCED from 55 (title)
            "axes.labelweight": "normal",
            
            # Axes and ticks
            "axes.edgecolor": "#222222",
            "axes.labelcolor": "#222222",
            "axes.linewidth": 3.5,        # ← REDUCED from 3.5
            
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "xtick.major.size": 10.0,      # ← REDUCED from 10.0
            "ytick.major.size": 10.0,      # ← REDUCED from 10.0
            "xtick.major.width": 2,
            "ytick.major.width": 2,
            "xtick.labelsize": 55,        # ← REDUCED from 55 (tick numbers!)
            "ytick.labelsize": 55,        # ← REDUCED from 55 (tick numbers!)
            "xtick.major.pad": 20,        # ← REDUCED from 20
            "ytick.major.pad": 20,        # ← REDUCED from 20
            
            # Grid
            "grid.color": "#cccccc",
            "grid.linestyle": "--",
            "grid.linewidth": 0.4,
            "grid.alpha": 0.6,
            
            # Lines and markers
            "lines.linewidth": 6,         # ← REDUCED from 6
            "lines.markersize": 22,       # ← REDUCED from 22
            "lines.antialiased": True,
            
            # Legend
            "legend.frameon": False,
            "legend.fontsize": 35,        # ← REDUCED from 35
            "legend.loc": "best",
            "legend.fancybox": True,
            
            # Figure size (optimized for papers)
            "figure.figsize": (20, 20),
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
            
            # Color cycle
            "axes.prop_cycle": plt.cycler(color=PRISM_RAIN_PALETTE),
        }
    },
}

# ============================================================================
# Helper function to apply theme
# ============================================================================
def set_plot_style(theme_name="prism_rain"):
    """Apply a publication-ready matplotlib theme.

    NOTE: This function now uses the new plotting system when available.
    The 'prism_rain' theme maps to 'prism_rain_large' in the new system
    to preserve your exact large font parameters.

    Parameters
    ----------
    theme_name : str, default="prism_rain"
        Name of the theme to apply

    Example
    -------
    >>> set_plot_style("prism_rain")
    >>> plt.plot([1, 2, 3], [1, 4, 9])
    >>> plt.show()
    """
    # Use new system if available (with theme mapping)
    if _USE_NEW_SYSTEM:
        # Map legacy theme names to new system
        theme_map = {
            'prism_rain': 'prism_rain_large',  # Your exact large parameters
        }
        new_theme = theme_map.get(theme_name, theme_name)

        try:
            _new_setup(theme=new_theme, use_scienceplots=True, dpi=300)
            return
        except Exception as e:
            print(f"Warning: New system failed ({e}), falling back to legacy")

    # Fallback to legacy implementation
    if theme_name not in THEMES:
        raise ValueError(f"Theme '{theme_name}' not found. Available: {list(THEMES.keys())}")

    theme = THEMES[theme_name]

    # Apply base styles if specified
    if "base" in theme:
        for base_style in theme["base"]:
            try:
                plt.style.use(base_style)
            except OSError:
                print(f"Warning: Base style '{base_style}' not found, skipping...")

    # Apply custom rc parameters
    plt.rcParams.update(theme["rc"])
    print(f"✓ Applied '{theme_name}' theme")