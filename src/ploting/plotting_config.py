"""
Publication-Ready Matplotlib Configuration

This module provides professional matplotlib styling for scientific publications.
Includes multiple themes and helper functions for consistent, high-quality figures.

Usage:
    from src.ploting.plotting_config import setup_publication_style, PALETTES

    # Use default publication style
    setup_publication_style()

    # Use custom theme
    setup_publication_style(theme='prism_rain')

    # Access color palettes
    colors = PALETTES['prism_rain']

Features:
    - Publication-ready defaults (300 DPI, proper sizing)
    - Colorblind-friendly palettes
    - Multiple themes (default, prism_rain, minimal)
    - Optional scienceplots integration
    - Consistent styling across all plots

Author: Nanolab Pipeline
Date: 2025-10-19
"""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import cycler
import warnings

# =============================================================================
# Color Palettes (Colorblind-Friendly)
# =============================================================================

PALETTES = {
    # Default matplotlib tableau palette (colorblind-safe)
    'default': [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#7f7f7f',  # Gray
        '#bcbd22',  # Yellow-green
        '#17becf',  # Cyan
    ],

    # Prism Rain palette (high contrast, vibrant)
    'prism_rain': [
        '#e41a1c',  # Red
        '#377eb8',  # Blue
        '#4daf4a',  # Green
        '#984ea3',  # Purple
        '#ff7f00',  # Orange
        '#00bfc4',  # Cyan-teal
        '#f781bf',  # Pink
        '#ffd92f',  # Bright yellow
        '#a65628',  # Warm brown-orange
        '#8dd3c7',  # Aqua-mint
        '#b2182b',  # Crimson
        '#2166ac',  # Royal blue
        '#1a9850',  # Rich green
        '#762a83',  # Deep violet
        '#e08214',  # Vivid amber
    ],

    # Prism Rain Vivid (maximum contrast)
    'prism_rain_vivid': [
        '#ff0054', '#0099ff', '#00cc66', '#cc33ff', '#ffaa00',
        '#00e6e6', '#ff66b2', '#ffe600', '#ff3300', '#00b3b3',
        '#3366ff', '#66ff33', '#9933ff', '#ff9933', '#33ccff',
    ],

    # Minimal palette (professional, understated)
    'minimal': [
        '#2E3440',  # Dark gray
        '#5E81AC',  # Blue
        '#88C0D0',  # Light blue
        '#81A1C1',  # Medium blue
        '#BF616A',  # Red
        '#D08770',  # Orange
        '#EBCB8B',  # Yellow
        '#A3BE8C',  # Green
        '#B48EAD',  # Purple
    ],

    # Scientific publication palette (Nature-inspired)
    'scientific': [
        '#0173B2',  # Blue
        '#DE8F05',  # Orange
        '#029E73',  # Green
        '#CC78BC',  # Purple
        '#CA9161',  # Brown
        '#949494',  # Gray
        '#ECE133',  # Yellow
        '#56B4E9',  # Sky blue
    ],
}

# =============================================================================
# Theme Definitions
# =============================================================================


THEMES = {
    # Default publication theme
    'default': {
        'font.size': 11,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.figsize': (8, 6),
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        'axes.linewidth': 1.0,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'axes.grid': False,
        'axes.prop_cycle': cycler(color=PALETTES['default']),
    },

    # Prism Rain theme (moderate sizing, publication-friendly)
    'prism_rain': {
        'font.size': 11,
        'font.sans-serif': ['Source Sans Pro', 'DejaVu Sans', 'Arial'],
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.figsize': (10, 8),
        'lines.linewidth': 2.0,
        'lines.markersize': 8,
        'axes.linewidth': 1.5,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'xtick.major.size': 5.0,
        'ytick.major.size': 5.0,
        'axes.grid': False,
        'legend.frameon': False,
        'axes.prop_cycle': cycler(color=PALETTES['prism_rain']),
    },

    # Prism Rain Large
    'prism_rain_large': {
        # Background colors
        'figure.facecolor': '#ffffff',
        'axes.facecolor': '#ffffff',
        'savefig.facecolor': '#ffffff',

        # Typography - YOUR EXACT SIZES
        'font.sans-serif': ['Source Sans Pro Black', 'Source Sans 3', 'DejaVu Sans'],
        'font.size': 35,
        'axes.labelsize': 55,
        'axes.titlesize': 55,
        'axes.labelweight': 'normal',

        # Axes and ticks - YOUR EXACT VALUES
        'axes.edgecolor': '#222222',
        'axes.labelcolor': '#222222',
        'axes.linewidth': 3.5,

        'xtick.color': '#333333',
        'ytick.color': '#333333',
        'xtick.major.size': 10.0,
        'ytick.major.size': 10.0,
        'xtick.major.width': 2,
        'ytick.major.width': 2,
        'xtick.labelsize': 55,
        'ytick.labelsize': 55,
        'xtick.major.pad': 20,
        'ytick.major.pad': 20,

        # No grid
        'axes.grid': False,

        # Lines and markers
        'lines.linewidth': 6,
        'lines.markersize': 22,
        'lines.antialiased': True,

        # Legend
        'legend.frameon': False,
        'legend.fontsize': 35,
        'legend.loc': 'best',
        'legend.fancybox': True,

        # Figure size
        'figure.figsize': (20, 20),
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,

        # Text rendering
        'text.usetex': False,

        # Color cycle
        'axes.prop_cycle': cycler(color=PALETTES['prism_rain']),
    },

    # Minimal theme (clean, professional)
    'minimal': {
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.figsize': (7, 5),
        'lines.linewidth': 1.2,
        'lines.markersize': 5,
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': False,
        'legend.frameon': False,
        'axes.prop_cycle': cycler(color=PALETTES['minimal']),
    },

    # Large presentation theme (for slides)
    'presentation': {
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.figsize': (12, 9),
        'lines.linewidth': 3.0,
        'lines.markersize': 10,
        'axes.linewidth': 2.0,
        'xtick.major.width': 2.0,
        'ytick.major.width': 2.0,
        'axes.grid': False,
        'axes.prop_cycle': cycler(color=PALETTES['prism_rain']),
    },
}

# =============================================================================
# Helper Functions
# =============================================================================

def setup_publication_style(theme='default', use_scienceplots=True, dpi=300):
    """
    Setup matplotlib with publication-ready styling.

    Parameters
    ----------
    theme : str, default='default'
        Theme name. Options:
        - 'default': Publication standard (11pt fonts, 1.5pt lines)
        - 'prism_rain': High contrast moderate (12pt fonts, 2pt lines)
        - 'prism_rain_large': LARGE fonts for posters (35-55pt fonts, 6pt lines)
        - 'minimal': Clean simple (9pt fonts, 1.2pt lines)
        - 'presentation': Slides (14-18pt fonts, 3pt lines)
    use_scienceplots : bool, default=True
        If True, apply scienceplots 'science' style as base (RECOMMENDED)
    dpi : int, default=300
        DPI for saved figures (publication standard)

    Returns
    -------
    None

    Examples
    --------
    >>> setup_publication_style()  # Default publication style
    >>> setup_publication_style(theme='prism_rain')  # High contrast
    >>> setup_publication_style(theme='prism_rain_large')  # LARGE fonts for posters
    >>> setup_publication_style(use_scienceplots=True)  # With scienceplots

    Notes
    -----
    This function modifies global matplotlib rcParams. Call once at the
    beginning of your script before creating figures.
    """
    # Reset to defaults first
    plt.rcdefaults()

    # Apply scienceplots base if requested
    if use_scienceplots:
        try:
            # Try standard style name first
            plt.style.use('science')
        except OSError:
            # Fallback: try direct path to scienceplots style file
            try:
                import scienceplots
                import os
                sp_style_path = os.path.join(
                    os.path.dirname(scienceplots.__file__),
                    'styles',
                    'science.mplstyle'
                )
                if os.path.exists(sp_style_path):
                    plt.style.use(sp_style_path)
                else:
                    warnings.warn(
                        "scienceplots style file not found. "
                        "Install with: pip install scienceplots"
                    )
            except ImportError:
                warnings.warn(
                    "scienceplots package not found. "
                    "Install with: pip install scienceplots"
                )

    # Load matplotlibrc defaults
    matplotlibrc_path = Path(__file__).parent / 'matplotlibrc'
    if matplotlibrc_path.exists():
        plt.style.use(str(matplotlibrc_path))
    else:
        warnings.warn(f"matplotlibrc not found at {matplotlibrc_path}")

    # Apply theme-specific settings
    if theme not in THEMES:
        available = ', '.join(THEMES.keys())
        raise ValueError(
            f"Theme '{theme}' not found. Available themes: {available}"
        )

    plt.rcParams.update(THEMES[theme])

    # Set publication DPI
    plt.rcParams['savefig.dpi'] = dpi
    plt.rcParams['figure.dpi'] = 100  # Screen display

    # Ensure tight layout and good spacing
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.1

    print(f"✓ Applied '{theme}' publication style (DPI: {dpi})")


def get_color_cycle(palette='default', n_colors=None):
    """
    Get a list of colors from a palette.

    Parameters
    ----------
    palette : str, default='default'
        Palette name from PALETTES
    n_colors : int, optional
        Number of colors to return. If None, returns all colors in palette.

    Returns
    -------
    list of str
        List of hex color codes

    Examples
    --------
    >>> colors = get_color_cycle('prism_rain', n_colors=5)
    >>> plt.plot(x, y, color=colors[0])
    """
    if palette not in PALETTES:
        available = ', '.join(PALETTES.keys())
        raise ValueError(
            f"Palette '{palette}' not found. Available: {available}"
        )

    colors = PALETTES[palette]

    if n_colors is None:
        return colors
    else:
        # Cycle through colors if n_colors > palette length
        return [colors[i % len(colors)] for i in range(n_colors)]


def save_figure(fig, filepath, dpi=300, formats=None, **kwargs):
    """
    Save figure in multiple formats with publication settings.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save
    filepath : str or Path
        Output path (without extension)
    dpi : int, default=300
        Resolution for raster formats
    formats : list of str, optional
        Output formats. Default: ['png', 'pdf']
    **kwargs
        Additional arguments passed to fig.savefig()

    Returns
    -------
    list of Path
        Paths to saved files

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3], [1, 4, 9])
    >>> save_figure(fig, 'my_plot', formats=['png', 'svg', 'pdf'])
    """
    if formats is None:
        formats = ['png', 'pdf']

    filepath = Path(filepath)
    saved_files = []

    # Default save arguments
    save_kwargs = {
        'dpi': dpi,
        'bbox_inches': 'tight',
        'facecolor': 'white',
        'edgecolor': 'none',
    }
    save_kwargs.update(kwargs)

    for fmt in formats:
        output_path = filepath.with_suffix(f'.{fmt}')
        fig.savefig(output_path, format=fmt, **save_kwargs)
        saved_files.append(output_path)
        print(f"Saved: {output_path}")

    return saved_files


def apply_grid(ax, which='both', alpha=0.3, linestyle='--', linewidth=0.5):
    """
    Apply a subtle grid to axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to modify
    which : {'major', 'minor', 'both'}, default='both'
        Which grid lines to draw
    alpha : float, default=0.3
        Grid transparency
    linestyle : str, default='--'
        Grid line style
    linewidth : float, default=0.5
        Grid line width

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3], [1, 4, 9])
    >>> apply_grid(ax)
    """
    ax.grid(True, which=which, alpha=alpha, linestyle=linestyle,
            linewidth=linewidth)


def set_spine_visibility(ax, top=False, right=False, left=True, bottom=True):
    """
    Control visibility of axis spines (box around plot).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to modify
    top, right, left, bottom : bool
        Visibility of each spine

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> set_spine_visibility(ax, top=False, right=False)  # Minimal style
    """
    ax.spines['top'].set_visible(top)
    ax.spines['right'].set_visible(right)
    ax.spines['left'].set_visible(left)
    ax.spines['bottom'].set_visible(bottom)


# =============================================================================
# Backward Compatibility (for your existing styles.py)
# =============================================================================

# Alias for backward compatibility
set_plot_style = setup_publication_style


# Export common settings for direct access
COMMON_RC = {
    "text.usetex": False,
    "lines.markersize": 6,
    "legend.fontsize": 10,
    "axes.grid": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
}


# =============================================================================
# Context Managers (Advanced Usage)
# =============================================================================

class PlotStyle:
    """
    Context manager for temporary style changes.

    Examples
    --------
    >>> with PlotStyle('prism_rain'):
    ...     fig, ax = plt.subplots()
    ...     ax.plot([1, 2, 3], [1, 4, 9])
    ...     plt.show()
    """
    def __init__(self, theme='default', **kwargs):
        self.theme = theme
        self.kwargs = kwargs
        self.original_params = None

    def __enter__(self):
        # Save current settings
        self.original_params = plt.rcParams.copy()
        # Apply new style
        setup_publication_style(theme=self.theme, **self.kwargs)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original settings
        plt.rcParams.update(self.original_params)


# =============================================================================
# Quick Setup for Scripts
# =============================================================================

def quick_setup(theme='default'):
    """
    Quick setup for publication plots with sensible defaults.

    This is a convenience function that sets up matplotlib with
    publication-ready defaults. Use at the start of your script.

    Parameters
    ----------
    theme : str, default='default'
        Theme to apply

    Examples
    --------
    >>> from src.ploting.plotting_config import quick_setup
    >>> quick_setup('prism_rain')
    >>> # Now create your plots
    """
    setup_publication_style(theme=theme, dpi=300)
    # Additional common settings
    plt.rcParams['figure.autolayout'] = False
    plt.rcParams['figure.constrained_layout.use'] = True
    return plt


if __name__ == '__main__':
    # Demo: Show all available themes
    print("Available themes:")
    for theme_name in THEMES.keys():
        print(f"  - {theme_name}")

    print("\nAvailable palettes:")
    for palette_name in PALETTES.keys():
        print(f"  - {palette_name}")

    print("\nExample usage:")
    print("  from src.ploting.plotting_config import setup_publication_style")
    print("  setup_publication_style('prism_rain')")
    print("  fig, ax = plt.subplots()")
    print("  ax.plot([1, 2, 3], [1, 4, 9])")
    print("  plt.show()")
