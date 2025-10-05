import matplotlib.pyplot as plt 
# Common settings to apply to all styles
import matplotlib.pyplot as plt

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
    # Primary vibrant colors (most used)
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#a65628", "#f781bf", "#999999",
    # Secondary soft colors
    "#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3",
    # Tertiary accent colors
    "#a6d854", "#ffd92f", "#e5c494", "#b3b3b3",
    # Additional deep colors
    "#1b9e77", "#d95f02", "#7570b3", "#e7298a"
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
            
            # Typography
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 11.5,
            "axes.labelsize": 13,
            "axes.titlesize": 14.5,
            "axes.labelweight": "normal",
            
            # Axes and ticks
            "axes.edgecolor": "#222222",
            "axes.labelcolor": "#222222",
            "axes.linewidth": 1.2,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            
            # Grid
            "grid.color": "#cccccc",
            "grid.linestyle": "--",
            "grid.linewidth": 0.4,
            "grid.alpha": 0.6,
            
            # Lines and markers
            "lines.linewidth": 1.8,
            "lines.markersize": 6,
            "lines.antialiased": True,
            
            # Legend
            "legend.frameon": False,
            "legend.fontsize": 11,
            "legend.loc": "best",
            "legend.fancybox": False,
            
            # Figure size (optimized for papers)
            "figure.figsize": (8.6, 4.6),
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
def apply_theme(theme_name="prism_rain"):
    """Apply a publication-ready matplotlib theme.
    
    Parameters
    ----------
    theme_name : str, default="prism_rain"
        Name of the theme to apply
        
    Example
    -------
    >>> apply_theme("prism_rain")
    >>> plt.plot([1, 2, 3], [1, 4, 9])
    >>> plt.show()
    """
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