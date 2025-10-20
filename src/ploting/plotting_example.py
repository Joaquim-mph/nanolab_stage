#!/usr/bin/env python3
"""
Plotting Configuration Examples

Demonstrates how to use the publication-ready plotting configuration
for creating consistent, high-quality scientific figures.

Usage:
    python src/ploting/plotting_example.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import the plotting configuration
from plotting_config import (
    setup_publication_style,
    save_figure,
    get_color_cycle,
    apply_grid,
    set_spine_visibility,
    PlotStyle,
    PALETTES,
)


def example_1_basic_plot():
    """Example 1: Basic line plot with default style"""
    print("\n" + "="*70)
    print("Example 1: Basic Publication Plot")
    print("="*70)

    # Setup publication style
    setup_publication_style(theme='default')

    # Generate data
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(x, y1, label='sin(x)', linewidth=2)
    ax.plot(x, y2, label='cos(x)', linewidth=2)

    ax.set_xlabel('X-axis (units)')
    ax.set_ylabel('Y-axis (units)')
    ax.set_title('Basic Publication Plot')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Save in multiple formats
    output_dir = Path('plots/examples')
    output_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, output_dir / 'example_1_basic', formats=['png', 'pdf'])

    plt.close()


def example_2_prism_rain_theme():
    """Example 2: Using prism_rain theme with multiple datasets"""
    print("\n" + "="*70)
    print("Example 2: Prism Rain Theme")
    print("="*70)

    # Setup prism_rain style
    setup_publication_style(theme='prism_rain')

    # Generate data
    x = np.linspace(0, 10, 100)
    n_datasets = 5

    fig, ax = plt.subplots(figsize=(10, 7))

    for i in range(n_datasets):
        y = np.sin(x + i * np.pi/4) * (1 + i * 0.2)
        ax.plot(x, y, label=f'Dataset {i+1}', linewidth=2.5)

    ax.set_xlabel('Voltage (V)', fontweight='bold')
    ax.set_ylabel('Current (A)', fontweight='bold')
    ax.set_title('Multiple Datasets - Prism Rain Theme', fontweight='bold')
    ax.legend(loc='upper right')

    # Save
    output_dir = Path('plots/examples')
    save_figure(fig, output_dir / 'example_2_prism_rain', formats=['png', 'svg'])

    plt.close()


def example_3_minimal_style():
    """Example 3: Minimal style with clean aesthetics"""
    print("\n" + "="*70)
    print("Example 3: Minimal Clean Style")
    print("="*70)

    # Setup minimal style
    setup_publication_style(theme='minimal')

    # Generate data
    x = np.linspace(0, 10, 100)
    y = x**2
    yerr = 10 * np.random.rand(100)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.errorbar(x, y, yerr=yerr, fmt='o', markersize=4,
                capsize=3, alpha=0.7, label='Data with error bars')

    # Fit line
    coeffs = np.polyfit(x, y, 2)
    y_fit = np.polyval(coeffs, x)
    ax.plot(x, y_fit, '--', linewidth=2, label='Quadratic fit', alpha=0.8)

    ax.set_xlabel('X (units)')
    ax.set_ylabel('Y (units)')
    ax.set_title('Error Bars with Fit - Minimal Style')
    ax.legend()

    # Minimal style: no top/right spines
    set_spine_visibility(ax, top=False, right=False)

    # Save
    output_dir = Path('plots/examples')
    save_figure(fig, output_dir / 'example_3_minimal', dpi=300)

    plt.close()


def example_4_subplots():
    """Example 4: Multi-panel figure with shared axes"""
    print("\n" + "="*70)
    print("Example 4: Multi-Panel Figure")
    print("="*70)

    setup_publication_style(theme='default')

    # Generate data
    x = np.linspace(0, 10, 100)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Multi-Panel Scientific Figure', fontsize=14, fontweight='bold')

    # Plot 1: Linear
    axes[0, 0].plot(x, x, linewidth=2)
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    axes[0, 0].set_title('(a) Linear')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Quadratic
    axes[0, 1].plot(x, x**2, linewidth=2, color='C1')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Y')
    axes[0, 1].set_title('(b) Quadratic')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Exponential
    axes[1, 0].semilogy(x, np.exp(x/3), linewidth=2, color='C2')
    axes[1, 0].set_xlabel('X')
    axes[1, 0].set_ylabel('Y (log scale)')
    axes[1, 0].set_title('(c) Exponential')
    axes[1, 0].grid(True, alpha=0.3, which='both')

    # Plot 4: Scatter
    y_scatter = x + np.random.randn(100) * 0.5
    axes[1, 1].scatter(x, y_scatter, alpha=0.6, s=30, color='C3')
    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel('Y')
    axes[1, 1].set_title('(d) Scatter with Noise')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    output_dir = Path('plots/examples')
    save_figure(fig, output_dir / 'example_4_multipanel', formats=['png', 'pdf'])

    plt.close()


def example_5_color_cycles():
    """Example 5: Demonstrating different color palettes"""
    print("\n" + "="*70)
    print("Example 5: Color Palette Comparison")
    print("="*70)

    palettes_to_show = ['default', 'prism_rain', 'minimal', 'scientific']

    x = np.linspace(0, 10, 100)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Color Palette Comparison', fontsize=14, fontweight='bold')

    axes_flat = axes.flatten()

    for idx, palette_name in enumerate(palettes_to_show):
        ax = axes_flat[idx]
        colors = get_color_cycle(palette_name, n_colors=8)

        for i, color in enumerate(colors):
            y = np.sin(x + i * np.pi/4)
            ax.plot(x, y + i*0.5, color=color, linewidth=2,
                   label=f'Line {i+1}')

        ax.set_xlabel('X')
        ax.set_ylabel('Y (offset)')
        ax.set_title(f'{palette_name.title()} Palette')
        ax.legend(ncol=2, fontsize=8)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()

    # Save
    output_dir = Path('plots/examples')
    save_figure(fig, output_dir / 'example_5_palettes', formats=['png'])

    plt.close()


def example_6_context_manager():
    """Example 6: Using context manager for temporary style"""
    print("\n" + "="*70)
    print("Example 6: Context Manager Usage")
    print("="*70)

    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    # Create figure with default style
    setup_publication_style(theme='default')
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(x, y, linewidth=2)
    ax1.set_title('Default Style')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')

    output_dir = Path('plots/examples')
    save_figure(fig1, output_dir / 'example_6a_default', formats=['png'])
    plt.close()

    # Use context manager for prism_rain style temporarily
    with PlotStyle('prism_rain'):
        fig2, ax2 = plt.subplots(figsize=(10, 7))
        ax2.plot(x, y, linewidth=3)
        ax2.set_title('Prism Rain Style (Temporary)')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        save_figure(fig2, output_dir / 'example_6b_prism', formats=['png'])
        plt.close()

    # Back to previous style
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    ax3.plot(x, y, linewidth=2)
    ax3.set_title('Back to Default Style')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    save_figure(fig3, output_dir / 'example_6c_back', formats=['png'])
    plt.close()


def example_7_presentation_style():
    """Example 7: Large presentation-ready plot"""
    print("\n" + "="*70)
    print("Example 7: Presentation Style (Large Text)")
    print("="*70)

    # Setup presentation style (large fonts)
    setup_publication_style(theme='presentation')

    x = np.linspace(-5, 5, 200)
    y1 = np.exp(-x**2/2) / np.sqrt(2*np.pi)
    y2 = 0.5 * np.exp(-(x-1)**2/2) / np.sqrt(2*np.pi)

    fig, ax = plt.subplots(figsize=(14, 10))

    ax.plot(x, y1, linewidth=4, label='Standard Normal')
    ax.plot(x, y2, linewidth=4, label='Shifted Normal')
    ax.fill_between(x, 0, y1, alpha=0.2)

    ax.set_xlabel('X Value')
    ax.set_ylabel('Probability Density')
    ax.set_title('Gaussian Distributions - Presentation Style')
    ax.legend(fontsize=16)
    ax.grid(True, alpha=0.3)

    # Save
    output_dir = Path('plots/examples')
    save_figure(fig, output_dir / 'example_7_presentation', formats=['png', 'pdf'])

    plt.close()


def example_8_iv_style_realistic():
    """Example 8: Realistic IV curve with hysteresis"""
    print("\n" + "="*70)
    print("Example 8: Realistic IV Curve")
    print("="*70)

    setup_publication_style(theme='prism_rain')

    # Simulate IV curve with hysteresis
    V_fwd = np.linspace(-8, 8, 100)
    V_ret = np.linspace(8, -8, 100)

    # Forward: higher current (less resistance)
    I_fwd = 1e-9 * (V_fwd + 0.1 * V_fwd**3 + np.random.randn(100) * 0.05)

    # Return: lower current (higher resistance, hysteresis)
    I_ret = 1e-9 * (V_ret + 0.1 * V_ret**3 - 0.3 * V_ret + np.random.randn(100) * 0.05)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Linear scale
    ax1.plot(V_fwd, I_fwd * 1e9, 'o-', label='Forward', markersize=4, linewidth=2)
    ax1.plot(V_ret, I_ret * 1e9, 's-', label='Return', markersize=4, linewidth=2)
    ax1.set_xlabel('Voltage V$_{sd}$ (V)', fontweight='bold')
    ax1.set_ylabel('Current I (nA)', fontweight='bold')
    ax1.set_title('(a) IV Curve - Linear Scale', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='k', linewidth=0.8, linestyle='--', alpha=0.5)
    ax1.axvline(0, color='k', linewidth=0.8, linestyle='--', alpha=0.5)

    # Log scale (absolute value)
    I_fwd_abs = np.abs(I_fwd)
    I_ret_abs = np.abs(I_ret)

    ax2.semilogy(V_fwd, I_fwd_abs, 'o-', label='Forward', markersize=4, linewidth=2)
    ax2.semilogy(V_ret, I_ret_abs, 's-', label='Return', markersize=4, linewidth=2)
    ax2.set_xlabel('Voltage V$_{sd}$ (V)', fontweight='bold')
    ax2.set_ylabel('|Current| (A)', fontweight='bold')
    ax2.set_title('(b) IV Curve - Log Scale', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    # Save
    output_dir = Path('plots/examples')
    save_figure(fig, output_dir / 'example_8_iv_realistic', formats=['png', 'pdf'])

    plt.close()


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("PLOTTING CONFIGURATION EXAMPLES")
    print("="*70)
    print("\nGenerating publication-ready plot examples...")
    print("Output directory: plots/examples/\n")

    # Run all examples
    example_1_basic_plot()
    example_2_prism_rain_theme()
    example_3_minimal_style()
    example_4_subplots()
    example_5_color_cycles()
    example_6_context_manager()
    example_7_presentation_style()
    example_8_iv_style_realistic()

    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70)
    print(f"\nCheck 'plots/examples/' for generated figures")
    print("\nGenerated files:")

    output_dir = Path('plots/examples')
    if output_dir.exists():
        for file in sorted(output_dir.glob('*')):
            print(f"  • {file.name}")

    print("\nUsage in your scripts:")
    print("  from src.ploting.plotting_config import setup_publication_style")
    print("  setup_publication_style('prism_rain')")
    print("  # ... create your plots ...")
    print()


if __name__ == '__main__':
    main()
