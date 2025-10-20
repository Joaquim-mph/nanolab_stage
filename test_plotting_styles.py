#!/usr/bin/env python3
"""
Test script to verify plotting styles work correctly.

Tests:
1. Legacy styles.py works
2. New plotting_config.py works
3. prism_rain_large has exact parameters from original styles.py
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add paths
_root = Path(__file__).parent
sys.path.insert(0, str(_root))  # For styles.py in root
sys.path.insert(0, str(_root / 'src' / 'ploting'))  # For plotting_config

def test_legacy_style():
    """Test legacy styles.py"""
    print("\n" + "="*70)
    print("TEST 1: Legacy styles.py")
    print("="*70)

    from styles import set_plot_style
    set_plot_style("prism_rain")

    # Check key parameters
    params_to_check = {
        'font.size': 35,
        'axes.labelsize': 55,
        'axes.titlesize': 55,
        'xtick.labelsize': 55,
        'ytick.labelsize': 55,
        'legend.fontsize': 35,
        'lines.linewidth': 6,
        'lines.markersize': 22,
        'axes.linewidth': 3.5,
        'figure.figsize': (20, 20),
    }

    print("\nChecking parameters:")
    all_match = True
    for param, expected in params_to_check.items():
        actual = plt.rcParams[param]
        # Special handling for figure.figsize (can be list or tuple)
        if param == 'figure.figsize':
            match = tuple(actual) == expected
        else:
            match = actual == expected

        status = "✓" if match else "✗"
        if not match:
            all_match = False
        print(f"  {status} {param:20s}: {actual} (expected: {expected})")

    if all_match:
        print("\n✓ ALL PARAMETERS MATCH!")
    else:
        print("\n✗ Some parameters don't match")

    plt.rcdefaults()  # Reset for next test
    return all_match


def test_new_system_prism_rain_large():
    """Test new plotting_config.py with prism_rain_large theme"""
    print("\n" + "="*70)
    print("TEST 2: New system with prism_rain_large")
    print("="*70)

    from plotting_config import setup_publication_style
    setup_publication_style(theme='prism_rain_large')

    # Check same parameters
    params_to_check = {
        'font.size': 35,
        'axes.labelsize': 55,
        'axes.titlesize': 55,
        'xtick.labelsize': 55,
        'ytick.labelsize': 55,
        'legend.fontsize': 35,
        'lines.linewidth': 6,
        'lines.markersize': 22,
        'axes.linewidth': 3.5,
        'figure.figsize': (20, 20),
    }

    print("\nChecking parameters:")
    all_match = True
    for param, expected in params_to_check.items():
        actual = plt.rcParams[param]
        # Special handling for figure.figsize (can be list or tuple)
        if param == 'figure.figsize':
            match = tuple(actual) == expected
        else:
            match = actual == expected

        status = "✓" if match else "✗"
        if not match:
            all_match = False
        print(f"  {status} {param:20s}: {actual} (expected: {expected})")

    if all_match:
        print("\n✓ ALL PARAMETERS MATCH!")
    else:
        print("\n✗ Some parameters don't match")

    plt.rcdefaults()  # Reset for next test
    return all_match


def test_other_themes():
    """Test other available themes"""
    print("\n" + "="*70)
    print("TEST 3: Other themes")
    print("="*70)

    from plotting_config import setup_publication_style, THEMES

    print(f"\nAvailable themes: {list(THEMES.keys())}")

    for theme_name in ['default', 'prism_rain', 'minimal', 'presentation']:
        print(f"\nTesting theme: {theme_name}")
        try:
            setup_publication_style(theme=theme_name)
            print(f"  ✓ Theme '{theme_name}' loaded successfully")
            print(f"    Font size: {plt.rcParams['font.size']}")
            print(f"    Figure size: {plt.rcParams['figure.figsize']}")
            plt.rcdefaults()
        except Exception as e:
            print(f"  ✗ Theme '{theme_name}' failed: {e}")
            return False

    return True


def test_backward_compatibility():
    """Test that old code still works"""
    print("\n" + "="*70)
    print("TEST 4: Backward compatibility")
    print("="*70)

    print("\nTesting: from styles import set_plot_style")
    try:
        from styles import set_plot_style
        set_plot_style("prism_rain")
        print("✓ Legacy import works")

        # Verify it applied large params
        font_size = plt.rcParams['font.size']
        expected = 35
        if font_size == expected:
            print(f"✓ Applied correct parameters (font.size = {font_size})")
            plt.rcdefaults()
            return True
        else:
            print(f"✗ Font size mismatch: {font_size} != {expected}")
            plt.rcdefaults()
            return False

    except Exception as e:
        print(f"✗ Legacy import failed: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("PLOTTING STYLES TEST SUITE")
    print("="*70)

    results = {
        'Legacy styles.py': test_legacy_style(),
        'New prism_rain_large': test_new_system_prism_rain_large(),
        'Other themes': test_other_themes(),
        'Backward compatibility': test_backward_compatibility(),
    }

    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8s} - {test_name}")
        if not passed:
            all_passed = False

    print("="*70)

    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nYour exact prism_rain parameters (35pt, 55pt fonts, etc.)")
        print("are now available as 'prism_rain_large' theme.")
        print("\nUsage:")
        print("  # Option 1: Legacy (automatically uses new system)")
        print("  from styles import set_plot_style")
        print("  set_plot_style('prism_rain')")
        print()
        print("  # Option 2: New system (recommended)")
        print("  from src.ploting.plotting_config import setup_publication_style")
        print("  setup_publication_style('prism_rain_large')")
    else:
        print("\n❌ SOME TESTS FAILED")
        print("Check output above for details")
        sys.exit(1)


if __name__ == '__main__':
    main()
