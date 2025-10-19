#!/usr/bin/env python3
"""
Test script to validate IntermediateParameters and preprocessing script.
Tests both valid and invalid parameter configurations.
"""

import sys
from pathlib import Path
from pydantic import ValidationError

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.parameters import IntermediateParameters


def test_valid_config():
    """Test that valid config loads successfully."""
    print("=" * 80)
    print("TEST 1: Valid configuration")
    print("=" * 80)

    try:
        params = IntermediateParameters(
            stage_root=Path("data/02_stage/raw_measurements"),
            output_root=Path("data/03_intermediate"),
            procedure="IV",
            voltage_col="Vsd (V)",
            dv_threshold=0.001,
            min_segment_points=5,
            workers=8,
            polars_threads=2,
            force=False,
        )

        print("✅ Valid config loaded successfully!")
        print(f"\n📋 Parameters:")
        print(f"   Stage root: {params.stage_root}")
        print(f"   Output dir: {params.get_output_dir()}")
        print(f"   Procedure: {params.procedure}")
        print(f"   Voltage column: {params.voltage_col}")
        print(f"   dV threshold: {params.dv_threshold}")
        print(f"   Min segment points: {params.min_segment_points}")
        print(f"   Workers: {params.workers}")
        print(f"   Polars threads: {params.polars_threads}")
        print(f"   Force: {params.force}")

        return True

    except ValidationError as e:
        print(f"❌ UNEXPECTED: Valid config failed validation!")
        print(e)
        return False


def test_invalid_procedure():
    """Test that invalid procedure is rejected."""
    print("\n" + "=" * 80)
    print("TEST 2: Invalid procedure name")
    print("=" * 80)

    try:
        params = IntermediateParameters(
            stage_root=Path("data/02_stage/raw_measurements"),
            output_root=Path("data/03_intermediate"),
            procedure="INVALID_PROCEDURE",  # ❌ Not in allowed list
            voltage_col="Vsd (V)",
            workers=8,
        )

        print(f"❌ FAILED: Invalid procedure should have been rejected!")
        print(f"   Got: {params.procedure}")
        return False

    except ValidationError as e:
        print(f"✅ Correctly rejected invalid procedure!")
        print(f"   Error: {str(e.errors()[0]['msg'])}")
        return True


def test_invalid_threshold():
    """Test that out-of-range dv_threshold is rejected."""
    print("\n" + "=" * 80)
    print("TEST 3: Invalid dv_threshold (out of range)")
    print("=" * 80)

    try:
        params = IntermediateParameters(
            stage_root=Path("data/02_stage/raw_measurements"),
            output_root=Path("data/03_intermediate"),
            procedure="IV",
            dv_threshold=5.0,  # ❌ Must be 0.0 <= x <= 1.0
            workers=8,
        )

        print(f"❌ FAILED: Out-of-range threshold should have been rejected!")
        print(f"   Got: {params.dv_threshold}")
        return False

    except ValidationError as e:
        print(f"✅ Correctly rejected out-of-range threshold!")
        print(f"   Error: {str(e.errors()[0]['msg'])}")
        return True


def test_invalid_min_points():
    """Test that invalid min_segment_points is rejected."""
    print("\n" + "=" * 80)
    print("TEST 4: Invalid min_segment_points (too low)")
    print("=" * 80)

    try:
        params = IntermediateParameters(
            stage_root=Path("data/02_stage/raw_measurements"),
            output_root=Path("data/03_intermediate"),
            procedure="IV",
            min_segment_points=1,  # ❌ Must be >= 2
            workers=8,
        )

        print(f"❌ FAILED: Invalid min_segment_points should have been rejected!")
        print(f"   Got: {params.min_segment_points}")
        return False

    except ValidationError as e:
        print(f"✅ Correctly rejected invalid min_segment_points!")
        print(f"   Error: {str(e.errors()[0]['msg'])}")
        return True


def test_invalid_workers():
    """Test that invalid worker count is rejected."""
    print("\n" + "=" * 80)
    print("TEST 5: Invalid workers count (too high)")
    print("=" * 80)

    try:
        params = IntermediateParameters(
            stage_root=Path("data/02_stage/raw_measurements"),
            output_root=Path("data/03_intermediate"),
            procedure="IV",
            workers=100,  # ❌ Must be <= 32
        )

        print(f"❌ FAILED: Invalid workers count should have been rejected!")
        print(f"   Got: {params.workers}")
        return False

    except ValidationError as e:
        print(f"✅ Correctly rejected invalid workers count!")
        print(f"   Error: {str(e.errors()[0]['msg'])}")
        return True


def test_json_config_loading():
    """Test loading from JSON config file."""
    print("\n" + "=" * 80)
    print("TEST 6: Load from JSON config file")
    print("=" * 80)

    config_path = Path("config/examples/intermediate_config.json")

    if not config_path.exists():
        print(f"⚠️  SKIPPED: Config file not found: {config_path}")
        return None

    try:
        params = IntermediateParameters.model_validate_json(config_path.read_text())

        print("✅ JSON config loaded successfully!")
        print(f"\n📋 Loaded parameters:")
        print(f"   Stage root: {params.stage_root}")
        print(f"   Output dir: {params.get_output_dir()}")
        print(f"   Procedure: {params.procedure}")
        print(f"   Workers: {params.workers}")

        return True

    except ValidationError as e:
        print(f"❌ FAILED: JSON config failed validation!")
        print(e)
        return False
    except Exception as e:
        print(f"❌ FAILED: Error loading JSON config!")
        print(e)
        return False


def test_helper_methods():
    """Test helper methods like get_output_dir()."""
    print("\n" + "=" * 80)
    print("TEST 7: Helper method get_output_dir()")
    print("=" * 80)

    try:
        # Test IV procedure
        params_iv = IntermediateParameters(
            stage_root=Path("data/02_stage/raw_measurements"),
            output_root=Path("data/03_intermediate"),
            procedure="IV",
            workers=8,
        )

        output_dir = params_iv.get_output_dir()
        expected = Path("data/03_intermediate/iv_segments")

        if output_dir == expected:
            print(f"✅ IV procedure output dir correct!")
            print(f"   Expected: {expected}")
            print(f"   Got: {output_dir}")
        else:
            print(f"❌ FAILED: IV output dir mismatch!")
            print(f"   Expected: {expected}")
            print(f"   Got: {output_dir}")
            return False

        # Test IVg procedure
        params_ivg = IntermediateParameters(
            stage_root=Path("data/02_stage/raw_measurements"),
            output_root=Path("data/03_intermediate"),
            procedure="IVg",
            workers=8,
        )

        output_dir_ivg = params_ivg.get_output_dir()
        expected_ivg = Path("data/03_intermediate/ivg_segments")

        if output_dir_ivg == expected_ivg:
            print(f"✅ IVg procedure output dir correct!")
            print(f"   Expected: {expected_ivg}")
            print(f"   Got: {output_dir_ivg}")
        else:
            print(f"❌ FAILED: IVg output dir mismatch!")
            print(f"   Expected: {expected_ivg}")
            print(f"   Got: {output_dir_ivg}")
            return False

        return True

    except Exception as e:
        print(f"❌ FAILED: Helper method test failed!")
        print(e)
        return False


def main():
    """Run all validation tests."""
    print("\n" + "🧪" * 40)
    print("INTERMEDIATE PREPROCESSING PARAMETER VALIDATION TESTS")
    print("🧪" * 40 + "\n")

    results = []

    # Run all tests
    results.append(("Valid config", test_valid_config()))
    results.append(("Invalid procedure", test_invalid_procedure()))
    results.append(("Invalid threshold", test_invalid_threshold()))
    results.append(("Invalid min_points", test_invalid_min_points()))
    results.append(("Invalid workers", test_invalid_workers()))
    results.append(("JSON config loading", test_json_config_loading()))
    results.append(("Helper methods", test_helper_methods()))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, result in results if result is True)
    failed = sum(1 for _, result in results if result is False)
    skipped = sum(1 for _, result in results if result is None)
    total = len(results)

    for test_name, result in results:
        if result is True:
            status = "✅ PASS"
        elif result is False:
            status = "❌ FAIL"
        else:
            status = "⚠️  SKIP"

        print(f"{status} - {test_name}")

    print(f"\n📊 Results: {passed}/{total} passed, {failed} failed, {skipped} skipped")

    if failed == 0 and passed > 0:
        print("\n🎉 All validation tests passed!")
        return 0
    else:
        print(f"\n⚠️  Some tests failed or were skipped")
        return 1


if __name__ == "__main__":
    sys.exit(main())
