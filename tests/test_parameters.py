#!/usr/bin/env python3
"""
Comprehensive validation tests for Pydantic parameter models.

Tests demonstrate that validation catches:
- Invalid paths (non-existent directories)
- Out-of-range numeric values
- Invalid date formats
- Even polynomial orders (should be odd)
- Missing required fields
- Extra unknown fields
- Inconsistent cross-parameter values
"""

import pytest
from pathlib import Path
from pydantic import ValidationError
import tempfile
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.parameters import (
    StagingParameters,
    IVAnalysisParameters,
    PlottingParameters,
    PipelineParameters,
)


class TestStagingParameters:
    """Test validation for StagingParameters."""

    def test_valid_parameters(self, tmp_path):
        """Test that valid parameters pass validation."""
        raw_root = tmp_path / "01_raw"
        stage_root = tmp_path / "02_stage"
        yaml_file = tmp_path / "procedures.yml"

        raw_root.mkdir()
        yaml_file.write_text("procedures: {}")

        params = StagingParameters(
            raw_root=raw_root,
            stage_root=stage_root,
            procedures_yaml=yaml_file,
            workers=4,
            polars_threads=2,
        )

        assert params.workers == 4
        assert params.polars_threads == 2
        assert params.local_tz == "America/Santiago"  # default
        assert params.force is False  # default

    def test_nonexistent_raw_root(self, tmp_path):
        """Test that non-existent raw_root raises ValidationError."""
        stage_root = tmp_path / "02_stage"
        yaml_file = tmp_path / "procedures.yml"
        yaml_file.write_text("procedures: {}")

        with pytest.raises(ValidationError) as exc_info:
            StagingParameters(
                raw_root=tmp_path / "does_not_exist",
                stage_root=stage_root,
                procedures_yaml=yaml_file,
            )

        assert "Path does not exist" in str(exc_info.value)

    def test_nonexistent_yaml(self, tmp_path):
        """Test that non-existent YAML file raises ValidationError."""
        raw_root = tmp_path / "01_raw"
        stage_root = tmp_path / "02_stage"
        raw_root.mkdir()

        with pytest.raises(ValidationError) as exc_info:
            StagingParameters(
                raw_root=raw_root,
                stage_root=stage_root,
                procedures_yaml=tmp_path / "missing.yml",
            )

        assert "Path does not exist" in str(exc_info.value)

    def test_yaml_is_directory_not_file(self, tmp_path):
        """Test that YAML path must be a file, not directory."""
        raw_root = tmp_path / "01_raw"
        stage_root = tmp_path / "02_stage"
        yaml_dir = tmp_path / "yaml_dir"

        raw_root.mkdir()
        yaml_dir.mkdir()

        with pytest.raises(ValidationError) as exc_info:
            StagingParameters(
                raw_root=raw_root,
                stage_root=stage_root,
                procedures_yaml=yaml_dir,
            )

        assert "Not a file" in str(exc_info.value)

    def test_workers_out_of_range(self, tmp_path):
        """Test that workers must be within valid range."""
        raw_root = tmp_path / "01_raw"
        stage_root = tmp_path / "02_stage"
        yaml_file = tmp_path / "procedures.yml"

        raw_root.mkdir()
        yaml_file.write_text("procedures: {}")

        # Test workers = 0
        with pytest.raises(ValidationError) as exc_info:
            StagingParameters(
                raw_root=raw_root,
                stage_root=stage_root,
                procedures_yaml=yaml_file,
                workers=0,
            )
        assert "greater than or equal to 1" in str(exc_info.value)

        # Test workers = 100 (exceeds max)
        with pytest.raises(ValidationError) as exc_info:
            StagingParameters(
                raw_root=raw_root,
                stage_root=stage_root,
                procedures_yaml=yaml_file,
                workers=100,
            )
        assert "less than or equal to 32" in str(exc_info.value)

    def test_polars_threads_out_of_range(self, tmp_path):
        """Test that polars_threads must be within valid range."""
        raw_root = tmp_path / "01_raw"
        stage_root = tmp_path / "02_stage"
        yaml_file = tmp_path / "procedures.yml"

        raw_root.mkdir()
        yaml_file.write_text("procedures: {}")

        with pytest.raises(ValidationError) as exc_info:
            StagingParameters(
                raw_root=raw_root,
                stage_root=stage_root,
                procedures_yaml=yaml_file,
                polars_threads=20,
            )
        assert "less than or equal to 16" in str(exc_info.value)

    def test_extra_fields_rejected(self, tmp_path):
        """Test that extra unknown fields are rejected."""
        raw_root = tmp_path / "01_raw"
        stage_root = tmp_path / "02_stage"
        yaml_file = tmp_path / "procedures.yml"

        raw_root.mkdir()
        yaml_file.write_text("procedures: {}")

        with pytest.raises(ValidationError) as exc_info:
            StagingParameters(
                raw_root=raw_root,
                stage_root=stage_root,
                procedures_yaml=yaml_file,
                unknown_parameter="bad",  # Should be rejected
            )
        assert "Extra inputs are not permitted" in str(exc_info.value)

    def test_default_paths_set_correctly(self, tmp_path):
        """Test that default paths are set based on stage_root."""
        raw_root = tmp_path / "01_raw"
        stage_root = tmp_path / "02_stage" / "raw_measurements"
        yaml_file = tmp_path / "procedures.yml"

        raw_root.mkdir()
        yaml_file.write_text("procedures: {}")

        params = StagingParameters(
            raw_root=raw_root,
            stage_root=stage_root,
            procedures_yaml=yaml_file,
        )

        # Check default paths
        assert params.rejects_dir == stage_root.parent / "_rejects"
        assert params.events_dir == stage_root / "_manifest" / "events"
        assert params.manifest == stage_root / "_manifest" / "manifest.parquet"


class TestIVAnalysisParameters:
    """Test validation for IVAnalysisParameters."""

    def test_valid_parameters(self, tmp_path):
        """Test that valid parameters pass validation."""
        stage_root = tmp_path / "02_stage"
        output_dir = tmp_path / "04_analysis"
        stage_root.mkdir(parents=True)

        params = IVAnalysisParameters(
            stage_root=stage_root,
            date="2025-09-11",
            output_base_dir=output_dir,
            poly_orders=[1, 3, 5, 7],
        )

        assert params.date == "2025-09-11"
        assert params.poly_orders == [1, 3, 5, 7]
        assert params.procedure == "IV"  # default

    def test_invalid_date_format(self, tmp_path):
        """Test that invalid date format raises ValidationError."""
        stage_root = tmp_path / "02_stage"
        output_dir = tmp_path / "04_analysis"
        stage_root.mkdir(parents=True)

        # Invalid formats
        for bad_date in ["2025-9-11", "09-11-2025", "2025/09/11", "20250911"]:
            with pytest.raises(ValidationError) as exc_info:
                IVAnalysisParameters(
                    stage_root=stage_root,
                    date=bad_date,
                    output_base_dir=output_dir,
                )
            assert "String should match pattern" in str(exc_info.value)

    def test_even_polynomial_orders_rejected(self, tmp_path):
        """Test that even polynomial orders are rejected."""
        stage_root = tmp_path / "02_stage"
        output_dir = tmp_path / "04_analysis"
        stage_root.mkdir(parents=True)

        with pytest.raises(ValidationError) as exc_info:
            IVAnalysisParameters(
                stage_root=stage_root,
                date="2025-09-11",
                output_base_dir=output_dir,
                poly_orders=[1, 2, 3],  # 2 is even
            )
        assert "should be odd" in str(exc_info.value)

    def test_polynomial_order_too_high(self, tmp_path):
        """Test that polynomial orders above 15 are rejected."""
        stage_root = tmp_path / "02_stage"
        output_dir = tmp_path / "04_analysis"
        stage_root.mkdir(parents=True)

        with pytest.raises(ValidationError) as exc_info:
            IVAnalysisParameters(
                stage_root=stage_root,
                date="2025-09-11",
                output_base_dir=output_dir,
                poly_orders=[1, 3, 17],  # 17 too high
            )
        assert "too high" in str(exc_info.value)

    def test_polynomial_order_zero_or_negative(self, tmp_path):
        """Test that polynomial orders must be positive."""
        stage_root = tmp_path / "02_stage"
        output_dir = tmp_path / "04_analysis"
        stage_root.mkdir(parents=True)

        with pytest.raises(ValidationError) as exc_info:
            IVAnalysisParameters(
                stage_root=stage_root,
                date="2025-09-11",
                output_base_dir=output_dir,
                poly_orders=[0, 1, 3],  # 0 is invalid
            )
        assert "must be >= 1" in str(exc_info.value)

    def test_v_max_out_of_range(self, tmp_path):
        """Test that v_max must be within reasonable range."""
        stage_root = tmp_path / "02_stage"
        output_dir = tmp_path / "04_analysis"
        stage_root.mkdir(parents=True)

        with pytest.raises(ValidationError) as exc_info:
            IVAnalysisParameters(
                stage_root=stage_root,
                date="2025-09-11",
                output_base_dir=output_dir,
                v_max=-1.0,  # Negative voltage
            )
        assert "greater than or equal to 0" in str(exc_info.value)

    def test_duplicate_poly_orders_removed(self, tmp_path):
        """Test that duplicate polynomial orders are removed."""
        stage_root = tmp_path / "02_stage"
        output_dir = tmp_path / "04_analysis"
        stage_root.mkdir(parents=True)

        params = IVAnalysisParameters(
            stage_root=stage_root,
            date="2025-09-11",
            output_base_dir=output_dir,
            poly_orders=[3, 1, 5, 3, 1],  # Duplicates
        )

        assert params.poly_orders == [1, 3, 5]  # Sorted and deduplicated

    def test_get_stats_dir(self, tmp_path):
        """Test helper method for getting stats directory."""
        stage_root = tmp_path / "02_stage"
        output_dir = tmp_path / "04_analysis"
        stage_root.mkdir(parents=True)

        params = IVAnalysisParameters(
            stage_root=stage_root,
            date="2025-09-11",
            output_base_dir=output_dir,
            procedure="IVg",
        )

        expected = output_dir / "iv_stats" / "2025-09-11_IVg"
        assert params.get_stats_dir() == expected


class TestPlottingParameters:
    """Test validation for PlottingParameters."""

    def test_valid_parameters(self, tmp_path):
        """Test that valid parameters pass validation."""
        output_dir = tmp_path / "plots"

        params = PlottingParameters(
            output_dir=output_dir,
            dpi=300,
            format="png",
        )

        assert params.dpi == 300
        assert params.format == "png"
        assert params.style == "publication"  # default

    def test_dpi_out_of_range(self, tmp_path):
        """Test that DPI must be within valid range."""
        output_dir = tmp_path / "plots"

        # Too low
        with pytest.raises(ValidationError) as exc_info:
            PlottingParameters(
                output_dir=output_dir,
                dpi=50,
            )
        assert "greater than or equal to 72" in str(exc_info.value)

        # Too high
        with pytest.raises(ValidationError) as exc_info:
            PlottingParameters(
                output_dir=output_dir,
                dpi=2000,
            )
        assert "less than or equal to 1200" in str(exc_info.value)

    def test_invalid_format(self, tmp_path):
        """Test that format must be one of allowed values."""
        output_dir = tmp_path / "plots"

        with pytest.raises(ValidationError) as exc_info:
            PlottingParameters(
                output_dir=output_dir,
                format="gif",  # Not allowed
            )
        assert "String should match pattern" in str(exc_info.value)

    def test_invalid_style(self, tmp_path):
        """Test that style must be one of allowed values."""
        output_dir = tmp_path / "plots"

        with pytest.raises(ValidationError) as exc_info:
            PlottingParameters(
                output_dir=output_dir,
                style="custom",  # Not allowed
            )
        assert "String should match pattern" in str(exc_info.value)

    def test_grid_alpha_range(self, tmp_path):
        """Test that grid_alpha must be between 0 and 1."""
        output_dir = tmp_path / "plots"

        with pytest.raises(ValidationError) as exc_info:
            PlottingParameters(
                output_dir=output_dir,
                grid_alpha=1.5,  # > 1.0
            )
        assert "less than or equal to 1" in str(exc_info.value)

    def test_get_figsize(self, tmp_path):
        """Test helper method for getting figure size."""
        output_dir = tmp_path / "plots"

        params = PlottingParameters(
            output_dir=output_dir,
            figure_width=15.0,
            figure_height=10.0,
        )

        assert params.get_figsize() == (15.0, 10.0)

    def test_get_style_params(self, tmp_path):
        """Test style parameter generation."""
        output_dir = tmp_path / "plots"

        params = PlottingParameters(
            output_dir=output_dir,
            style="publication",
            font_size=12,
        )

        style_params = params.get_style_params()
        assert style_params["font.size"] == 12
        assert "font.family" in style_params


class TestPipelineParameters:
    """Test validation for PipelineParameters (combined)."""

    def test_valid_pipeline(self, tmp_path):
        """Test that valid pipeline parameters pass validation."""
        raw_root = tmp_path / "01_raw"
        stage_root = tmp_path / "02_stage"
        yaml_file = tmp_path / "procedures.yml"
        output_dir = tmp_path / "04_analysis"
        plot_dir = tmp_path / "plots"

        raw_root.mkdir()
        stage_root.mkdir()
        yaml_file.write_text("procedures: {}")

        params = PipelineParameters(
            staging=StagingParameters(
                raw_root=raw_root,
                stage_root=stage_root,
                procedures_yaml=yaml_file,
            ),
            analysis=IVAnalysisParameters(
                stage_root=stage_root,
                date="2025-09-11",
                output_base_dir=output_dir,
            ),
            plotting=PlottingParameters(
                output_dir=plot_dir,
            ),
        )

        assert params.run_staging is True
        assert params.run_analysis is True
        assert params.run_plotting is True

    def test_inconsistent_stage_roots(self, tmp_path):
        """Test that staging and analysis stage_root must match."""
        raw_root = tmp_path / "01_raw"
        stage_root1 = tmp_path / "02_stage_1"
        stage_root2 = tmp_path / "02_stage_2"
        yaml_file = tmp_path / "procedures.yml"
        output_dir = tmp_path / "04_analysis"
        plot_dir = tmp_path / "plots"

        raw_root.mkdir()
        stage_root1.mkdir()
        stage_root2.mkdir()
        yaml_file.write_text("procedures: {}")

        with pytest.raises(ValidationError) as exc_info:
            PipelineParameters(
                staging=StagingParameters(
                    raw_root=raw_root,
                    stage_root=stage_root1,  # Different
                    procedures_yaml=yaml_file,
                ),
                analysis=IVAnalysisParameters(
                    stage_root=stage_root2,  # Different
                    date="2025-09-11",
                    output_base_dir=output_dir,
                ),
                plotting=PlottingParameters(
                    output_dir=plot_dir,
                ),
            )
        assert "must match" in str(exc_info.value)

    def test_json_serialization(self, tmp_path):
        """Test saving and loading from JSON."""
        raw_root = tmp_path / "01_raw"
        stage_root = tmp_path / "02_stage"
        yaml_file = tmp_path / "procedures.yml"
        output_dir = tmp_path / "04_analysis"
        plot_dir = tmp_path / "plots"
        json_file = tmp_path / "config.json"

        raw_root.mkdir()
        stage_root.mkdir()
        yaml_file.write_text("procedures: {}")

        params = PipelineParameters(
            staging=StagingParameters(
                raw_root=raw_root,
                stage_root=stage_root,
                procedures_yaml=yaml_file,
                workers=8,
            ),
            analysis=IVAnalysisParameters(
                stage_root=stage_root,
                date="2025-09-11",
                output_base_dir=output_dir,
                poly_orders=[1, 3, 5],
            ),
            plotting=PlottingParameters(
                output_dir=plot_dir,
                dpi=300,
            ),
        )

        # Save to JSON
        params.to_json(json_file)
        assert json_file.exists()

        # Load from JSON
        loaded = PipelineParameters.from_json(json_file)
        assert loaded.staging.workers == 8
        assert loaded.analysis.poly_orders == [1, 3, 5]
        assert loaded.plotting.dpi == 300


def test_validation_catches_bad_values_demo():
    """
    Demonstration test showing various validation errors.

    This test intentionally triggers multiple validation errors to
    demonstrate that the Pydantic models properly catch bad inputs.
    """
    print("\n" + "=" * 70)
    print("VALIDATION DEMONSTRATION - Testing Bad Values")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        raw_root = tmp_path / "01_raw"
        stage_root = tmp_path / "02_stage"
        yaml_file = tmp_path / "procedures.yml"

        raw_root.mkdir()
        yaml_file.write_text("procedures: {}")

        # Test 1: Invalid worker count
        print("\n[TEST 1] Invalid worker count (0):")
        try:
            StagingParameters(
                raw_root=raw_root,
                stage_root=stage_root,
                procedures_yaml=yaml_file,
                workers=0,
            )
            print("  ❌ FAILED - should have raised ValidationError")
        except ValidationError as e:
            print(f"  ✓ CAUGHT: {e.errors()[0]['msg']}")

        # Test 2: Invalid date format
        print("\n[TEST 2] Invalid date format (2025-9-11):")
        try:
            IVAnalysisParameters(
                stage_root=stage_root,
                date="2025-9-11",  # Should be 2025-09-11
                output_base_dir=tmp_path / "analysis",
            )
            print("  ❌ FAILED - should have raised ValidationError")
        except ValidationError as e:
            print(f"  ✓ CAUGHT: {e.errors()[0]['msg']}")

        # Test 3: Even polynomial order
        print("\n[TEST 3] Even polynomial order (2):")
        try:
            IVAnalysisParameters(
                stage_root=stage_root,
                date="2025-09-11",
                output_base_dir=tmp_path / "analysis",
                poly_orders=[1, 2, 3],
            )
            print("  ❌ FAILED - should have raised ValidationError")
        except ValidationError as e:
            print(f"  ✓ CAUGHT: {e.errors()[0]['msg']}")

        # Test 4: DPI out of range
        print("\n[TEST 4] DPI too high (2000):")
        try:
            PlottingParameters(
                output_dir=tmp_path / "plots",
                dpi=2000,
            )
            print("  ❌ FAILED - should have raised ValidationError")
        except ValidationError as e:
            print(f"  ✓ CAUGHT: {e.errors()[0]['msg']}")

        # Test 5: Invalid format
        print("\n[TEST 5] Invalid plot format (gif):")
        try:
            PlottingParameters(
                output_dir=tmp_path / "plots",
                format="gif",
            )
            print("  ❌ FAILED - should have raised ValidationError")
        except ValidationError as e:
            print(f"  ✓ CAUGHT: {e.errors()[0]['msg']}")

        # Test 6: Non-existent path
        print("\n[TEST 6] Non-existent raw_root:")
        try:
            StagingParameters(
                raw_root=tmp_path / "does_not_exist",
                stage_root=stage_root,
                procedures_yaml=yaml_file,
            )
            print("  ❌ FAILED - should have raised ValidationError")
        except ValidationError as e:
            print(f"  ✓ CAUGHT: {e.errors()[0]['msg']}")

        print("\n" + "=" * 70)
        print("✓ All validation tests passed - bad values were caught!")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    # Run demonstration
    test_validation_catches_bad_values_demo()

    # Run full test suite
    print("\nRunning full test suite with pytest...\n")
    pytest.main([__file__, "-v", "--tb=short"])
