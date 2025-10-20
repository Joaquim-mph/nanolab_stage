#!/usr/bin/bash
#
# Complete IV Analysis Pipeline for September 2025 measurements
# with E-field and Current Density units
#
# Device dimensions: 100 μm × 50 μm
# E-field: E = V / (100 μm) = V × 100 V/cm
# Current density: J = I / (5×10⁻⁵ cm²)
#
# Measurement dates: Sept 11, 29, 30, 2025
#

set -e  # Exit on error

echo "======================================================================"
echo "IV ANALYSIS PIPELINE - SEPTEMBER 2025 (E-FIELD & CURRENT DENSITY)"
echo "======================================================================"
echo ""
echo "Device dimensions: 100 μm × 50 μm"
echo "E-field conversion: E = V × 100 V/cm"
echo "Current density conversion: J = I × 2×10⁴ A/cm²"
echo ""
echo "Processing dates: 2025-09-11, 2025-09-29, 2025-09-30"
echo "======================================================================"
echo ""

# Dates to process
DATES=("2025-09-11" "2025-09-29" "2025-09-30")
POLY_ORDERS="1 3 5 7"

# Step 1: Preprocessing (segment IV sweeps) - processes all dates
echo "STEP 1: IV Preprocessing (Segment Detection)"
echo "----------------------------------------------------------------------"
echo "Processing all September dates..."

python src/intermediate/IV/iv_preprocessing_script.py \
    --stage-root data/02_stage/raw_measurements \
    --output-root data/03_intermediate/iv_segments \
    --procedure IV \
    --workers 4

if [ $? -eq 0 ]; then
    echo "✓ Preprocessing complete"
else
    echo "✗ Preprocessing failed"
    exit 1
fi

echo ""
echo "======================================================================"
echo "STEP 2: IV Analysis (Statistics & Polynomial Fits)"
echo "----------------------------------------------------------------------"

for DATE in "${DATES[@]}"; do
    echo ""
    echo "Analyzing $DATE..."

    python src/analysis/IV/aggregate_iv_stats.py \
        --intermediate-root data/03_intermediate/iv_segments \
        --date "$DATE" \
        --procedure IV \
        --output-base-dir data/04_analysis \
        --poly-orders $POLY_ORDERS

    if [ $? -eq 0 ]; then
        echo "✓ Analysis complete for $DATE"
    else
        echo "✗ Analysis failed for $DATE"
        exit 1
    fi
done

echo ""
echo "======================================================================"
echo "STEP 3: Hysteresis Calculation"
echo "----------------------------------------------------------------------"

for DATE in "${DATES[@]}"; do
    echo ""
    echo "Computing hysteresis for $DATE..."

    python src/analysis/IV/compute_hysteresis.py \
        --stats-dir "data/04_analysis/iv_stats/${DATE}_IV" \
        --output-dir "data/04_analysis/hysteresis/${DATE}"

    if [ $? -eq 0 ]; then
        echo "✓ Hysteresis complete for $DATE"
    else
        echo "✗ Hysteresis failed for $DATE"
        exit 1
    fi
done

echo ""
echo "======================================================================"
echo "STEP 4: Hysteresis Peak Analysis"
echo "----------------------------------------------------------------------"

for DATE in "${DATES[@]}"; do
    echo ""
    echo "Analyzing hysteresis peaks for $DATE..."

    python src/analysis/IV/analyze_hysteresis_peaks.py \
        --hysteresis-dir "data/04_analysis/hysteresis/${DATE}" \
        --output-dir "data/04_analysis/hysteresis_peaks/${DATE}"

    if [ $? -eq 0 ]; then
        echo "✓ Peak analysis complete for $DATE"
    else
        echo "✗ Peak analysis failed for $DATE"
        exit 1
    fi
done

echo ""
echo "======================================================================"
echo "STEP 5: Comprehensive Comparison Plots (E-FIELD & CURRENT DENSITY)"
echo "----------------------------------------------------------------------"

echo ""
echo "Creating comprehensive comparison plots..."

python src/analysis/IV/comprehensive_comparison_efield_density.py \
    --dates 2025-09-11 2025-09-29 2025-09-30 \
    --output-dir plots/comprehensive_comparison_efield_density_sept \
    --base-dir data/04_analysis

if [ $? -eq 0 ]; then
    echo "✓ Comprehensive comparison plots created"
else
    echo "✗ Comprehensive comparison plots failed"
    exit 1
fi

echo ""
echo "======================================================================"
echo "STEP 6: Backward-Subtracted Hysteresis (E-FIELD & CURRENT DENSITY)"
echo "----------------------------------------------------------------------"

echo ""
echo "Computing backward-substracted hysteresis..."

python src/analysis/IV/compute_backward_substracted_efield.py \
    --hysteresis-dirs \
        data/04_analysis/hysteresis/2025-09-11 \
        data/04_analysis/hysteresis/2025-09-29 \
        data/04_analysis/hysteresis/2025-09-30 \
    --dates 2025-09-11 2025-09-29 2025-09-30 \
    --output-root data/04_analysis/backward_substracted_efield_sept \
    --plots-dir plots/backward_substracted_efield_sept \
    --voltage-ranges 3.0 4.0 5.0 6.0 7.0 8.0 \
    --poly-orders $POLY_ORDERS

if [ $? -eq 0 ]; then
    echo "✓ Backward-subtracted plots created"
else
    echo "✗ Backward-substracted plots failed"
    exit 1
fi

echo ""
echo "======================================================================"
echo "✓ COMPLETE PIPELINE FINISHED SUCCESSFULLY!"
echo "======================================================================"
echo ""
echo "Output locations:"
echo "  Intermediate data: data/03_intermediate/iv_segments/"
echo "  Analysis results:  data/04_analysis/"
echo "  Hysteresis data:   data/04_analysis/hysteresis/"
echo ""
echo "Plots:"
echo "  Comprehensive:     plots/comprehensive_comparison_efield_density_sept/"
echo "  Backward-sub:      plots/backward_substracted_efield_sept/"
echo ""
echo "All plots use:"
echo "  - Electric field (V/cm) on X-axis"
echo "  - Current density (A/cm²) on Y-axis"
echo "  - Device: 100 μm × 50 μm"
echo "======================================================================"
