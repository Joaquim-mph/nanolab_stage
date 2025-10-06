#!/bin/bash
# Convenience script to run backward-subtracted analysis for Sept 11, 29, 30

python src/analysis/IV/compute_backward_substracted.py \
  --hysteresis-dirs \
    data/04_analysis/hysteresis/2025-09-11 \
    data/04_analysis/hysteresis/2025-09-29 \
    data/04_analysis/hysteresis/2025-09-30 \
  --dates \
    2025-09-11 \
    2025-09-29 \
    2025-09-30 \
  --output-root data/04_analysis/backward_substracted \
  --plots-dir plots/backward_substracted \
  --voltage-ranges 3.0 4.0 5.0 6.0 7.0 8.0 \
  --poly-orders 1 3 5 7 \
  --exclude-ranges "2025-09-11:2.0"
