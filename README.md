# nanolab_stage
storage-processing-reduction of data


python src/warehouse/build_curated_from_stage_parallel.py \                
  --raw-root data/01_raw \
  --stage-root data/02_stage/raw_measurements \
  --procedures-yaml config/procedures.yml \
  --workers 4 \
  --polars-threads 2

python build_curated_from_stage_parallel.py \
  --stage-root 02_stage/raw_measurements \
  --warehouse-root 03_curated/warehouse \
  --workers 4 \
  --polars-threads 2 \
