# Data Layout

- `raw/`: original downloads and untouched source dumps.
- `natural_counterfactuals/`: natural counterfactual datasets stored as parquet following the `CounterfactualDatabase` schema (question + counterfactual pairs). e.g. the tabular data counterfactuals
- `generated_counterfactuals/`: model-generated counterfactual datasets using the same parquet schema.

All curated datasets in `natural_counterfactuals/` and `generated_counterfactuals/` should be parquet files compatible with the shared `CounterfactualDatabase` format.
