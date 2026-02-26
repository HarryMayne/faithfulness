# Pytest Guide for LLM Self-Explanations and Faithfulness

A comprehensive guide to running pytest tests on your sample_data.parquet dataset.

## Quick Start

### 1. Activate Your Conda Environment

```bash
conda activate faithfulness-env
```

### 2. Run All Tests

```bash
# Run all tests in the tests directory
pytest

# Run with verbose output
pytest -v

# Run with detailed output showing print statements
pytest -v -s
```
## Project-Specific Examples

### Test the Sample Data

```bash
# Run all sample data tests
pytest tests/test_sample_data.py -v

# Run only data loading tests
pytest tests/test_sample_data.py::TestDataLoading -v

# Run only data quality tests
pytest tests/test_sample_data.py::TestDataQuality -v

# Run with output showing what passed
pytest tests/test_sample_data.py -v -ra
```