# LLM Self-Explanations and Faithfulness

This repository contains the experimental code and analysis for the paper:

**[A Positive Case for Faithfulness: LLM Self-Explanations Help Predict Model Behavior](https://arxiv.org/abs/2602.02639)**

## Overview

This codebase implements experiments for analyzing the faithfulness and simulatability of LLM self-explanations. The repository provides tools for:

- Generating counterfactual questions from datasets
- Collecting reference answers from language models
- Evaluating predictor models on their ability to simulate model behavior
- Analyzing the utility of self-explanations for predicting model outputs

## Setup

### Environment Installation

1. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate faithfulness-env
```

2. Verify installation:
```bash
python -c "import vllm; print('vLLM installed successfully')"
```

### Data Preparation

Generate natural counterfactual datasets:

```bash
# Generate Hamming-ball style counterfactual datasets
python -m src.counterfactual_generation.tabular_counterfactual_generation.tabular_to_text \
    --output_dir data/natural_counterfactuals

# Generate moral machines counterfactual dataset
PYTHONHASHSEED=0 python -m src.counterfactual_generation.tabular_counterfactual_generation.moral_machines_generator

# Build combined dataset
python -m data.natural_counterfactuals.generate_combined
```

## Repository Structure

- **src/** - Core library code
  - `schema.py` - Data structures for experimental results
  - `utils.py` - Shared utilities (parsing, normalization, LLM configuration)
  - `templates/` - Dataset-specific prompt templates
  - `counterfactual_generation/` - Counterfactual generation logic
  - `prediction_generation/` - Predictor model answer generation
  - `reference_answer_generation/` - Reference model answer generation

- **analysis_scripts/** - Analysis scripts for processing experimental results
- **experiment_scripts/** - Scripts for running experiments
- **notebooks/** - Jupyter notebooks for exploratory analysis and visualization
- **tests/** - Unit tests (run with `pytest`)

## Usage

### Running Experiments

1. **Generate reference answers:**
```bash
CUDA_VISIBLE_DEVICES=0 python -m src.reference_answer_generation.generate_reference_answers \
    data/natural_counterfactuals/combined_dataset.parquet \
    --output-parquet experiments/reference_answers.parquet \
    --model Qwen/Qwen3-8B
```

2. **Generate predictor answers:**
```bash
CUDA_VISIBLE_DEVICES=0 python -m src.prediction_generation.generate_predictor_answers \
    experiments/reference_answers.parquet \
    --output-parquet experiments/predictor_answers.parquet \
    --predictor-model google/gemma-2-27b-it
```

3. **Analyze results:**
```bash
python -m analysis_scripts.analyze_simulatability \
    experiments/predictor_answers.parquet \
    --output results/simulatability_analysis.csv
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_sample_data.py -v

# See PYTEST_GUIDE.md for more testing options
```

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{mayne2026positivecasefaithfulnessllm,
      title={A Positive Case for Faithfulness: LLM Self-Explanations Help Predict Model Behavior}, 
      author={Harry Mayne and Justin Singh Kang and Dewi Gould and Kannan Ramchandran and Adam Mahdi and Noah Y. Siegel},
      year={2026},
      eprint={2602.02639},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2602.02639}, 
}
```

## License

See the LICENSE file for details.

## Contact

For questions or issues, please open a GitHub issue or contact the authors.
