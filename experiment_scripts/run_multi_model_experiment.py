"""
Run multi-model experiment on a parquet file.

This script runs multiple models on a single parquet file to generate
reference answers for comparison and analysis.

Usage:
    python experiment_scripts/run_multi_model_experiment.py input.parquet
    python experiment_scripts/run_multi_model_experiment.py input.parquet --models "google/gemma-3-27b-it" "Qwen/Qwen3-32B-reasoning"
    python experiment_scripts/run_multi_model_experiment.py input.parquet --output-folder experiments/my_experiment
"""

import argparse
import os
import asyncio
from pathlib import Path

# Set environment variables for stability
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_LOGGING_LEVEL'] = 'INFO'
os.environ['RAY_DEDUP_LOGS'] = '0'

from src.reference_answer_generation.multi_llm_runner import MultiLLMExperimentRunner, ExperimentConfig
from src.vllm_configs import VLLM_CONFIGS


def main():
    parser = argparse.ArgumentParser(
        description="Run multi-model experiment on a parquet file"
    )
    parser.add_argument(
        "input_parquet",
        type=str,
        help="Path to input parquet file with counterfactuals"
    )
    parser.add_argument(
        "--models",
        nargs='+',
        default=None,
        help=f"Model names to use (must be in VLLM_CONFIGS). If not provided, uses all models. Available: {list(VLLM_CONFIGS.keys())}"
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default="experiments/multi_model_results",
        help="Folder where timestamped run folder will be created (default: experiments/multi_model_results)"
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not Path(args.input_parquet).exists():
        print(f"Error: Input file not found: {args.input_parquet}")
        return
    
    # Select model configs
    if args.models:
        # User specified models
        llm_configs = []
        for model_name in args.models:
            if model_name not in VLLM_CONFIGS:
                print(f"Warning: Model '{model_name}' not found in VLLM_CONFIGS. Skipping.")
                print(f"Available models: {list(VLLM_CONFIGS.keys())}")
            else:
                llm_configs.append(VLLM_CONFIGS[model_name])
        
        if not llm_configs:
            print("Error: No valid models selected")
            return
    else:
        # Use all models from VLLM_CONFIGS
        llm_configs = list(VLLM_CONFIGS.values())
    
    # Print configuration
    print("="*80)
    print("MULTI-MODEL EXPERIMENT")
    print("="*80)
    print(f"Input: {args.input_parquet}")
    print(f"Output folder: {args.output_folder}")
    print(f"\nModels to test ({len(llm_configs)}):")
    for i, config in enumerate(llm_configs, 1):
        reasoning_tag = ""
        if hasattr(config, 'enable_reasoning') and config.enable_reasoning is not None:
            reasoning_tag = f" [reasoning={config.enable_reasoning}]"
        print(f"  {i}. {config.model_name}{reasoning_tag}")
    print("="*80)
    print()
    
    # Create experiment config
    experiment_config = ExperimentConfig(
        llm_configs=llm_configs,
        input_parquet=args.input_parquet,
        output_folder=args.output_folder
    )
    
    # Run experiment
    runner = MultiLLMExperimentRunner(experiment_config, max_batch_size=10000)
    asyncio.run(runner.run())
    
    print("\n" + "="*80)
    print("✓ MULTI-MODEL EXPERIMENT COMPLETE")
    print("="*80)
    print(f"Results saved to: {runner.run_folder}")


if __name__ == "__main__":
    main()
