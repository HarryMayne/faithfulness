"""
Run scaling laws experiment across multiple Qwen3 model sizes.

This script processes a parquet file with counterfactuals and generates
reference answers using multiple Qwen3 models of different sizes.

Updates:
    [19/11/2025] Changed to use real

Usage:
    CUDA_VISIBLE_DEVICES=2,3 python -m experiment_scripts.run_scaling_laws data/natural_counterfactuals/breast_cancer_recurrence_counterfactual_dataset_balanced.parquet

"""

import argparse
import asyncio
import os
from dataclasses import replace
from pathlib import Path

# Set environment variables for stability (same as run_experiment.py)
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_LOGGING_LEVEL'] = 'INFO'
os.environ['RAY_DEDUP_LOGS'] = '0'

from src.reference_answer_generation.multi_llm_runner import MultiLLMExperimentRunner, ExperimentConfig
from src.utils import LLMConfig
from src.vllm_configs import VLLM_CONFIGS

QWEN3_MODELS = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-32B",
]


# def get_qwen3_configs(enable_reasoning: bool = True):
#     """
#     Create LLMConfig for each Qwen3 model size.
#     Using consistent settings optimized for 2x H200 GPUs.
    
#     Args:
#         enable_reasoning: Whether to enable reasoning/thinking mode for models
#     """
#     # Qwen3 model sizes to test
#     models = [
#         "Qwen/Qwen3-0.6B",
#         "Qwen/Qwen3-1.7B",
#         "Qwen/Qwen3-4B",
#         "Qwen/Qwen3-8B",
#         "Qwen/Qwen3-14B",
#         "Qwen/Qwen3-32B",
#     ]
    
#     configs = []
#     for model_name in models:
#         # Adaptive settings based on model size
#         model_size = model_name.split('-')[-1]  # e.g., "0.6B", "32B"
        
#         # Larger models need more conservative settings
#         if model_size in ["14B", "32B"]:
#             gpu_mem = 0.75
#             max_len = 8192
#         else:
#             gpu_mem = 0.80
#             max_len = 8192
        
#         configs.append(LLMConfig(
#             model_name=model_name,
#             max_tokens=8000,
#             tensor_parallel_size=2,
#             gpu_memory_utilization=gpu_mem,
#             max_model_len=max_len,
#             trust_remote_code=True,
#             enable_reasoning=enable_reasoning,
#         ))
    
#     return configs


def run_scaling_laws(
    input_parquet_path: str,
    output_folder: str = "experiments/scaling_laws/qwen3",
    enable_reasoning: bool = True,
    seed: int = None
) -> str:
    """
    Run scaling laws experiment on a parquet file.
    
    Args:
        input_parquet_path: Path to input parquet file with counterfactuals
        output_folder: Folder where timestamped run folder will be created
        enable_reasoning: Whether to enable reasoning/thinking mode for models
        seed: Random seed for reproducibility
        
    Returns:
        Path to the output run folder
    """
    print("="*80)
    print("QWEN3 SCALING LAWS EXPERIMENT")
    print("="*80)
    print(f"Input: {input_parquet_path}")
    print(f"Output folder: {output_folder}")
    print(f"Enable reasoning: {enable_reasoning}")
    print(f"Seed: {seed}")
    print("="*80)
    
    # Pull configs for all desired Qwen models from the shared registry
    llm_configs: list[LLMConfig] = []
    for model_name in QWEN3_MODELS:
        base_config = VLLM_CONFIGS.get(model_name)
        if base_config is None:
            raise ValueError(f"Model config for {model_name} not found in VLLM_CONFIGS")
        llm_configs.append(replace(base_config, enable_reasoning=enable_reasoning))
    
    print(f"\nModels to test: {len(llm_configs)}")
    for i, config in enumerate(llm_configs, 1):
        print(f"  {i}. {config.model_name}")
    
    print("="*80)
    print()
    
    # Create experiment config
    experiment_config = ExperimentConfig(
        llm_configs=llm_configs,
        input_parquet=input_parquet_path,
        output_folder=output_folder
    )
    
    # Run experiment using the working MultiLLMExperimentRunner. Added optional max_batch_size parameter
    runner = MultiLLMExperimentRunner(experiment_config, max_batch_size=10000)
    asyncio.run(runner.run())
    
    print("\n" + "="*80)
    print("✓ SCALING LAWS EXPERIMENT COMPLETE")
    print("="*80)
    print(f"Results saved to: {runner.run_folder}")
    
    return str(runner.run_folder)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run Qwen3 scaling laws experiment across multiple model sizes"
    )
    parser.add_argument(
        "input_parquet",
        type=str,
        help="Path to input parquet file with counterfactuals"
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default="experiments/scaling_laws/qwen3",
        help="Folder where timestamped run folder will be created (default: experiments/scaling_laws/qwen3)"
    )
    parser.add_argument(
        "--no-reasoning",
        action="store_true",
        help="Disable reasoning/thinking mode for models (default: reasoning enabled)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Run the scaling laws experiment
    run_scaling_laws(
        input_parquet_path=args.input_parquet,
        output_folder=args.output_folder,
        enable_reasoning=not args.no_reasoning,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
