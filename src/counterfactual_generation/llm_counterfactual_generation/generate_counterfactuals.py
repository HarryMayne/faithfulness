"""
The main script for generating counterfactuals using LLMs. 
Run this from the repo root as a module
It accepts parquet datasets which are a CounterfactualsDatabases. It then cleans it down to only the original questions, then generates counterfactuals using the desired method. This means that you can input:
1. A new CounterfactualsDatabase parquet with fresh questions. e.g. We're making a new dataset.
2. One of the tabular dataset parquets, e.g. Breat Cancer. It will first get rid of the hamming ball counterfactuals.

Select the type of generator via the --generator flag. At the moment it supports the following:
- BasicGenerator: A single LLM call

Usage:
    python -m src.counterfactual_generation.llm_counterfactual_generation.generate_counterfactuals tabular_results/sample.parquet --output-parquet test.parquet --model Qwen/Qwen3-0.6B --generator BasicGenerator --dataset-name breast_cancer_recurrence
"""
from openai import OpenAI
from google import genai
import asyncio
import os
import json
import argparse
import sys
from pathlib import Path
from tqdm import tqdm
from vllm import LLM, SamplingParams
from src.utils import LLMConfig, filter_records_by_reference_answer
from src.vllm_configs import VLLM_CONFIGS
from src.counterfactual_generation.llm_counterfactual_generation.generator_classes import BasicGenerator
from src.schema import CounterfactualDatabase, ModelInfo, Response
from typing import Any, Dict, Optional, Type, List, Tuple
from src.templates.base import TabularDataset



async def generate_counterfactuals(input_parquet_path, output_parquet_path, config, batch_size, generator_name: str,
                                   dataset_class:Type[TabularDataset],
                                   n_counterfactuals: int=8) -> None:
    """
    Generate counterfactuals from an input Parquet already in CounterfactualDatabase format.

    The input file must include the prefixed columns that `CounterfactualDatabase.load_parquet`
    expects (e.g., `original_dataset`, `original_question`, `original_question_idx`,
    `counterfactual_generator_model`, `counterfactual_generator_method`,
    `counterfactual_question`). The loader reconstructs `FaithfulnessRecord` objects and we
    re-flatten to a DataFrame for the generator.
    
    Args:
        input_parquet_path: Path to a Parquet loadable by `CounterfactualDatabase.load_parquet`.
        output_parquet_path: Destination for the generated Parquet.
        config: LLM configuration from `VLLM_CONFIGS`.
        batch_size: Batch size passed through to the generator.
        generator_name: Name of the generator class to instantiate.
        dataset_class: The dataset class used to format prompts. #TODO: should create a default here that creates basic formatted prompts. 
    """

    print("="*80)
    print("COUNTERFACTUAL GENERATION")
    print("="*80)
    print(f"Input: {input_parquet_path}")
    print(f"Output: {output_parquet_path}")
    print(f"Model: {config.model_name}")
    print(f"Batch size: {batch_size}")

    # Load the parquet using the schema-aware loader to enforce required columns. Parse to df.
    try:
        cf_db = CounterfactualDatabase.load_parquet(input_parquet_path)
    except Exception as e:
        print(f"Failed to load Parquet as CounterfactualDatabase: {e}")
        return
    dataset = cf_db.to_dataframe()

    # Zero out all columns except the original question details we care about
    keep_cols = {
        "original_dataset",
        "original_question",
        "original_question_idx",
        "original_ground_truth",
        "original_answer_first",
        "original_description",
        "original_question_options",
    }
    for col in dataset.columns:
        if col not in keep_cols:
            dataset[col] = None

    # Remove duplicate questions.
    original_count = len(dataset)
    dataset.drop_duplicates(subset=["original_question"], inplace=True)
    unique_count = len(dataset)
    reduction_pct = ((original_count - unique_count) / original_count * 100) if original_count else 0
    print(f"Original questions: {original_count}, unique questions: {unique_count}, reduction: {reduction_pct:.2f}%")

    # Create generator (it will initialize the LLM automatically). Parse the clean dataset into the generator
    if generator_name == "BasicGenerator":
        generator = BasicGenerator(
            config=config,
            max_batch_size = 1000,
            n_counterfactuals =n_counterfactuals,
            allowed_method = [1,2,3,4],
            dataset_class=dataset_class,
            )
    else:
        raise ValueError(f"Unsupported generator: {generator_name}")

    # Run the generation over the dataset. All agents need to have this function with the same signature.
    generated_db = await generator.run_generator(dataset)
    if generated_db is None:
        print("No counterfactuals generated; skipping save.")
        return

    # Save results
    print(f"\nSaving generated counterfactuals to: {output_parquet_path}")
    generated_db.save_parquet(output_parquet_path)
    print("Done")

    # Print summary
    total_records = len(generated_db.records)
    total_questions = dataset.shape[0]
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total original questions processed: {total_questions}")
    print(f"Total counterfactual rows stored: {total_records}")
    print(f"Output saved to: {output_parquet_path}")
    
    print("\n" + "="*80)
    print("DONE")
    print("="*80)

    return ""











async def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate counterfactuals using specified agent"
    )
    parser.add_argument(
        "input_parquet",
        type=str,
        help="Path to input parquet file with reference answers"
    )
    parser.add_argument(
        "--output-parquet",
        type=str,
        help="Path to save output parquet file (default: input_with_predictor_answers.parquet)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-3-27b-it",
        help=f"Model name to use (must be in VLLM_CONFIGS). Available: {list(VLLM_CONFIGS.keys())}"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for processing prompts (default: 50)"
    )
    parser.add_argument(
        "--generator",
        type=str,
        default="BasicGenerator",
        help="Name of counterfactual generator to use (default: BasicGenerator)"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="Name of the dataset class to use for prompt formatting (e.g., 'heart_disease','pima_diabetes','breast_cancer_recurrence','trait','multiple_choice_dataset')"
    )

    args = parser.parse_args()
    
    # Generate output path if not provided
    if args.output_parquet is None:
        input_path = Path(args.input_parquet)
        output_path = input_path.parent / f"{input_path.stem}_counterfactuals_generated.parquet"
        args.output_parquet = str(output_path)
    
    # Get LLM configuration from VLLM_CONFIGS
    if args.model not in VLLM_CONFIGS:
        print(f"Error: Model '{args.model}' not found in VLLM_CONFIGS")
        print(f"Available models: {list(VLLM_CONFIGS.keys())}")
        return

    config = VLLM_CONFIGS[args.model]
    print(f"Using model config for: {args.model}\n")

    db = CounterfactualDatabase
    dataset_class = db.dataset_class_map.get(args.dataset_name)

    # logic
    await generate_counterfactuals(
        input_parquet_path=args.input_parquet,
        output_parquet_path=args.output_parquet,
        config=config,
        batch_size=args.batch_size,
        generator_name=args.generator,
        dataset_class=dataset_class,
    )

if __name__ == "__main__":
    asyncio.run(main())
