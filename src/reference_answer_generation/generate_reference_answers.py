"""
Main script for reference model answer generation. Uses the counterfactual parquets.
Run from the repo root as a module
Works for vLLM and closed models.

# Example usage
CUDA_VISIBLE_DEVICES=3 \
  python -m src.reference_answer_generation.generate_reference_answers \
      tabular_results/breast_cancer_recurrence_counterfactual_dataset_balanced.parquet \
      --output-parquet tabular_results/breast_cancer_recurrence_reference.parquet \
      --model Qwen/Qwen3-8B \
      --batch-size 10000

CUDA_VISIBLE_DEVICES=2 \
  python -m src.reference_answer_generation.generate_reference_answers \
      data/natural_counterfactuals/breast_cancer_recurrence_counterfactual_dataset_balanced.parquet \
      --output-parquet experiments/breast_cancer/test/responses.parquet \
      --model Qwen/Qwen3-0.6B \
      --batch-size 10000
"""
import asyncio
from src.reference_answer_generation.reference_answer_generator import ReferenceAnswerGenerator
from src.utils import LLMConfig, filter_records_by_reference_answer
from src.schema import CounterfactualDatabase
from src.vllm_configs import VLLM_CONFIGS
import argparse
from pathlib import Path


async def generate_reference_answers(
    input_parquet_path: str,
    output_parquet_path: str,
    config: LLMConfig,
    batch_size: int = 128,
    answer_first_only: bool = False
) -> None:
    """
    Generate reference answers for a parquet file.
    
    This is a reusable function that can be called programmatically.
    It automatically detects the dataset type from the parquet file.
    
    Args:
        input_parquet_path: Path to input parquet file (with counterfactuals but no reference answers)
        output_parquet_path: Path to save output parquet file (with reference answers added)
        config: LLMConfig with model settings
        batch_size: Maximum batch size for LLM inference (default: 128)
        answer_first_only: If True, only process records where answer_first=True
    """
    print("="*60)
    print("GENERATING REFERENCE ANSWERS")
    print("="*60)
    print(f"Input: {input_parquet_path}")
    print(f"Output: {output_parquet_path}")
    print(f"Model: {config.model_name}\n")
    if answer_first_only:
        print(f"Filtering: answer_first=True only")
    
    # Load the parquet to detect dataset type
    db = CounterfactualDatabase.load_parquet(input_parquet_path)
    
    if len(db.records) == 0:
        raise ValueError(f"No records found in {input_parquet_path}")
    
    print(f"Loaded {len(db.records)} total records")

    # Detect unique datasets in the database
    unique_datasets = {r.original_question.dataset for r in db.records}
    if len(unique_datasets) == 1:
        dataset_name = next(iter(unique_datasets))
        dataset_class = db.dataset_class_map[dataset_name]
        print(f"Detected dataset: {dataset_name}")
        print(f"Using dataset class: {dataset_class.__name__}")
    else:
        dataset_name = "combined"
        print(f"Detected dataset: {dataset_name} ({len(unique_datasets)} dataset types)")
        for ds in sorted(unique_datasets):
            print(f"  - {ds}")
    
    # Filter records if requested
    if answer_first_only:
        original_count = len(db.records)
        db.records = [r for r in db.records if r.original_question.answer_first]
        filtered_count = len(db.records)
        dropped = original_count - filtered_count
        print(f"\nFiltered to {filtered_count} records")
        print(f"  Dropped (answer_first=False): {dropped}")
    
    print()
    
    # Create generator (it will initialize the LLM automatically)
    generator = ReferenceAnswerGenerator(config)
    await generator.process_parquet(input_path=input_parquet_path, output_path=output_parquet_path, max_batch_size=batch_size)
    
    print("\n" + "="*60)
    print("✓ DONE!")
    print("="*60)
    print(f"Enhanced dataset saved to: {output_parquet_path}\n")

async def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser(
        description="Generate reference answers for counterfactual datasets"
    )
    parser.add_argument(
        "input_parquet",
        type=str,
        help="Path to input parquet file (with counterfactuals but no reference answers)"
    )
    parser.add_argument(
        "--output-parquet",
        type=str,
        default=None,
        help="Path to output parquet file (defaults to input_with_reference_answers.parquet)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-30B-A3B-Instruct-2507",
        help=f"Model name to use (must be in VLLM_CONFIGS). Available: {list(VLLM_CONFIGS.keys())}"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for processing prompts (default: 128)"
    )
    parser.add_argument(
        "--answer-first-only",
        action="store_true",
        help="Only process records where answer_first=True (better parsing success)"
    )
    
    args = parser.parse_args()
    
    # Determine output path
    if args.output_parquet is None:
        input_path = Path(args.input_parquet)
        output_path = input_path.parent / f"{input_path.stem}_with_reference_answers.parquet"
    else:
        output_path = Path(args.output_parquet)
    
    print("="*60)
    print("Reference Answer Generator")
    print("="*60)
    print(f"Input:  {args.input_parquet}")
    print(f"Output: {output_path}\n")
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get LLM configuration from VLLM_CONFIGS
    if args.model not in VLLM_CONFIGS:
        print(f"Error: Model '{args.model}' not found in VLLM_CONFIGS")
        print(f"Available models: {list(VLLM_CONFIGS.keys())}")
        return
    
    model_config = VLLM_CONFIGS[args.model]
    print(f"Using model config for: {args.model}\n")
    
    # Generate reference answers (generator will initialize LLM automatically). The processing of the function.
    await generate_reference_answers(
        input_parquet_path=args.input_parquet,
        output_parquet_path=str(output_path),
        config=model_config,
        batch_size=args.batch_size,
        answer_first_only=args.answer_first_only
    )


if __name__ == "__main__":
    asyncio.run(main())