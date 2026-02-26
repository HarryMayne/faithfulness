
"""
Generate predictor answers for counterfactual prompts in a parquet file.

This script processes a parquet file containing counterfactuals with prompts
and generates predictor model responses for both with-explanation and
without-explanation versions.

Usage:
    # Basic usage
    CUDA_VISIBLE_DEVICES=3 python -m src.prediction_generation.generate_predictor_answers \
        tabular_results/breast_cancer_recurrence_reference.parquet \
        --output-parquet tabular_results/breast_cancer_recurrence_predictions.parquet \
        --model Qwen/Qwen3-8B \
        --batch-size 10000

    # Note: this only adds scores, it doesn't do any filtering (not a feature)
    CUDA_VISIBLE_DEVICES=0 python -m src.prediction_generation.generate_predictor_answers \
        tabular_results/sample_reference.parquet \
        --output-parquet tabular_results/sample_predictions.parquet \
        --model openai/gpt-oss-20b \
        --assess-testability
    
    # Do the scaling laws processing
    CUDA_VISIBLE_DEVICES=0 python -m src.prediction_generation.generate_predictor_answers \
        experiments/scaling_laws/qwen3/gpt_oss_results/breast_cancer_recurrence_multi_model_responses.parquet \
        --output-parquet experiments/scaling_laws/qwen3/gpt_oss_results/breast_cancer_recurrence_predictions_testabillity.parquet \
        --model openai/gpt-oss-20b \
        --batch-size 100000 \

    # Ensemble support. You have two levers here: 
    (i)     multiple different predictor models (list multiple models)
    (ii)    repeats of each predictor model (set repeats). This only acts on the WITH explanation predictions
    These can be combined to have multiple predictor models do multiple repeats. 

    python -m src.prediction_generation.generate_predictor_answers \
        tabular_results/sample_reference.parquet \
        --output-parquet tabular_results/sample_predictions.parquet \
        --model Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B \
        --predictor-repeats 2

"""
import asyncio
import argparse
from pathlib import Path
from typing import List
from vllm import LLM

from src.prediction_generation.counterfactual_prediction import PredictorAnswerGenerator
from src.utils import LLMConfig, filter_records_by_reference_answer, cleanup_after_model, get_model_name
from src.schema import CounterfactualDatabase
from src.vllm_configs import VLLM_CONFIGS


async def generate_predictor_answers(
    input_parquet_path: str,
    output_parquet_path: str,
    configs: List[LLMConfig],
    batch_size: int = 50,
    answer_first_only: bool = False,
    assess_testability: bool = False,
    predictor_repeats: int = 1
) -> None:
    """
    Generate predictor answers for counterfactuals in a parquet file.

    Args:
        input_parquet_path: Path to input parquet file with reference answers
        output_parquet_path: Path to save output parquet file with predictor answers
        configs: List of LLM configurations for the predictor models (potentially multiple)
        batch_size: Batch size for processing prompts
        answer_first_only: If True, only process records where answer_first=True
        assess_testability: If True, assess counterfactual testability before generating predictions (only for first model).
        predictor_repeats: Number of times to run each predictor over each WITH-explanation prompt
    """
    print("="*80)
    print("PREDICTOR ANSWER GENERATION")
    print("="*80)
    print(f"Input: {input_parquet_path}")
    print(f"Output: {output_parquet_path}")
    print(f"Predictor models ({len(configs)}):")
    for i, config in enumerate(configs, 1):
        print(f"  {i}. {config.model_name}")
    print(f"Batch size: {batch_size}")
    print(f"Repeats per predictor model: {predictor_repeats}")
    if answer_first_only:
        print(f"Filtering: answer_first=True only")
    if assess_testability:
        print(f"Testability assessment: ENABLED")

    # Per-load the database
    db = CounterfactualDatabase.load_parquet(input_parquet_path)
    if not db.records:
        raise ValueError(f"No records found in {input_parquet_path}")

    print(f"\nLoaded {len(db.records)} total records")

    # Detect unique datasets and pre-load their classes
    unique_datasets = {r.original_question.dataset for r in db.records}
    dataset_classes = {name: db.dataset_class_map[name] for name in unique_datasets}

    if len(unique_datasets) == 1:
        dataset_name = next(iter(unique_datasets))
        print(f"Detected dataset: {dataset_name}")
        print(f"Using dataset class: {dataset_classes[dataset_name].__name__}")
        print(f"Valid answers: {dataset_classes[dataset_name].VALID_ANSWERS}")
    else:
        dataset_name = "combined"
        print(f"Detected dataset: {dataset_name} ({len(unique_datasets)} dataset types)")
        for ds in sorted(unique_datasets):
            print(f"  - {ds}: {dataset_classes[ds].VALID_ANSWERS}")

    # Filter records using utility function with per-record VALID_ANSWERS
    filtered_records, stats = filter_records_by_reference_answer(
        records=db.records,
        answer_first_only=answer_first_only,
        dataset_classes=dataset_classes
    )

    # Update database with filtered records
    db.records = filtered_records

    print(f"\nFiltered to {stats['filtered_count']} records")
    print(f"  Dropped (invalid/missing reference answer): {stats['dropped_invalid_answer']}")
    if answer_first_only:
        print(f"  Dropped (answer_first=False): {stats['dropped_answer_last']}")
    print()

    # Detect if predictions already exist -- used for incremental predictor addition
    has_existing_predictions = any(
        r.counterfactual.predictor_response_with_explanation is not None
        for r in db.records
    )
    if has_existing_predictions:
        print("Detected existing predictions - will append new predictor results")

    # Detect if testability already assessed
    has_existing_testability = any(
        r.counterfactual.predictor_counterfactual_testability_score is not None
        for r in db.records
    )
    if has_existing_testability:
        print("Detected existing testability scores")

    # Main loop for the multi-model setup (default to standard processing if 1 model)
    for model_idx, config in enumerate(configs):
        # Only treat as first model if no existing predictions AND first in this run. Allows predictior addition stuff.
        is_first_model = (model_idx == 0) and not has_existing_predictions

        # only assess testability if it doesn't already exist and are calling it
        should_assess_testability = (
            assess_testability and
            model_idx == 0 and
            not has_existing_testability
        )

        print("\n" + "="*80)
        print(f"PREDICTOR MODEL {model_idx + 1}/{len(configs)}: {config.model_name}")
        print("="*80)

        # Create generator for this model (it will initialize the LLM automatically)
        generator = PredictorAnswerGenerator(
            config=config,
            assess_testability=should_assess_testability # add.
        )

        # Run the generation. Note that it now uses the preloaded db where possible
        await generator.process_parquet(
            input_path=input_parquet_path,
            output_path=output_parquet_path,
            max_batch_size=batch_size,
            predictor_repeats=predictor_repeats,
            db=db,
            is_first_model=is_first_model
        )

        # Explicitly del LLM to free GPU memory.
        if hasattr(generator, 'llm_client') and generator.llm_client is not None:
            del generator.llm_client
            import gc
            gc.collect()
            try:
                cleanup_after_model(generator)
            except ImportError:
                pass

    print("\n" + "="*80)
    print("="*80)
    print(f"Results saved to: {output_parquet_path}\n")


async def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate predictor answers for counterfactual prompts"
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
        nargs='+',
        default=["google/gemma-3-27b-it"],
        help=f"Model name(s) to use (must be in VLLM_CONFIGS). Can specify multiple models. Available: {list(VLLM_CONFIGS.keys())}. New addition to the pipeline."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100000,
        help="Batch size for processing prompts (default: 100000, vLLM handles batching automatically)"
    )
    parser.add_argument(
        "--answer-first-only",
        action="store_true",
        help="Only process records where answer_first=True (better parsing success)"
    )
    parser.add_argument(
        "--assess-testability",
        action="store_true",
        help="Assess counterfactual testability (0-10 score) before generating predictions"
    )
    parser.add_argument(
        "--predictor-repeats",
        type=int,
        default=1,
        help="Number of times to run the predictor over each prompt (default: 1)"
    )

    args = parser.parse_args()

    # Validate predictor-repeats. Dont' really need this.
    if args.predictor_repeats < 1:
        print("Error: --predictor-repeats must be >= 1")
        return
    
    # Generate output path if not provided
    if args.output_parquet is None:
        input_path = Path(args.input_parquet)
        output_path = input_path.parent / f"{input_path.stem}_with_predictor_answers.parquet"
        args.output_parquet = str(output_path)
    
    # Get LLM configurations from VLLM_CONFIGS. Break if incorrect model name
    configs = []
    for model_name in args.model:
        if model_name not in VLLM_CONFIGS:
            print(f"Error: Model '{model_name}' not found in VLLM_CONFIGS")
            print(f"Available models: {list(VLLM_CONFIGS.keys())}")
            return
        configs.append(VLLM_CONFIGS[model_name])

    print(f"\nUsing {len(configs)} predictor model(s):")
    for i, config in enumerate(configs, 1):
        print(f"  {i}. {config.model_name}")
    print()

    # Generate predictor answers (generator will initialize LLM automatically)
    await generate_predictor_answers(
        input_parquet_path=args.input_parquet,
        output_parquet_path=args.output_parquet,
        configs=configs,
        batch_size=args.batch_size,
        answer_first_only=args.answer_first_only,
        assess_testability=args.assess_testability,
        predictor_repeats=args.predictor_repeats
    )


if __name__ == "__main__":
    asyncio.run(main())