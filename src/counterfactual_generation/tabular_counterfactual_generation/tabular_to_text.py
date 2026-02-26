"""
Main pipeline for tabular data counterfactuals generation. All functions in TabularToTextConverter class (tabular_to_text_converter.py)
Tabular datasets each have templates to convert tabular data -> text
Run as a module from the repo root

Usage:
    python -m src.counterfactual_generation.tabular_counterfactual_generation.tabular_to_text --output_dir data/natural_counterfactuals
    
"""

from src.counterfactual_generation.tabular_counterfactual_generation.tabular_to_text_converter import TabularToTextConverter
from src.templates.heart_disease import HeartDisease
from src.templates.pima_diabetes import PimaDiabetes
from src.templates.breast_cancer_recurrence import BreastCancerRecurrence
from src.templates.income import IncomeDataset
from src.templates.attrition import AttritionDataset
from src.templates.bank_marketing import BankMarketing
import os
import argparse

def convert_to_text(dataset, answer_first_only: bool = False, output_dir: str = "tabular_results"):
    """
    main function

    Args:
        - answer_first_only: Whether to only include examples of answering first
        - output_dir: Output directory
    """
    # load using custom dataset class.
    df = dataset.load_dataset()
    print(f"\nDataset shape: {df.shape}")
    converter = TabularToTextConverter(
        df, 
        target_col='target',
        description_generator=dataset.description_generator,
        prompt_generator=dataset.create_reference_prompt,
        dataset_name=dataset.to_string(),
        target_formatter=dataset.format_target  # Pass the format_target method
    )
    print("\n" + "="*60)
    print("METHOD: REPEATED HAMMING BALLS (ONE PER DATA POINT)")
    print("="*60)
    groups = converter.find_hamming_balls_repeated(
        max_distance=2,
        min_group_size=5,
        max_group_size=10,
        epsilon=0.3
    )
    if groups:
        # Special handling for Income and Bank Marketing datasets: Sub-sample to 500 groups (center points)
        if dataset.to_string() in ["income", "bank_marketing"] and len(groups) > 500:
            print(f"\n{dataset.to_string()} Dataset: Randomly sub-sampling 500 Hamming groups from {len(groups)} total groups...")
            import random # Local import or move to top
            random.seed(42)
            groups = random.sample(groups, 500)
            # CRITICAL: Update the converter's internal state so export uses the sampled groups
            converter.hamming_groups = groups
            
        print(f"Total hamming groups: {len(groups)}")
        total_entries = sum(len(g.counterfactual_indices) for g in groups)
        print(f"Total entries (with repetition): {total_entries}")
        print(f"Average entries per point: {total_entries/len(df):.1f}")
        
        # Calculate number of unique rows used
        unique_rows_used = set()
        for g in groups:
            unique_rows_used.update(g.counterfactual_indices)
        unused_rows = len(df) - len(unique_rows_used)
        print(f"Unique rows used: {len(unique_rows_used)}/{len(df)}")
        print(f"Unused rows: {unused_rows}")
        
        output_file = os.path.join(output_dir, f'{dataset.to_string()}_counterfactual_dataset_balanced.parquet')
        converter.export_to_parquet(output_file, answer_first_only=answer_first_only)
    else:
        print("\nNo hamming groups found with the specified parameters.")
        print("Try adjusting max_distance, min_group_size, or max_group_size and epsilon.")

def main():
    """
    Main execution function demonstrating the workflow
    """
    
    parser = argparse.ArgumentParser(
        description="Convert tabular datasets to text-based counterfactual datasets"
    )
    parser.add_argument(
        "--answer-first-only",
        action="store_true",
        help="Only generate answer_first=True versions (better parsing success)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tabular_results",
        help="Directory to store results"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("Tabular to Text Dataset Converter")
    print("="*60)
    
    # Create output folder if it doesn't exist. Use arg.
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output folder: {args.output_dir}")
    
    # Optional param to only consider answer first regime.
    if args.answer_first_only:
        print("Filtering: answer_first=True only")

    datasets = [
                HeartDisease,
                PimaDiabetes,
                BreastCancerRecurrence,
                IncomeDataset,
                AttritionDataset,
                BankMarketing
    ]

    # Convert dataset to text format
    for dataset in datasets:
        convert_to_text(dataset, answer_first_only=args.answer_first_only, output_dir=args.output_dir)

if __name__ == "__main__":
    main()
