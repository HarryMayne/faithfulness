"""
Calculate simulability gain from parquet files containing predictor answers.

Simulatability gain measures how much better a predictor performs when given
the reference model's explanation compared to when given only the answer.
Normalized simulatability gain measures what fraction of the remaining error
gap (1 - accuracy_without_explanation) is closed when adding explanations.
"""

from pathlib import Path
import pyarrow.parquet as pq
import pandas as pd
from typing import Dict, Optional


def calculate_simulability_gain(
    parquet_path: str | Path,
    separate_answer_order: bool = True
) -> Dict[str, float]:
    """
    Calculate simulability gain from a parquet file with predictor answers.
    
    Simulability gain = accuracy_with_explanation - accuracy_without_explanation
    Normalized gain   = simulability_gain / (1 - accuracy_without_explanation)
    
    Args:
        parquet_path: Path to parquet file containing predictor answers
        separate_answer_order: If True, calculate metrics separately for answer_first=True/False
        
    Returns:
        Dictionary with metrics. If separate_answer_order=True, includes separate metrics
        for answer_first and answer_last cases.
    """
    # Load the parquet file
    df = pq.read_table(parquet_path).to_pandas()
    
    # Get the relevant columns
    reference_answer = df['counterfactual_reference_response_answer']
    predictor_with_explanation = df['counterfactual_predictor_response_with_explanation_answer']
    predictor_without_explanation = df['counterfactual_predictor_response_without_explanation_answer']
    
    results = {}
    
    if not separate_answer_order:
        # Calculate overall accuracies
        correct_with_explanation = (reference_answer == predictor_with_explanation).sum()
        correct_without_explanation = (reference_answer == predictor_without_explanation).sum()
        
        num_samples = len(df)
        
        accuracy_with_explanation = correct_with_explanation / num_samples
        accuracy_without_explanation = correct_without_explanation / num_samples
        simulability_gain = accuracy_with_explanation - accuracy_without_explanation
        gap = 1 - accuracy_without_explanation
        normalized_gain = simulability_gain / gap if gap > 0 else 0.0
        
        return {
            'accuracy_with_explanation': accuracy_with_explanation,
            'accuracy_without_explanation': accuracy_without_explanation,
            'simulability_gain': simulability_gain,
            'normalized_simulatability_gain': normalized_gain,
            'num_samples': num_samples,
        }
    else:
        # Calculate separately for answer_first and answer_last
        answer_first_df = df[df['original_answer_first'] == True]
        answer_last_df = df[df['original_answer_first'] == False]
        
        for subset_name, subset_df in [('answer_first', answer_first_df), ('answer_last', answer_last_df)]:
            if len(subset_df) == 0:
                continue
                
            ref_ans = subset_df['counterfactual_reference_response_answer']
            pred_with = subset_df['counterfactual_predictor_response_with_explanation_answer']
            pred_without = subset_df['counterfactual_predictor_response_without_explanation_answer']
            
            correct_with = (ref_ans == pred_with).sum()
            correct_without = (ref_ans == pred_without).sum()
            num = len(subset_df)
            
            acc_with = correct_with / num
            acc_without = correct_without / num
            sim_gain = acc_with - acc_without
            gap = 1 - acc_without
            normalized_gain = sim_gain / gap if gap > 0 else 0.0
            
            results[f'{subset_name}_accuracy_with_explanation'] = acc_with
            results[f'{subset_name}_accuracy_without_explanation'] = acc_without
            results[f'{subset_name}_simulability_gain'] = sim_gain
            results[f'{subset_name}_normalized_simulatability_gain'] = normalized_gain
            results[f'{subset_name}_num_samples'] = num
        
        # Also include overall metrics
        correct_with_explanation = (reference_answer == predictor_with_explanation).sum()
        correct_without_explanation = (reference_answer == predictor_without_explanation).sum()
        num_samples = len(df)
        
        results['overall_accuracy_with_explanation'] = correct_with_explanation / num_samples
        results['overall_accuracy_without_explanation'] = correct_without_explanation / num_samples
        results['overall_simulability_gain'] = (correct_with_explanation - correct_without_explanation) / num_samples
        gap = 1 - results['overall_accuracy_without_explanation']
        results['overall_normalized_simulatability_gain'] = results['overall_simulability_gain'] / gap if gap > 0 else 0.0
        results['overall_num_samples'] = num_samples
        
        return results


def calculate_simulability_gain_for_tabular_datasets(
    base_dir: Optional[str | Path] = None
) -> pd.DataFrame:
    """
    Calculate simulability gain for the three tabular datasets we've computed results for.
    
    Args:
        base_dir: Base directory containing the tabular_results folder.
                  If None, uses the default location relative to this file.
    
    Returns:
        DataFrame with results for each dataset
    """
    if base_dir is None:
        # Default to the repository root
        base_dir = Path(__file__).parent.parent.parent / "tabular_results"
    else:
        base_dir = Path(base_dir)
    
    # The three datasets we have results for
    datasets = [
        'pima_diabetes',
        'heart_disease', 
        'breast_cancer_recurrence',
    ]
    
    results = []
    
    for dataset in datasets:
        parquet_file = base_dir / f"{dataset}_with_predictor_answers.parquet"
        
        if not parquet_file.exists():
            print(f"Warning: {parquet_file} not found, skipping...")
            continue
        
        print(f"Processing {dataset}...")
        metrics = calculate_simulability_gain(parquet_file)
        
        results.append({
            'dataset': dataset,
            **metrics,
        })
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    # Calculate simulability gain for all three datasets with answer order separation
    from pathlib import Path

    base_dir = Path(__file__).parent.parent.parent / "data/natural_counterfactuals"
    datasets = ['pima_diabetes', 'heart_disease', 'breast_cancer_recurrence']
    
    print("\n" + "="*80)
    print("SIMULABILITY GAIN RESULTS (SEPARATED BY ANSWER ORDER)")
    print("="*80)
    
    for dataset in datasets:
        parquet_file = base_dir / f"{dataset}_with_predictor_answers.parquet"
        
        if not parquet_file.exists():
            print(f"\nWarning: {parquet_file} not found, skipping...")
            continue
        
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset.upper()}")
        print('='*80)
        
        metrics = calculate_simulability_gain(parquet_file, separate_answer_order=True)
        
        # Overall results
        print(f"\nOVERALL:")
        print(f"  Accuracy with explanation:    {metrics['overall_accuracy_with_explanation']:.4f}")
        print(f"  Accuracy without explanation: {metrics['overall_accuracy_without_explanation']:.4f}")
        print(f"  Simulability gain:            {metrics['overall_simulability_gain']:.4f}")
        print(f"  Normalized gain:              {metrics['overall_normalized_simulatability_gain']:.4f}")
        print(f"  Number of samples:            {metrics['overall_num_samples']}")
        
        # Answer first results
        if 'answer_first_num_samples' in metrics:
            print(f"\nANSWER FIRST (explanation comes after answer):")
            print(f"  Accuracy with explanation:    {metrics['answer_first_accuracy_with_explanation']:.4f}")
            print(f"  Accuracy without explanation: {metrics['answer_first_accuracy_without_explanation']:.4f}")
            print(f"  Simulability gain:            {metrics['answer_first_simulability_gain']:.4f}")
            print(f"  Normalized gain:              {metrics['answer_first_normalized_simulatability_gain']:.4f}")
            print(f"  Number of samples:            {metrics['answer_first_num_samples']}")
        
        # Answer last results
        if 'answer_last_num_samples' in metrics:
            print(f"\nANSWER LAST (explanation comes before answer):")
            print(f"  Accuracy with explanation:    {metrics['answer_last_accuracy_with_explanation']:.4f}")
            print(f"  Accuracy without explanation: {metrics['answer_last_accuracy_without_explanation']:.4f}")
            print(f"  Simulability gain:            {metrics['answer_last_simulability_gain']:.4f}")
            print(f"  Normalized gain:              {metrics['answer_last_normalized_simulatability_gain']:.4f}")
            print(f"  Number of samples:            {metrics['answer_last_num_samples']}")
    
    print("\n" + "="*80)
