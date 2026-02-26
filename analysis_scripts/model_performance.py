"""
Analyze reference model performance against ground truth.

Computes accuracy metrics for reference models by comparing their answers to ground truth.
Breaks down results by reference model and dataset.

Usage:
    python -m analysis_scripts.model_performance file1.parquet file2.parquet ...

For each parquet file, outputs a CSV with columns:
    model, dataset, total, correct, accuracy, accuracy_ci_lower, accuracy_ci_upper
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

from src.schema import CounterfactualDatabase


def compute_bootstrap_ci(
    records: List[Tuple[int, bool]],
    n_bootstrap: int = 5, #1000
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Calculate accuracy with bootstrap confidence intervals.

    Uses cluster-based resampling for variance estimation (micro-averaging).

    Args:
        records: List of (question_idx, is_correct) tuples
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level

    Returns:
        Tuple of (accuracy, ci_lower, ci_upper)
    """
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    # Group by question_idx for cluster-based bootstrap
    cluster_stats = {}
    for q_idx, is_correct in records:
        if q_idx not in cluster_stats:
            cluster_stats[q_idx] = [0, 0]  # [correct_count, total_count]
        cluster_stats[q_idx][1] += 1
        if is_correct:
            cluster_stats[q_idx][0] += 1

    stats_array = np.array(list(cluster_stats.values()))
    n_clusters = len(stats_array)

    if n_clusters == 0:
        return 0.0, 0.0, 0.0

    # Helper to compute micro accuracy
    def calc_micro_accuracy(sample_stats):
        total_correct = np.sum(sample_stats[:, 0])
        total_count = np.sum(sample_stats[:, 1])
        if total_count == 0:
            return 0.0
        return total_correct / total_count * 100

    # Observed accuracy
    obs_accuracy = calc_micro_accuracy(stats_array)

    # Bootstrap
    boot_accuracies = []
    for _ in range(n_bootstrap):
        indices = np.random.randint(0, n_clusters, size=n_clusters)
        sample = stats_array[indices]
        boot_acc = calc_micro_accuracy(sample)
        boot_accuracies.append(boot_acc)

    ci_lower = np.percentile(boot_accuracies, lower_percentile)
    ci_upper = np.percentile(boot_accuracies, upper_percentile)

    return obs_accuracy, ci_lower, ci_upper


def analyze_model_performance(parquet_path: str) -> pd.DataFrame:
    """
    Analyze reference model performance against ground truth, broken down by model and dataset.

    Args:
        parquet_path: Path to parquet file

    Returns:
        DataFrame with columns: model, dataset, total, correct, accuracy, accuracy_ci_lower, accuracy_ci_upper
    """
    print("=" * 80)
    print(f"Analyzing: {parquet_path}")
    print("=" * 80)

    db = CounterfactualDatabase.load_parquet(parquet_path)
    print(f"Total records: {len(db.records)}\n")

    # Group records by (model, dataset)
    # Key: (model, dataset) -> {'records': [(q_idx, is_correct), ...], 'correct': int, 'total': int}
    stats_by_model_dataset: Dict[Tuple[str, str], Dict] = {}

    skipped_no_ground_truth = 0
    skipped_moral_machines = 0
    skipped_missing_data = 0

    for record in db.records:
        dataset = record.original_question.dataset

        # Skip moral_machines dataset (no ground truth)
        if dataset == 'moral_machines':
            skipped_moral_machines += 1
            continue

        # Check if ground truth exists
        ground_truth = record.original_question.ground_truth
        if ground_truth is None:
            skipped_no_ground_truth += 1
            continue

        # Process reference response
        ref_response = record.original_question.reference_response
        if not ref_response or not ref_response.model_info:
            skipped_missing_data += 1
            continue

        ref_model = ref_response.model_info.model
        # Append thinking effort to model name if present
        if ref_response.model_info.thinking:
            ref_model = f"{ref_model}_{ref_response.model_info.thinking}"

        ref_answer = ref_response.answer
        if not ref_answer:
            skipped_missing_data += 1
            continue

        key = (ref_model, dataset)

        # Initialize if needed
        if key not in stats_by_model_dataset:
            stats_by_model_dataset[key] = {
                'records': [],
                'correct': 0,
                'total': 0,
            }

        stats = stats_by_model_dataset[key]
        stats['total'] += 1

        # Check if answer matches ground truth
        is_correct = (ref_answer == ground_truth)
        stats['records'].append((record.original_question.question_idx, is_correct))

        if is_correct:
            stats['correct'] += 1

    # Print skipped records summary
    print(f"Skipped records:")
    print(f"  Moral machines (no ground truth): {skipped_moral_machines}")
    print(f"  Missing ground truth: {skipped_no_ground_truth}")
    print(f"  Missing reference data: {skipped_missing_data}")
    print(f"  Total skipped: {skipped_moral_machines + skipped_no_ground_truth + skipped_missing_data}\n")

    # Calculate metrics with bootstrap for each (model, dataset) pair
    n_pairs = len(stats_by_model_dataset)
    print(f"Calculating metrics and bootstrap confidence intervals for {n_pairs} (model, dataset) pairs...")
    rows = []
    for i, ((model, dataset), stats) in enumerate(stats_by_model_dataset.items()):
        if stats['total'] == 0:
            continue

        print(f"  [{i+1}/{n_pairs}] {model} / {dataset} ({stats['total']} samples)...")
        accuracy, ci_lower, ci_upper = compute_bootstrap_ci(stats['records'], n_bootstrap=1000)

        rows.append({
            'model': model,
            'dataset': dataset,
            'total': stats['total'],
            'correct': stats['correct'],
            'accuracy': accuracy,
            'accuracy_ci_lower': ci_lower,
            'accuracy_ci_upper': ci_upper,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(['model', 'dataset'])
    return df


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description="Analyze reference model performance against ground truth"
    )
    parser.add_argument(
        "parquet_files",
        type=str,
        nargs='+',
        help="Path(s) to parquet file(s)"
    )

    args = parser.parse_args()
    np.random.seed(42)

    all_dfs = []

    for parquet_path in args.parquet_files:
        parquet_path = Path(parquet_path)

        if not parquet_path.exists():
            print(f"ERROR: Parquet file not found: {parquet_path}")
            continue

        # Analyze (returns df with model, dataset, total, correct, accuracy, ci columns)
        df = analyze_model_performance(str(parquet_path))

        if len(df) == 0:
            print(f"No results for {parquet_path.name}")
            continue

        # Save CSV
        output_csv = parquet_path.parent / f"{parquet_path.stem}_performance_analysis.csv"
        df.to_csv(output_csv, index=False)
        print(f"Saved to: {output_csv}\n")

        # Add source file column and collect
        df['source_file'] = parquet_path.name
        all_dfs.append(df)

    # Save combined results if multiple files
    if len(all_dfs) > 1:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        # Use parent of first file for combined output
        first_parent = Path(args.parquet_files[0]).parent
        combined_csv = first_parent / "combined_performance_analysis.csv"
        combined_df.to_csv(combined_csv, index=False)
        print(f"Combined results saved to: {combined_csv}")


if __name__ == "__main__":
    main()
