"""
Cross-Model Analysis V2: Analysis-Time Cross-Model Comparison

Instead of pre-generating paired rows, this script computes cross-model
comparisons at analysis time by:
1. Loading all prediction files
2. Building a ground truth index across all models
3. For each (target_model, question), computing accuracy using:
   - Same-model: target's own explanation
   - Cross-model: explanations from different-family models with same ref answer

Usage:
    python -m analysis_scripts.cross_model_analysis_v2 \
        claude_4_5_predictions.parquet \
        gemini_3_predictions.parquet \
        gemma_3_predictions.parquet \
        gpt_5_predictions.parquet \
        qwen_3_predictions.parquet


# Harry notes:
1. Implement this for more models. 


"""

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.schema import CounterfactualDatabase


def get_model_family(model: str) -> str:
    """Extract broad model family from model name."""
    model_lower = model.lower()
    if 'claude' in model_lower: return 'claude'
    if 'gpt' in model_lower: return 'gpt'
    if 'gemini' in model_lower: return 'gemini'
    if 'gemma' in model_lower: return 'gemma'
    if 'qwen' in model_lower: return 'qwen'
    return 'other'


def predictor_matches_model(predictor_name: str, model_name: str, match_type: str) -> bool:
    """Check if predictor should be excluded based on match type.

    Args:
        predictor_name: Name of the predictor model
        model_name: Name of the target/explainer model
        match_type: "none" (no exclusion), "family" (same family), or "exact" (same string)

    Returns:
        True if predictor should be excluded
    """
    if match_type == "none":
        return False
    elif match_type == "exact":
        return predictor_name == model_name
    else:  # family
        return get_model_family(predictor_name) == get_model_family(model_name)


def load_all_predictions(parquet_files: List[str]) -> CounterfactualDatabase:
    """Load and merge all prediction files into a single database."""
    print("Loading prediction files...")
    all_records = []
    
    for f in parquet_files:
        print(f"  Loading {f}...")
        db = CounterfactualDatabase.load_parquet(f)
        all_records.extend(db.records)
        print(f"    {len(db.records)} records")
    
    merged_db = CounterfactualDatabase()
    merged_db.records = all_records
    print(f"Total: {len(all_records)} records")
    return merged_db


def build_ground_truth_index(
    db: CounterfactualDatabase,
    exclude_models: Optional[List[str]] = None
) -> Dict:
    """
    Build lookup: (key, model) -> (cf_answer, ref_answer, predictor_answers)
    
    key = (dataset, orig_q_idx, cf_q_idx, answer_first)
    """
    if exclude_models is None:
        exclude_models = []
    print("\nBuilding ground truth index...")
    index = {}
    
    for record in tqdm(db.records, desc="Indexing"):
        orig_q = record.original_question
        cf = record.counterfactual
        
        # Get key
        dataset = orig_q.dataset
        orig_q_idx = orig_q.question_idx
        cf_q_idx = cf.question_idx
        answer_first = orig_q.answer_first
        key = (dataset, orig_q_idx, cf_q_idx, answer_first)
        
        # Get model info
        ref_response = cf.reference_response
        if not ref_response or not ref_response.model_info:
            continue
        model = ref_response.model_info.model
        # Append thinking effort to model name if present (for ITC experiments)
        if ref_response.model_info.thinking:
            model = f"{model}_{ref_response.model_info.thinking}"

        # Skip excluded models
        if any(excl.lower() in model.lower() for excl in exclude_models):
            continue
        
        # Get answers
        orig_ref = orig_q.reference_response
        if not orig_ref:
            continue
        ref_answer = orig_ref.answer  # Answer on original question
        cf_answer = ref_response.answer  # Answer on counterfactual
        
        if ref_answer is None or cf_answer is None:
            continue
        
        # Get predictor answers (with explanation)
        pred_with = cf.predictor_response_with_explanation
        if not pred_with:
            continue
        if pred_with.predictor_answers is None or len(pred_with.predictor_answers) == 0:
            continue
        if pred_with.predictor_names is None or len(pred_with.predictor_names) == 0:
            continue
        
        # Get predictor answers (without explanation)
        pred_without = cf.predictor_response_without_explanation
        if not pred_without:
            continue
        if pred_without.predictor_answers is None or len(pred_without.predictor_answers) == 0:
            continue
        
        predictor_names = list(pred_with.predictor_names)
        predictor_answers_with = list(pred_with.predictor_answers)
        predictor_answers_without = list(pred_without.predictor_answers)
        
        index[(key, model)] = {
            'cf_answer': cf_answer,
            'ref_answer': ref_answer,
            'predictor_names': predictor_names,
            'predictor_answers': predictor_answers_with,
            'predictor_answers_without': predictor_answers_without,
            'family': get_model_family(model)
        }
    
    print(f"Indexed {len(index)} (key, model) pairs")
    
    # Count unique keys and models
    unique_keys = set(k[0] for k in index.keys())
    unique_models = set(k[1] for k in index.keys())
    print(f"Unique questions: {len(unique_keys)}")
    print(f"Unique models: {len(unique_models)}")
    
    return index


def compute_metrics(question_data_dict: Dict) -> Dict:
    """Compute metrics from a dict of question -> data.

    Args:
        question_data_dict: Dict mapping question keys to dicts with 'same_with',
                           'same_without', and 'cross_with' lists.

    Returns:
        Dict with computed metrics: same_with_acc, cross_with_acc, acc_diff,
        same_norm, cross_norm, norm_diff, without_acc, n_samples.
    """
    all_same_with = []
    all_same_without = []
    all_cross_with = []
    for q_data in question_data_dict.values():
        all_same_with.extend(q_data['same_with'])
        all_same_without.extend(q_data['same_without'])
        all_cross_with.extend(q_data['cross_with'])

    if not all_same_with:
        return {'same_with_acc': 0, 'cross_with_acc': 0, 'acc_diff': 0,
                'same_norm': 0, 'cross_norm': 0, 'norm_diff': 0,
                'without_acc': 0, 'n_samples': 0}

    same_with_acc = np.mean(all_same_with) * 100
    same_without_acc = np.mean(all_same_without) * 100
    cross_with_acc = np.mean(all_cross_with) * 100

    # Raw accuracy difference
    acc_diff = same_with_acc - cross_with_acc

    # Normalized gains
    same_gain = same_with_acc - same_without_acc
    cross_gain = cross_with_acc - same_without_acc
    room = 100.0 - same_without_acc
    same_norm = (same_gain / room * 100) if room > 1e-6 else 0
    cross_norm = (cross_gain / room * 100) if room > 1e-6 else 0

    return {
        'same_with_acc': same_with_acc,
        'cross_with_acc': cross_with_acc,
        'acc_diff': acc_diff,
        'same_norm': same_norm,
        'cross_norm': cross_norm,
        'norm_diff': same_norm - cross_norm,
        'without_acc': same_without_acc,
        'n_samples': len(all_same_with)
    }


def compute_cross_model_accuracy(
    index: Dict,
    predictor_indices: Optional[List[int]] = None,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    exclude_self_predictors: str = "family"
) -> Tuple[pd.DataFrame, Dict]:
    """
    Compute same-model and cross-model accuracy with bootstrap CIs.
    
    Uses cluster-based resampling where each question is a cluster.
    """
    print("\nComputing accuracy metrics with bootstrap CIs...")
    
    # Get all predictor names from first entry
    first_entry = next(iter(index.values()))
    all_predictor_names = first_entry['predictor_names']
    n_predictors = len(all_predictor_names)
    
    if predictor_indices is None:
        predictor_indices = list(range(n_predictors))
    
    print(f"Predictors: {[all_predictor_names[i] for i in predictor_indices]}")
    print(f"Bootstrap iterations: {n_bootstrap}")
    
    # Group by key to find cross-family matches
    key_to_entries = defaultdict(list)
    for (key, model), data in index.items():
        key_to_entries[key].append((model, data))
    
    # Per-model, per-question data for bootstrap
    # Structure: model -> {question_key -> {'same_with': [bool], 'same_without': [bool], 'cross_with': [float]}}
    model_question_data = defaultdict(lambda: defaultdict(lambda: {'same_with': [], 'same_without': [], 'cross_with': []}))
    model_n_skipped = defaultdict(int)
    
    # Process each (key, target_model)
    for (key, target_model), target_data in tqdm(index.items(), desc="Collecting data"):
        target_family = target_data['family']
        target_cf_answer = target_data['cf_answer']
        target_ref_answer = target_data['ref_answer']
        target_pred_answers = target_data['predictor_answers']
        target_pred_answers_without = target_data['predictor_answers_without']
        
        # Cross-model accuracy: find all cross-family explainers with same ref answer
        cross_family_explainers = []
        for model, data in key_to_entries[key]:
            if model == target_model:
                continue
            if data['family'] == target_family:
                continue
            if data['ref_answer'] != target_ref_answer:
                continue
            cross_family_explainers.append((model, data))  # Include explainer model name
        
        # STRICT MODE: Skip if no cross-family explainers
        if not cross_family_explainers:
            model_n_skipped[target_model] += 1
            continue
        
        q_data = model_question_data[target_model][key]
        
        # Same-model accuracy (with and without)
        for pred_idx in predictor_indices:
            if pred_idx < len(target_pred_answers):
                # Skip if predictor matches target (who generated the explanation)
                if exclude_self_predictors != "none":
                    predictor_name = target_data['predictor_names'][pred_idx]
                    if predictor_matches_model(predictor_name, target_model, exclude_self_predictors):
                        continue

                pred_answer = target_pred_answers[pred_idx]
                is_correct = (pred_answer == target_cf_answer)
                q_data['same_with'].append(is_correct)

                pred_answer_without = target_pred_answers_without[pred_idx]
                is_correct_without = (pred_answer_without == target_cf_answer)
                q_data['same_without'].append(is_correct_without)
        
        # Cross-model accuracy
        for pred_idx in predictor_indices:
            scores = []
            for explainer_model, explainer_data in cross_family_explainers:
                # Skip if predictor matches explainer (who generated the explanation)
                if exclude_self_predictors != "none":
                    predictor_name = explainer_data['predictor_names'][pred_idx]
                    if predictor_matches_model(predictor_name, explainer_model, exclude_self_predictors):
                        continue

                explainer_pred_answers = explainer_data['predictor_answers']
                if pred_idx < len(explainer_pred_answers):
                    pred_answer = explainer_pred_answers[pred_idx]
                    is_correct = (pred_answer == target_cf_answer)
                    scores.append(is_correct)
            if scores:
                q_data['cross_with'].append(sum(scores) / len(scores))
    
    # Bootstrap CI calculation
    alpha = 1 - confidence
    lower_pct = (alpha / 2) * 100
    upper_pct = (1 - alpha / 2) * 100

    rows = []
    for model in tqdm(sorted(model_question_data.keys()), desc="Bootstrap CIs"):
        question_data = model_question_data[model]
        question_keys = list(question_data.keys())
        n_questions = len(question_keys)
        
        # Point estimate
        metrics = compute_metrics(question_data)
        
        # Bootstrap
        boot_acc_diff = []
        boot_norm_diff = []
        boot_same_with = []
        boot_cross_with = []
        boot_same_norm = []
        boot_cross_norm = []
        
        for _ in range(n_bootstrap):
            sampled_indices = np.random.randint(0, n_questions, size=n_questions)
            sampled_data = {question_keys[i]: question_data[question_keys[i]] for i in sampled_indices}
            
            b = compute_metrics(sampled_data)
            boot_acc_diff.append(b['acc_diff'])
            boot_norm_diff.append(b['norm_diff'])
            boot_same_with.append(b['same_with_acc'])
            boot_cross_with.append(b['cross_with_acc'])
            boot_same_norm.append(b['same_norm'])
            boot_cross_norm.append(b['cross_norm'])
        
        rows.append({
            'model': model,
            'family': get_model_family(model),
            'without_acc': metrics['without_acc'],
            # Raw accuracy
            'same_with_acc': metrics['same_with_acc'],
            'same_with_ci': (np.percentile(boot_same_with, lower_pct), np.percentile(boot_same_with, upper_pct)),
            'cross_with_acc': metrics['cross_with_acc'],
            'cross_with_ci': (np.percentile(boot_cross_with, lower_pct), np.percentile(boot_cross_with, upper_pct)),
            'acc_diff': metrics['acc_diff'],
            'acc_diff_ci': (np.percentile(boot_acc_diff, lower_pct), np.percentile(boot_acc_diff, upper_pct)),
            # Normalized gain
            'same_norm_gain': metrics['same_norm'],
            'same_norm_ci': (np.percentile(boot_same_norm, lower_pct), np.percentile(boot_same_norm, upper_pct)),
            'cross_norm_gain': metrics['cross_norm'],
            'cross_norm_ci': (np.percentile(boot_cross_norm, lower_pct), np.percentile(boot_cross_norm, upper_pct)),
            'norm_gain_diff': metrics['norm_diff'],
            'diff_ci': (np.percentile(boot_norm_diff, lower_pct), np.percentile(boot_norm_diff, upper_pct)),
            'n_samples': metrics['n_samples'],
            'n_skipped': model_n_skipped[model]
        })

    return pd.DataFrame(rows), dict(model_question_data)


def compute_family_bootstrap(
    model_question_data: Dict,
    n_bootstrap: int = 1000,
    confidence: float = 0.95
) -> pd.DataFrame:
    """
    Compute family-level metrics with bootstrap CIs.

    Pools all questions from models in each family, then bootstraps
    at the question level to get proper family-level CIs.

    Args:
        model_question_data: Dict mapping model -> {question_key -> data}
        n_bootstrap: Number of bootstrap iterations
        confidence: Confidence level for CIs (default 0.95)

    Returns:
        DataFrame with family-level metrics and bootstrap CIs
    """
    alpha = 1 - confidence
    lower_pct = (alpha / 2) * 100
    upper_pct = (1 - alpha / 2) * 100

    # Group by family: family -> {(model, question_key) -> data}
    # Note: same question from different models becomes separate entries
    family_question_data = defaultdict(dict)
    for model, question_data in model_question_data.items():
        family = get_model_family(model)
        for q_key, q_data in question_data.items():
            # Use (model, q_key) as unique identifier to avoid collision
            family_question_data[family][(model, q_key)] = q_data

    rows = []
    for family in sorted(family_question_data.keys()):
        question_data = family_question_data[family]
        question_keys = list(question_data.keys())
        n_questions = len(question_keys)

        # Point estimate
        metrics = compute_metrics(question_data)

        # Bootstrap
        boot_metrics = {key: [] for key in ['same_with_acc', 'cross_with_acc', 'acc_diff',
                                            'same_norm', 'cross_norm', 'norm_diff']}

        for _ in range(n_bootstrap):
            sampled_indices = np.random.randint(0, n_questions, size=n_questions)
            sampled_data = {question_keys[i]: question_data[question_keys[i]] for i in sampled_indices}
            b = compute_metrics(sampled_data)
            for key in boot_metrics:
                boot_metrics[key].append(b[key])

        rows.append({
            'family': family,
            'same_with_acc': metrics['same_with_acc'],
            'same_with_ci': (np.percentile(boot_metrics['same_with_acc'], lower_pct),
                           np.percentile(boot_metrics['same_with_acc'], upper_pct)),
            'cross_with_acc': metrics['cross_with_acc'],
            'cross_with_ci': (np.percentile(boot_metrics['cross_with_acc'], lower_pct),
                            np.percentile(boot_metrics['cross_with_acc'], upper_pct)),
            'acc_diff': metrics['acc_diff'],
            'acc_diff_ci': (np.percentile(boot_metrics['acc_diff'], lower_pct),
                          np.percentile(boot_metrics['acc_diff'], upper_pct)),
            'same_norm_gain': metrics['same_norm'],
            'same_norm_ci': (np.percentile(boot_metrics['same_norm'], lower_pct),
                           np.percentile(boot_metrics['same_norm'], upper_pct)),
            'cross_norm_gain': metrics['cross_norm'],
            'cross_norm_ci': (np.percentile(boot_metrics['cross_norm'], lower_pct),
                            np.percentile(boot_metrics['cross_norm'], upper_pct)),
            'norm_gain_diff': metrics['norm_diff'],
            'diff_ci': (np.percentile(boot_metrics['norm_diff'], lower_pct),
                       np.percentile(boot_metrics['norm_diff'], upper_pct)),
            'n_questions': n_questions,
            'without_acc': metrics['without_acc']
        })

    return pd.DataFrame(rows)


def print_results(df: pd.DataFrame):
    """Print formatted results tables with 95% CIs."""
    
    # === RAW ACCURACY TABLE ===
    print("\n" + "="*120)
    print("CROSS-MODEL COMPARISON: Raw Accuracy with 95% Bootstrap CIs")
    print("="*120)
    
    print(f"\n{'Model':<25} {'Same-Model Acc':<22} {'Cross-Model Acc':<22} {'Diff':<20}")
    print("-"*120)
    
    for _, row in df.iterrows():
        model = row['model'].split('/')[-1][:24]
        same_ci = row['same_with_ci']
        cross_ci = row['cross_with_ci']
        diff_ci = row['acc_diff_ci']
        
        same_str = f"{row['same_with_acc']:.1f}% [{same_ci[0]:.1f}, {same_ci[1]:.1f}]"
        cross_str = f"{row['cross_with_acc']:.1f}% [{cross_ci[0]:.1f}, {cross_ci[1]:.1f}]"
        diff_str = f"{row['acc_diff']:+.1f}% [{diff_ci[0]:+.1f}, {diff_ci[1]:+.1f}]"
        
        print(f"{model:<25} {same_str:<22} {cross_str:<22} {diff_str:<20}")
    
    avg_same_acc = df['same_with_acc'].mean()
    avg_cross_acc = df['cross_with_acc'].mean()
    avg_acc_diff = avg_same_acc - avg_cross_acc
    
    print("-"*120)
    print(f"{'AVERAGE':<25} {avg_same_acc:<21.1f}% {avg_cross_acc:<21.1f}% {avg_acc_diff:<19.1f}%")
    print("="*120)
    
    # === NORMALIZED GAIN TABLE ===
    print("\n" + "="*120)
    print("CROSS-MODEL COMPARISON: Normalized Simulatability Gain with 95% Bootstrap CIs")
    print("="*120)
    
    print(f"\n{'Model':<25} {'Same-Norm':<22} {'Cross-Norm':<22} {'Diff':<20}")
    print("-"*120)
    
    for _, row in df.iterrows():
        model = row['model'].split('/')[-1][:24]
        same_ci = row['same_norm_ci']
        cross_ci = row['cross_norm_ci']
        diff_ci = row['diff_ci']
        
        same_str = f"{row['same_norm_gain']:.1f}% [{same_ci[0]:.1f}, {same_ci[1]:.1f}]"
        cross_str = f"{row['cross_norm_gain']:.1f}% [{cross_ci[0]:.1f}, {cross_ci[1]:.1f}]"
        diff_str = f"{row['norm_gain_diff']:+.1f}% [{diff_ci[0]:+.1f}, {diff_ci[1]:+.1f}]"
        
        print(f"{model:<25} {same_str:<22} {cross_str:<22} {diff_str:<20}")
    
    avg_same_norm = df['same_norm_gain'].mean()
    avg_cross_norm = df['cross_norm_gain'].mean()
    avg_norm_diff = avg_same_norm - avg_cross_norm
    
    print("-"*120)
    print(f"{'AVERAGE':<25} {avg_same_norm:<21.1f}% {avg_cross_norm:<21.1f}% {avg_norm_diff:<19.1f}%")
    print("="*120)
    
    if avg_norm_diff > 0:
        print(f"\n✓ Same-model explanations provide {avg_norm_diff:.1f}% MORE normalized gain on average")
    else:
        print(f"\n✗ Cross-model explanations provide {-avg_norm_diff:.1f}% MORE normalized gain on average")


def print_family_summary(family_df: pd.DataFrame):
    """Print family-level summary with bootstrap CIs.

    Args:
        family_df: DataFrame from compute_family_bootstrap() with family-level
                   metrics and bootstrap CIs.
    """

    # === RAW ACCURACY FAMILY SUMMARY ===
    print("\n" + "="*120)
    print("FAMILY-LEVEL SUMMARY (Raw Accuracy) with 95% Bootstrap CIs")
    print("="*120)

    print(f"\n{'Family':<15} {'Same-Model Acc':<25} {'Cross-Model Acc':<25} {'Diff':<22} {'N':>8}")
    print("-"*120)

    for _, row in family_df.iterrows():
        same_ci = row['same_with_ci']
        cross_ci = row['cross_with_ci']
        diff_ci = row['acc_diff_ci']

        same_str = f"{row['same_with_acc']:.1f}% [{same_ci[0]:.1f}, {same_ci[1]:.1f}]"
        cross_str = f"{row['cross_with_acc']:.1f}% [{cross_ci[0]:.1f}, {cross_ci[1]:.1f}]"
        diff_str = f"{row['acc_diff']:+.1f}% [{diff_ci[0]:+.1f}, {diff_ci[1]:+.1f}]"

        print(f"{row['family']:<15} {same_str:<25} {cross_str:<25} {diff_str:<22} {row['n_questions']:>8}")

    print("="*120)

    # === NORMALIZED GAIN FAMILY SUMMARY ===
    print("\n" + "="*120)
    print("FAMILY-LEVEL SUMMARY (Normalized Gain) with 95% Bootstrap CIs")
    print("="*120)

    print(f"\n{'Family':<15} {'Same-Norm':<25} {'Cross-Norm':<25} {'Diff':<22} {'N':>8}")
    print("-"*120)

    for _, row in family_df.iterrows():
        same_ci = row['same_norm_ci']
        cross_ci = row['cross_norm_ci']
        diff_ci = row['diff_ci']

        same_str = f"{row['same_norm_gain']:.1f}% [{same_ci[0]:.1f}, {same_ci[1]:.1f}]"
        cross_str = f"{row['cross_norm_gain']:.1f}% [{cross_ci[0]:.1f}, {cross_ci[1]:.1f}]"
        diff_str = f"{row['norm_gain_diff']:+.1f}% [{diff_ci[0]:+.1f}, {diff_ci[1]:+.1f}]"

        print(f"{row['family']:<15} {same_str:<25} {cross_str:<25} {diff_str:<22} {row['n_questions']:>8}")

    print("="*120)


def main():
    parser = argparse.ArgumentParser(
        description="Analysis-time cross-model comparison"
    )
    parser.add_argument(
        "parquet_files",
        nargs="+",
        help="Prediction parquet files to analyze"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="cross_model_analysis_v2.csv",
        help="Output CSV path"
    )
    parser.add_argument(
        "--exclude-models",
        nargs="+",
        default=[],
        help="Models to exclude (substring match, e.g., 'nano' 'gemma-1b')"
    )
    parser.add_argument(
        "--exclude-self-predictors",
        choices=["none", "family", "exact"],
        default="family",
        help="Exclude predictors matching explanation source: 'none' (disabled), 'family' (same model family, default), 'exact' (same model string)"
    )

    args = parser.parse_args()
    np.random.seed(42)
    
    # Validate files exist
    for f in args.parquet_files:
        if not Path(f).exists():
            print(f"ERROR: File not found: {f}")
            return
    
    # Load and index
    db = load_all_predictions(args.parquet_files)
    index = build_ground_truth_index(db, exclude_models=args.exclude_models)
    
    if args.exclude_models:
        print(f"Excluded models matching: {args.exclude_models}")

    if args.exclude_self_predictors != "none":
        print(f"Excluding self-predictors by: {args.exclude_self_predictors}")

    # Compute accuracy
    df, model_question_data = compute_cross_model_accuracy(
        index,
        exclude_self_predictors=args.exclude_self_predictors
    )

    # Compute family-level with bootstrap CIs
    family_df = compute_family_bootstrap(model_question_data)

    # Print results
    print_results(df)
    print_family_summary(family_df)

    # Save
    df.to_csv(args.output, index=False)
    print(f"\n✓ Saved: {args.output}")


if __name__ == "__main__":
    main()
