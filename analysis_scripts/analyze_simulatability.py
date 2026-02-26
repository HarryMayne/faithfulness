"""
Analyze simulatability gain from predictor answers.

Computes:
1. Accuracy with explanation (predictor matches reference when given explanation)
2. Accuracy without explanation (predictor matches reference without explanation)
3. Simulatability gain (improvement from explanation)
4. Normalized simulatability gain (share of the remaining error gap closed by explanations)

Breaks down results by reference model to see if larger models have better simulatability.

Usage:
    # Standard run on a single parquet file
    python -m analysis_scripts.analyze_simulatability path/to/predictions.parquet

    # Use normalized gain for plots
    python -m analysis_scripts.analyze_simulatability path/to/predictions.parquet --normalized

    # Multi-predictor mode: analyze all *predictor_answers*.parquet files in a directory
    python -m analysis_scripts.analyze_simulatability path/to/directory/ --multi-predictor

    # Consistency filtering: compare all data vs only cases where predictors agree
    # This filters to records where all predictor answers in the WITH explanation
    # case are identical, then plots two curves showing the comparison.
    python -m analysis_scripts.analyze_simulatability path/to/predictions.parquet --consistency-filter

    # Combine flags as needed
    python -m analysis_scripts.analyze_simulatability path/to/predictions.parquet --normalized --consistency-filter

    # Per-dataset breakdown: creates a by_dataset/ subdirectory with results for each dataset
    python -m analysis_scripts.analyze_simulatability path/to/predictions.parquet --by-dataset

    # average --average
    python -m analysis_scripts.analyze_simulatability path/to/predictions.parquet --normalized --average


Options:
    --output            Custom output CSV path (default: auto-generated from input)
    --multi-predictor   Search for all predictor parquet files in directory
    --normalized        Use normalized simulatability gain for plots
    --consistency-filter  Show comparison of all data vs consistent predictions only
    --by-dataset        Create per-dataset breakdown in a 'by_dataset/' subdirectory
    --average
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

from src.schema import CounterfactualDatabase
from human_eval_app.utils import check_predictor_consistency

# Module-level list to collect bootstrap diagnostics
_bootstrap_diagnostics: List[Dict] = []


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
    """Check if predictor should be excluded based on match type."""
    if match_type == "none":
        return False
    elif match_type == "exact":
        return predictor_name == model_name
    else:  # family
        return get_model_family(predictor_name) == get_model_family(model_name)


def filter_to_consistent_records(db: CounterfactualDatabase) -> CounterfactualDatabase:
    """
    Filter database to records where WITH explanation predictors all agree.

    Args:
        db: CounterfactualDatabase to filter

    Returns:
        New CounterfactualDatabase containing only records where all predictor
        answers in the WITH explanation case are consistent (all the same).
    """
    filtered_db = CounterfactualDatabase()

    for record in db.records:
        cf = record.counterfactual
        pred_with = cf.predictor_response_with_explanation

        if pred_with is None:
            # No predictor response - skip this record
            continue

        predictor_answers = pred_with.predictor_answers

        if check_predictor_consistency(predictor_answers):
            filtered_db.records.append(record)

    return filtered_db


def filter_db_by_dataset(db: CounterfactualDatabase, dataset_name: str) -> CounterfactualDatabase:
    """
    Filter database to records from a specific dataset.

    Args:
        db: CounterfactualDatabase to filter
        dataset_name: Name of dataset to keep (e.g., 'heart_disease', 'pima_diabetes')

    Returns:
        New CounterfactualDatabase containing only records from the specified dataset.
    """
    filtered_db = CounterfactualDatabase()
    for record in db.records:
        if record.original_question.dataset == dataset_name:
            filtered_db.records.append(record)
    return filtered_db


def compute_simulatability_metrics(
    records_by_model: Dict[str, List[Tuple[int, bool, bool, bool]]],
    n_bootstrap: int = 1000,
    confidence: float = 0.95
) -> Dict[str, Dict[str, Union[float, Tuple[float, float]]]]:
    """
    Calculate simulatability metrics (gain, accuracy, precision, recall) and their bootstrap confidence intervals.
    This is where all of the bootstrapping happens -- this is non-trivial because of the multiple sources of correlation
    
    Uses cluster-based resampling for variance estimation, but MICRO-averaging for all metrics.
    
    Args:
        records_by_model: Dict mapping model name to list of (question_idx, with_match, without_match, is_diff) tuples
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
        
    Returns:
        Dict mapping model name to results dict:
        {
            'gain': float, 'gain_ci': (low, high),
            'with_acc': float, 'with_acc_ci': (low, high),
            'without_acc': float, 'without_acc_ci': (low, high),
            'norm_gain': float, 'norm_gain_ci': (low, high),
            'precision': float, 'precision_ci': (low, high),
            'precision_without': float, 'precision_without_ci': (low, high),
            'recall': float, 'recall_ci': (low, high),
            'precision_without': float, 'precision_without_ci': (low, high),
            'recall': float, 'recall_ci': (low, high),
            'recall_without': float, 'recall_without_ci': (low, high),
            'diff_pct': float
        }
    """
    np.random.seed(42)
    results = {}
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    for model, records in records_by_model.items():
        # Pre-process: Group by question_idx
        cluster_stats = {}
        for q_idx, w_match, wo_match, is_diff in records:
            if q_idx not in cluster_stats:
                # 0: with_correct (all)
                # 1: without_correct (all)
                # 2: total (all)
                # 3: diff_with_correct
                # 4: diff_without_correct  <-- NEW
                # 5: diff_total
                # 6: same_with_correct
                # 7: same_without_correct  <-- NEW
                # 8: same_total
                cluster_stats[q_idx] = [0, 0, 0, 0, 0, 0, 0, 0, 0]
            stats = cluster_stats[q_idx]
            stats[2] += 1
            if w_match: stats[0] += 1
            if wo_match: stats[1] += 1
            
            if is_diff:
                stats[5] += 1
                if w_match: stats[3] += 1
                if wo_match: stats[4] += 1
            else:
                stats[8] += 1
                if w_match: stats[6] += 1
                if wo_match: stats[7] += 1
        
        # Each row is a cluster stats array
        stats_array = np.array(list(cluster_stats.values()))
        n_clusters = len(stats_array)

        # Record diagnostics
        if n_clusters > 0:
            total_obs = int(np.sum(stats_array[:, 2]))
            avg_cluster_size = total_obs / n_clusters
            _bootstrap_diagnostics.append({
                'model': model,
                'n_clusters': n_clusters,
                'total_observations': total_obs,
                'avg_cluster_size': avg_cluster_size,
            })

        if n_clusters == 0:
            results[model] = {
                'gain': 0.0, 'gain_ci': (0.0, 0.0),
                'with_acc': 0.0, 'with_acc_ci': (0.0, 0.0),
                'without_acc': 0.0, 'without_acc_ci': (0.0, 0.0),
                'norm_gain': 0.0, 'norm_gain_ci': (0.0, 0.0),
                'precision': 0.0, 'precision_ci': (0.0, 0.0),
                'precision_without': 0.0, 'precision_without_ci': (0.0, 0.0),
                'recall': 0.0, 'recall_ci': (0.0, 0.0),
                'recall_without': 0.0, 'recall_without_ci': (0.0, 0.0)
            }
            continue

        # --- Helper to compute micro metrics (Gain/Acc) ---
        def calc_micro_gain_acc(sample_stats):
            total_count = np.sum(sample_stats[:, 2])
            total_with = np.sum(sample_stats[:, 0])
            total_without = np.sum(sample_stats[:, 1])
            
            if total_count == 0:
                return 0.0, 0.0, 0.0
                
            acc_with = total_with / total_count * 100
            acc_without = total_without / total_count * 100
            gain = acc_with - acc_without
            
            return acc_with, acc_without, gain

        # --- Helper to compute micro metrics (Precision/Recall) ---
        def calc_micro_prec_rec(sample_stats):
            total_diff_correct_with = np.sum(sample_stats[:, 3])
            total_diff_correct_wo = np.sum(sample_stats[:, 4])
            total_diff = np.sum(sample_stats[:, 5])
            
            total_same_correct_with = np.sum(sample_stats[:, 6])
            total_same_correct_wo = np.sum(sample_stats[:, 7])
            total_same = np.sum(sample_stats[:, 8])
            
            # Precision = Accuracy on is_diff items
            prec_with = (total_diff_correct_with / total_diff * 100) if total_diff > 0 else 0.0
            prec_wo = (total_diff_correct_wo / total_diff * 100) if total_diff > 0 else 0.0
            
            # Recall = Accuracy on !is_diff items
            rec_with = (total_same_correct_with / total_same * 100) if total_same > 0 else 0.0
            rec_wo = (total_same_correct_wo / total_same * 100) if total_same > 0 else 0.0
            
            return prec_with, prec_wo, rec_with, rec_wo

        # 1. Compute Point Estimates (Observed)
        obs_with, obs_without, obs_gain = calc_micro_gain_acc(stats_array)
        obs_prec_with, obs_prec_wo, obs_rec_with, obs_rec_wo = calc_micro_prec_rec(stats_array)
        
        # Calculate diff percentage
        total_count = np.sum(stats_array[:, 2])
        total_diff = np.sum(stats_array[:, 5])
        obs_diff_pct = (total_diff / total_count * 100) if total_count > 0 else 0.0
        
        # Calculate normalized gain
        norm_gain_denom = 100 - obs_without
        obs_norm_gain = (obs_gain / norm_gain_denom * 100) if norm_gain_denom > 1e-6 else 0.0
        
        # 2. Bootstrap
        boot_gains = []
        boot_with = []
        boot_without = []
        boot_norm_gains = []
        boot_p_with = []
        boot_p_wo = []
        boot_r_with = []
        boot_r_wo = []
        
        for _ in range(n_bootstrap):
            indices = np.random.randint(0, n_clusters, size=n_clusters)
            sample = stats_array[indices]
            
            b_with, b_without, b_gain = calc_micro_gain_acc(sample)
            b_pw, b_pwo, b_rw, b_rwo = calc_micro_prec_rec(sample)
            
            denom = 100 - b_without
            b_norm_gain = (b_gain / denom * 100) if denom > 1e-6 else 0.0
            
            boot_gains.append(b_gain)
            boot_with.append(b_with)
            boot_without.append(b_without)
            boot_norm_gains.append(b_norm_gain)
            boot_p_with.append(b_pw)
            boot_p_wo.append(b_pwo)
            boot_r_with.append(b_rw)
            boot_r_wo.append(b_rwo)
            
        results[model] = {
            'gain': obs_gain,
            'gain_ci': (np.percentile(boot_gains, lower_percentile), np.percentile(boot_gains, upper_percentile)),
            'with_acc': obs_with,
            'with_acc_ci': (np.percentile(boot_with, lower_percentile), np.percentile(boot_with, upper_percentile)),
            'without_acc': obs_without,
            'without_acc_ci': (np.percentile(boot_without, lower_percentile), np.percentile(boot_without, upper_percentile)),
            'norm_gain': obs_norm_gain,
            'norm_gain_ci': (np.percentile(boot_norm_gains, lower_percentile), np.percentile(boot_norm_gains, upper_percentile)),
            'precision': obs_prec_with,
            'precision_ci': (np.percentile(boot_p_with, lower_percentile), np.percentile(boot_p_with, upper_percentile)),
            'precision_without': obs_prec_wo,
            'precision_without_ci': (np.percentile(boot_p_wo, lower_percentile), np.percentile(boot_p_wo, upper_percentile)),
            'recall': obs_rec_with,
            'recall_ci': (np.percentile(boot_r_with, lower_percentile), np.percentile(boot_r_with, upper_percentile)),
            'recall_without': obs_rec_wo,
            'recall_without_ci': (np.percentile(boot_r_wo, lower_percentile), np.percentile(boot_r_wo, upper_percentile)),
            'diff_pct': obs_diff_pct
        }
    
    return results


def analyze_simulatability(
    parquet_path: str = None,
    db: CounterfactualDatabase = None
) -> pd.DataFrame:
    """
    Analyze simulatability gain from predictor responses.

    Args:
        parquet_path: Path to parquet file with predictor answers
        db: Pre-loaded CounterfactualDatabase (if provided, parquet_path is ignored)

    Returns:
        DataFrame with accuracy metrics by reference model
    """
    print("=" * 80)
    print("SIMULATABILITY ANALYSIS")
    print("=" * 80)

    # Load database if not provided
    if db is None:
        if parquet_path is None:
            raise ValueError("Either parquet_path or db must be provided")
        print(f"Loading: {parquet_path}\n")
        db = CounterfactualDatabase.load_parquet(parquet_path)

    print(f"Total records: {len(db.records)}\n")
    
    # Group records by reference model
    results_by_model: Dict[str, Dict[str, int]] = {}
    records_by_model: Dict[str, List[Tuple[int, bool, bool, bool]]] = {}  # For bootstrap
    
    for record in db.records:
        cf = record.counterfactual
        
        # Original Answer for difference check
        orig_ref_response = record.original_question.reference_response
        orig_ref_answer = orig_ref_response.answer if orig_ref_response else None
        
        # Process reference response
        ref_response = cf.reference_response
        if not ref_response or not ref_response.model_info:
            print(f"Skipping record: missing reference response")
            continue
        ref_model = ref_response.model_info.model
        # Append thinking effort to model name if present. Just do this in general to make it more finegrainde. 
        if ref_response.model_info.thinking:
            ref_model = f"{ref_model}_{ref_response.model_info.thinking}"
        ref_answer = ref_response.answer
        
        # Process predictor responses
        pred_with = cf.predictor_response_with_explanation
        pred_without = cf.predictor_response_without_explanation
        if not pred_with or not pred_without:
            print(f"Skipping record: missing predictor responses")
            continue
        pred_answer_with = pred_with.answer
        pred_answer_without = pred_without.answer
        
        # Skip if any answer is None
        if not ref_answer or not pred_answer_with or not pred_answer_without:
            print(f"Skipping record {record.original_question.question_idx}: missing answers")
            continue
            
        is_diff = (ref_answer != orig_ref_answer)
        
        # Initialize model stats if needed
        if ref_model not in results_by_model:
            results_by_model[ref_model] = {
                'total': 0,
                'with_correct': 0,
                'without_correct': 0,
                'both_correct': 0,
                'both_wrong': 0,
                'only_with_correct': 0,
                'only_without_correct': 0,
            }
            records_by_model[ref_model] = []
        
        stats = results_by_model[ref_model]
        stats['total'] += 1
        
        # Check matches
        with_match = (pred_answer_with == ref_answer)
        without_match = (pred_answer_without == ref_answer)
        
        # Store for bootstrap. Cluster by (dataset, question_idx) since idx is only unique within dataset
        cluster_key = (record.original_question.dataset, record.original_question.question_idx)
        records_by_model[ref_model].append((cluster_key, with_match, without_match, is_diff))
        
        if with_match:
            stats['with_correct'] += 1
        if without_match:
            stats['without_correct'] += 1
        
        if with_match and without_match:
            stats['both_correct'] += 1
        elif not with_match and not without_match:
            stats['both_wrong'] += 1
        elif with_match and not without_match:
            stats['only_with_correct'] += 1
        elif without_match and not with_match:
            stats['only_without_correct'] += 1
    
    # Calculate metrics with bootstrap
    print("Calculating metrics and bootstrap confidence intervals (1000 iterations)...")
    metrics_results = compute_simulatability_metrics(records_by_model, n_bootstrap=1000)
    
    # Convert to DataFrame
    rows = []
    for model, stats in results_by_model.items():
        total = stats['total']
        if total == 0:  continue
        res = metrics_results[model]

        rows.append({
            'model': model,
            'total': total,
            'with_explanation_accuracy': res['with_acc'],
            'without_explanation_accuracy': res['without_acc'],
            'simulatability_gain': res['gain'],
            'gain_ci_lower': res['gain_ci'][0],
            'gain_ci_upper': res['gain_ci'][1],
            'with_ci_lower': res['with_acc_ci'][0],
            'with_ci_upper': res['with_acc_ci'][1],
            'without_ci_lower': res['without_acc_ci'][0],
            'without_ci_upper': res['without_acc_ci'][1],
            'normalized_gain': res['norm_gain'],
            'norm_gain_ci_lower': res['norm_gain_ci'][0],
            'norm_gain_ci_upper': res['norm_gain_ci'][1],
            'precision': res['precision'],
            'precision_ci_lower': res['precision_ci'][0],
            'precision_ci_upper': res['precision_ci'][1],
            'precision_without': res['precision_without'],
            'precision_without_ci_lower': res['precision_without_ci'][0],
            'precision_without_ci_upper': res['precision_without_ci'][1],
            'recall': res['recall'],
            'recall_ci_lower': res['recall_ci'][0],
            'recall_ci_upper': res['recall_ci'][1],
            'recall_without': res['recall_without'],
            'recall_without_ci_lower': res['recall_without_ci'][0],
            'recall_without_ci_upper': res['recall_without_ci'][1],
            'diff_pct': res['diff_pct'],
            'both_correct': stats['both_correct'],
            'both_wrong': stats['both_wrong'],
            'only_with_correct': stats['only_with_correct'],
            'only_without_correct': stats['only_without_correct'],
        })
    
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values('model')
    return df


def analyze_simulatability_for_predictor_index(
    db: CounterfactualDatabase,
    predictor_index: int,
    silent: bool = False,
    exclude_self_predictors: str = "none"
) -> pd.DataFrame:
    """
    Analyze simulatability for a specific predictor index within predictor_answers list -- idea is to break it down by p[redictor model

    Args:
        db: CounterfactualDatabase with predictor answers
        predictor_index: Index into predictor_answers list to analyze
        silent: If True, suppress print output

    Returns:
        DataFrame with accuracy metrics by reference model
    """
    if not silent:
        print("=" * 80)
        print(f"SIMULATABILITY ANALYSIS (Predictor Index {predictor_index})")
        print("=" * 80)
        print(f"Total records: {len(db.records)}\n")

    # Group records by reference model
    results_by_model: Dict[str, Dict[str, int]] = {}
    records_by_model: Dict[str, List[Tuple[int, bool, bool, bool]]] = {}

    # loop over each of the predictors
    for record in db.records:
        cf = record.counterfactual # counterfactual

        # Original Answer for difference check. This is only used to calculate Justin's precision and recall metrics.
        orig_ref_response = record.original_question.reference_response
        orig_ref_answer = orig_ref_response.answer if orig_ref_response else None

        # Process reference model responses. Get the counterfactual answer for the reference model
        ref_response = cf.reference_response
        if not ref_response or not ref_response.model_info:
            continue
        ref_model = ref_response.model_info.model
        # Append thinking effort to model name if present
        if ref_response.model_info.thinking:
            ref_model = f"{ref_model}_{ref_response.model_info.thinking}"
        ref_answer = ref_response.answer

        # Process predictor responses - use predictor_answers[predictor_index]
        # Narrow it down to the with and without areas.,
        pred_with = cf.predictor_response_with_explanation # narrow this ddown
        pred_without = cf.predictor_response_without_explanation
        if not pred_with or not pred_without:
            continue

        # Get answers from predictor_answers list at specified index. Continue if empty.
        # only loops of the list of predictor answers (doesn't touch pred_with.answer)
        if (pred_with.predictor_answers is None or
            pred_without.predictor_answers is None or
            predictor_index >= len(pred_with.predictor_answers) or
            predictor_index >= len(pred_without.predictor_answers)):
            continue

        # Exclusion check: skip if predictor matches reference model
        if exclude_self_predictors != "none":
            predictor_name = pred_with.predictor_names[predictor_index]
            if predictor_matches_model(predictor_name, ref_model, exclude_self_predictors):
                continue

        # Exract the answer for the specific predictor index...
        pred_answer_with = pred_with.predictor_answers[predictor_index]
        pred_answer_without = pred_without.predictor_answers[predictor_index]

        # Skip if any answer is None. Might want to change this at a later point... if we want more finegrained breakdown
        # maybe write some separate code for the finegrained breakdown. Pretty easy to do in a notebook.
        if not ref_answer or not pred_answer_with or not pred_answer_without:
            continue

        # used for the precision, recall stuff.
        is_diff = (ref_answer != orig_ref_answer)

        # Initialize model stats if needed (justin's metrics)
        if ref_model not in results_by_model:
            results_by_model[ref_model] = {
                'total': 0,
                'with_correct': 0,
                'without_correct': 0,
                'both_correct': 0,
                'both_wrong': 0,
                'only_with_correct': 0,
                'only_without_correct': 0,
            }
            records_by_model[ref_model] = []

        stats = results_by_model[ref_model]
        stats['total'] += 1

        # Check matches, does the predictor get the CF answer with/without
        with_match = (pred_answer_with == ref_answer)
        without_match = (pred_answer_without == ref_answer)

        # Store for bootstrap. Cluster by (dataset, question_idx) since idx is only unique within dataset
        cluster_key = (record.original_question.dataset, record.original_question.question_idx)
        records_by_model[ref_model].append((cluster_key, with_match, without_match, is_diff))

        if with_match:
            stats['with_correct'] += 1
        if without_match:
            stats['without_correct'] += 1

        if with_match and without_match:
            stats['both_correct'] += 1
        elif not with_match and not without_match:
            stats['both_wrong'] += 1
        elif with_match and not without_match:
            stats['only_with_correct'] += 1
        elif without_match and not with_match:
            stats['only_without_correct'] += 1

    # Calculate metrics with bootstrap
    if not silent:
        print("Calculating metrics and bootstrap confidence intervals (1000 iterations)...")
    metrics_results = compute_simulatability_metrics(records_by_model, n_bootstrap=1000)

    # Convert to DataFrame for the specific predictor model...
    rows = []
    for model, stats in results_by_model.items():
        total = stats['total']
        if total == 0:
            continue
        res = metrics_results[model]

        rows.append({
            'model': model,
            'total': total,
            'with_explanation_accuracy': res['with_acc'],
            'without_explanation_accuracy': res['without_acc'],
            'simulatability_gain': res['gain'],
            'gain_ci_lower': res['gain_ci'][0],
            'gain_ci_upper': res['gain_ci'][1],
            'with_ci_lower': res['with_acc_ci'][0],
            'with_ci_upper': res['with_acc_ci'][1],
            'without_ci_lower': res['without_acc_ci'][0],
            'without_ci_upper': res['without_acc_ci'][1],
            'normalized_gain': res['norm_gain'],
            'norm_gain_ci_lower': res['norm_gain_ci'][0],
            'norm_gain_ci_upper': res['norm_gain_ci'][1],
            'precision': res['precision'],
            'precision_ci_lower': res['precision_ci'][0],
            'precision_ci_upper': res['precision_ci'][1],
            'precision_without': res['precision_without'],
            'precision_without_ci_lower': res['precision_without_ci'][0],
            'precision_without_ci_upper': res['precision_without_ci'][1],
            'recall': res['recall'],
            'recall_ci_lower': res['recall_ci'][0],
            'recall_ci_upper': res['recall_ci'][1],
            'recall_without': res['recall_without'],
            'recall_without_ci_lower': res['recall_without_ci'][0],
            'recall_without_ci_upper': res['recall_without_ci'][1],
            'diff_pct': res['diff_pct'],
            'both_correct': stats['both_correct'],
            'both_wrong': stats['both_wrong'],
            'only_with_correct': stats['only_with_correct'],
            'only_without_correct': stats['only_without_correct'],
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values('model')
    return df


def analyze_simulatability_averaged(
    db: CounterfactualDatabase,
    predictor_indices: List[int],
    silent: bool = False,
    exclude_self_predictors: str = "none"
) -> pd.DataFrame:
    """
    Analyze simulatability averaged across multiple predictors with correct bootstrap CIs.

    Pools all predictor observations into the same clusters (by original_question_idx), as we normally do,
    then bootstraps by resampling clusters. Correctly handles correlation between predictors answering the same questions.

    Args:
        db: CounterfactualDatabase with predictor answers
        predictor_indices: List of indices into predictor_answers list to include
        silent: If True, suppress print output

    Returns:
        DataFrame with accuracy metrics by reference model (averaged across predictors)
    """
    if not silent:
        print("=" * 80)
        print(f"SIMULATABILITY ANALYSIS (Averaged across {len(predictor_indices)} predictors)")
        print("=" * 80)
        print(f"Total records: {len(db.records)}\n")

    # Group records by reference model
    # Each observation is a (counterfactual, predictor) pair
    results_by_model: Dict[str, Dict[str, int]] = {}
    records_by_model: Dict[str, List[Tuple[int, bool, bool, bool]]] = {}

    for record in db.records:
        cf = record.counterfactual
        # Cluster by (dataset, question_idx) since idx is only unique within dataset
        q_idx = (record.original_question.dataset, record.original_question.question_idx)

        # Original answer for is_diff calculation
        orig_ref_response = record.original_question.reference_response
        orig_ref_answer = orig_ref_response.answer if orig_ref_response else None

        # Process reference model response
        ref_response = cf.reference_response
        if not ref_response or not ref_response.model_info:
            continue
        ref_model = ref_response.model_info.model
        if ref_response.model_info.thinking:
            ref_model = f"{ref_model}_{ref_response.model_info.thinking}"
        ref_answer = ref_response.answer

        # Get predictor response objects
        pred_with = cf.predictor_response_with_explanation
        pred_without = cf.predictor_response_without_explanation
        if not pred_with or not pred_without:
            continue

        # Check predictor_answers exists
        if pred_with.predictor_answers is None or pred_without.predictor_answers is None:
            continue

        # Calculate is_diff once per counterfactual (same for all predictors)
        is_diff = (ref_answer != orig_ref_answer)

        # Initialize model stats if needed
        if ref_model not in results_by_model:
            results_by_model[ref_model] = {
                'total': 0,
                'with_correct': 0,
                'without_correct': 0,
                'both_correct': 0,
                'both_wrong': 0,
                'only_with_correct': 0,
                'only_without_correct': 0,
            }
            records_by_model[ref_model] = []

        # Loop over ALL predictor indices - each becomes a separate observation
        # All share the same q_idx, so they'll be clustered together for bootstrap
        for pred_idx in predictor_indices:
            # Skip if index out of range for this record
            if (pred_idx >= len(pred_with.predictor_answers) or
                pred_idx >= len(pred_without.predictor_answers)):
                continue

            # Exclusion check: skip if predictor matches reference model
            if exclude_self_predictors != "none":
                pred_name = pred_with.predictor_names[pred_idx]
                if predictor_matches_model(pred_name, ref_model, exclude_self_predictors):
                    continue

            pred_answer_with = pred_with.predictor_answers[pred_idx]
            pred_answer_without = pred_without.predictor_answers[pred_idx]

            # Skip if any answer is None
            if not ref_answer or not pred_answer_with or not pred_answer_without:
                continue

            stats = results_by_model[ref_model]
            stats['total'] += 1

            # Check matches
            with_match = (pred_answer_with == ref_answer)
            without_match = (pred_answer_without == ref_answer)

            # Store for bootstrap - same q_idx for all predictors on same counterfactual
            records_by_model[ref_model].append((q_idx, with_match, without_match, is_diff))

            if with_match:
                stats['with_correct'] += 1
            if without_match:
                stats['without_correct'] += 1

            if with_match and without_match:
                stats['both_correct'] += 1
            elif not with_match and not without_match:
                stats['both_wrong'] += 1
            elif with_match and not without_match:
                stats['only_with_correct'] += 1
            elif without_match and not with_match:
                stats['only_without_correct'] += 1

    # Calculate metrics with bootstrap
    # The bootstrap will resample by q_idx, keeping all predictors together
    if not silent:
        print("Calculating metrics and bootstrap confidence intervals (1000 iterations)...")
    metrics_results = compute_simulatability_metrics(records_by_model, n_bootstrap=1000)

    # Convert to DataFrame
    rows = []
    for model, stats in results_by_model.items():
        total = stats['total']
        if total == 0:
            continue
        res = metrics_results[model]

        rows.append({
            'model': model,
            'total': total,
            'n_predictors': len(predictor_indices),
            'with_explanation_accuracy': res['with_acc'],
            'without_explanation_accuracy': res['without_acc'],
            'simulatability_gain': res['gain'],
            'gain_ci_lower': res['gain_ci'][0],
            'gain_ci_upper': res['gain_ci'][1],
            'with_ci_lower': res['with_acc_ci'][0],
            'with_ci_upper': res['with_acc_ci'][1],
            'without_ci_lower': res['without_acc_ci'][0],
            'without_ci_upper': res['without_acc_ci'][1],
            'normalized_gain': res['norm_gain'],
            'norm_gain_ci_lower': res['norm_gain_ci'][0],
            'norm_gain_ci_upper': res['norm_gain_ci'][1],
            'precision': res['precision'],
            'precision_ci_lower': res['precision_ci'][0],
            'precision_ci_upper': res['precision_ci'][1],
            'precision_without': res['precision_without'],
            'precision_without_ci_lower': res['precision_without_ci'][0],
            'precision_without_ci_upper': res['precision_without_ci'][1],
            'recall': res['recall'],
            'recall_ci_lower': res['recall_ci'][0],
            'recall_ci_upper': res['recall_ci'][1],
            'recall_without': res['recall_without'],
            'recall_without_ci_lower': res['recall_without_ci'][0],
            'recall_without_ci_upper': res['recall_without_ci'][1],
            'diff_pct': res['diff_pct'],
            'both_correct': stats['both_correct'],
            'both_wrong': stats['both_wrong'],
            'only_with_correct': stats['only_with_correct'],
            'only_without_correct': stats['only_without_correct'],
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values('model')
    return df


def print_results(df: pd.DataFrame):
    """Print formatted results"""
    print("\n" + "=" * 80)
    print("RESULTS BY REFERENCE MODEL")
    print("=" * 80)

    if df.empty:
        print("\n  (No results - all records were excluded)")
        return

    for _, row in df.iterrows():
        print(f"\n{row['model']}:")
        print(f"  Total samples: {row['total']}")
        print(f"  With explanation:    {row['with_explanation_accuracy']:.2f}% (95% CI: [{row['with_ci_lower']:.2f}%, {row['with_ci_upper']:.2f}%])")
        print(f"  Without explanation: {row['without_explanation_accuracy']:.2f}% (95% CI: [{row['without_ci_lower']:.2f}%, {row['without_ci_upper']:.2f}%])")
        print(f"  Simulatability gain: {row['simulatability_gain']:+.2f}% (95% CI: [{row['gain_ci_lower']:+.2f}%, {row['gain_ci_upper']:+.2f}%])")
        print(f"  Normalized gain:     {row['normalized_gain']:+.2f}% (95% CI: [{row['norm_gain_ci_lower']:+.2f}%, {row['norm_gain_ci_upper']:+.2f}%])")
        print(f"  Precision (With):    {row['precision']:.2f}% (95% CI: [{row['precision_ci_lower']:.2f}%, {row['precision_ci_upper']:.2f}%])")
        print(f"  Precision (Without): {row['precision_without']:.2f}% (95% CI: [{row['precision_without_ci_lower']:.2f}%, {row['precision_without_ci_upper']:.2f}%])")
        print(f"  Recall (With):       {row['recall']:.2f}% (95% CI: [{row['recall_ci_lower']:.2f}%, {row['recall_ci_upper']:.2f}%])")
        print(f"  Recall (Without):    {row['recall_without']:.2f}% (95% CI: [{row['recall_without_ci_lower']:.2f}%, {row['recall_without_ci_upper']:.2f}%])")
        print(f"\n  Breakdown:")
        print(f"    Both correct:          {row['both_correct']:4d} ({row['both_correct']/row['total']*100:.1f}%)")
        print(f"    Both wrong:            {row['both_wrong']:4d} ({row['both_wrong']/row['total']*100:.1f}%)")
        print(f"    Only with correct:     {row['only_with_correct']:4d} ({row['only_with_correct']/row['total']*100:.1f}%)")
        print(f"    Only without correct:  {row['only_without_correct']:4d} ({row['only_without_correct']/row['total']*100:.1f}%)")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Average simulatability gain: {df['simulatability_gain'].mean():.2f}%")
    print(f"Std dev simulatability gain: {df['simulatability_gain'].std():.2f}%")
    print(f"Min simulatability gain: {df['simulatability_gain'].min():.2f}% ({df.loc[df['simulatability_gain'].idxmin(), 'model']})")
    print(f"Max simulatability gain: {df['simulatability_gain'].max():.2f}% ({df.loc[df['simulatability_gain'].idxmax(), 'model']})")
    print(f"Average normalized gain: {df['normalized_gain'].mean():.2f}% of gap") # changed to correct name


def extract_model_size(model_name: str) -> float:
    """
    Extract model size in billions of parameters from model name.
    
    Examples:
        "Qwen/Qwen3-0.6B" -> 0.6
        "Qwen/Qwen3-1.7B" -> 1.7
        "Qwen/Qwen3-14B" -> 14.0
        
    Returns:
        Model size in billions, or None if not parseable
    """
    import re
    # Look for pattern like "0.6B", "1.7B", "14B", "32B"
    match = re.search(r'(\d+(?:\.\d+)?)[Bb]', model_name)
    if match:
        return float(match.group(1))
    return None


def extract_predictor_name_from_parquet(parquet_path: str) -> str:
    """
    Extract predictor model name directly from the parquet file.
    Assumes all records use the same predictor model.
    
    Args:
        parquet_path: Path to parquet file
        
    Returns:
        Predictor model name (e.g., "google/gemma-2-27b-it")
    """
    db = CounterfactualDatabase.load_parquet(parquet_path)
    if len(db.records) == 0:
        return "Unknown"
    record = db.records[0]
    if record.counterfactual.predictor_response_with_explanation and \
       record.counterfactual.predictor_response_with_explanation.model_info:
        return record.counterfactual.predictor_response_with_explanation.model_info.model
    if record.counterfactual.predictor_response_without_explanation and \
       record.counterfactual.predictor_response_without_explanation.model_info:
        return record.counterfactual.predictor_response_without_explanation.model_info.model
    return "Unknown"


def plot_multi_predictor_comparison(
    dfs_by_predictor: Dict[str, pd.DataFrame],
    output_path: Path,
    use_normalized: bool = False
):
    """
    Plot comparison of multiple predictors.
    """
    plt.figure(figsize=(12, 8))
    
    if use_normalized:
        y_col = 'normalized_gain'
        y_label = 'Normalized Simulatability Gain (%)'
        ci_lower = 'norm_gain_ci_lower'
        ci_upper = 'norm_gain_ci_upper'
        title = 'Normalized Simulatability Gain by Predictor'
    else:
        y_col = 'simulatability_gain'
        y_label = 'Simulatability Gain (%)'
        ci_lower = 'gain_ci_lower'
        ci_upper = 'gain_ci_upper'
        title = 'Simulatability Gain by Predictor'
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(dfs_by_predictor)))
    
    all_sizes = [] # Collect all sizes for x-axis scaling
    for (predictor, df), color in zip(sorted(dfs_by_predictor.items()), colors): # Sort for consistent color mapping
        # Extract model sizes
        df = df.copy()
        df['model_size_b'] = df['model'].apply(extract_model_size)
        df_plot = df[df['model_size_b'].notna()].sort_values('model_size_b')
        
        if len(df_plot) == 0:
            continue
            
        # Calculate errors
        y_err_lower = df_plot[y_col] - df_plot[ci_lower]
        y_err_upper = df_plot[ci_upper] - df_plot[y_col]
        
        plt.errorbar(
            df_plot['model_size_b'],
            df_plot[y_col],
            yerr=[y_err_lower, y_err_upper],
            fmt='o-',
            capsize=3,
            label=predictor,
            color=color,
            alpha=0.8
        )
        
        all_sizes.extend(df_plot['model_size_b'].tolist())
    
    # Add horizontal line at y=0
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.xlabel('Reference Model Size (Billions of Parameters)', fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc='best', title='Predictor Model')
    
    # Set exact xticks at the model sizes
    if all_sizes:
        unique_sizes = sorted(set(all_sizes))
        plt.xticks(unique_sizes, [str(s) for s in unique_sizes])
        
        # Use log scale for x-axis if range is large
        if max(all_sizes) / min(all_sizes) > 10:
            plt.xscale('log')
            plt.xlabel('Reference Model Size (Billions of Parameters, log scale)', fontsize=12)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Multi-predictor plot saved to: {output_path}")
    plt.close()


def plot_accuracy_comparison_bars(
    dfs_by_predictor: Dict[str, pd.DataFrame],
    output_path: Path
):
    """
    Plot grouped stacked bars showing improvements in accuracy.
    
    Structure:
    - Segments: Each group on x-axis is a reference model size
    - Bars within segment: Different predictors
    - Stack:
        - Base bar: min(with_acc, without_acc)
        - Stack bar: abs(with_acc - without_acc)
        - Color: Green if with > without (gain), Red if without > with (loss)
    """
    plt.figure(figsize=(14, 8))
    
    # 1. Collect all model sizes and predictors
    all_sizes = set()
    for df in dfs_by_predictor.values():
        df['model_size_b'] = df['model'].apply(extract_model_size)
        all_sizes.update(df[df['model_size_b'].notna()]['model_size_b'].tolist())
    
    sorted_sizes = sorted(list(all_sizes))
    size_to_idx = {s: i for i, s in enumerate(sorted_sizes)}
    n_groups = len(sorted_sizes)
    
    predictor_names = sorted(dfs_by_predictor.keys())
    n_predictors = len(predictor_names)
    
    # Layout configuration
    group_width = 0.8  # Total width of a group
    bar_width = group_width / n_predictors
    
    # Colors
    color_base = '#e0e0e0'  # Light gray for base accuracy
    color_gain = '#2ca02c'  # Green for improvement
    color_loss = '#d62728'  # Red for regression
    
    # Create axes
    ax = plt.gca()
    
    # 2. Plot bars for each predictor
    for i, predictor in enumerate(predictor_names):
        df = dfs_by_predictor[predictor].copy()
        
        # Calculate x positions for this predictor's bars across all groups
        # Center of group is at integer indices 0, 1, 2...
        # Offset shifts bars to be side-by-side centered around group center
        offset = (i - (n_predictors - 1) / 2) * bar_width
        
        # Prepare data arrays aligned with sorted_sizes
        base_heights = []
        stack_heights = []
        stack_colors = []
        x_positions = []
        
        for size in sorted_sizes:
            row = df[df['model_size_b'] == size]
            if len(row) == 0:
                continue
            
            row = row.iloc[0]
            with_acc = row['with_explanation_accuracy']
            without_acc = row['without_explanation_accuracy']
            
            # Logic: Base is min, Stack is difference
            base = min(with_acc, without_acc)
            diff = with_acc - without_acc
            stack = abs(diff)
            
            base_heights.append(base)
            stack_heights.append(stack)
            stack_colors.append(color_gain if diff >= 0 else color_loss)
            x_positions.append(size_to_idx[size] + offset)
            
        # Plot Base (Neutral)
        ax.bar(x_positions, base_heights, width=bar_width, color=color_base, 
               edgecolor='white', linewidth=0.5, label='Base Accuracy' if i == 0 else "")
        
        # Plot Stack (Gain/Loss)
        ax.bar(x_positions, stack_heights, width=bar_width, bottom=base_heights,
               color=stack_colors, edgecolor='white', linewidth=0.5, label='Gain/Loss' if i == 0 else "")
        
        # Add predictor label at the bottom of the group or legend?
        # Standard legend is better for predictors
    
    # Custom Legend for Predictors is tricky because bars are colored by Gain/Loss
    # Strategy: Use proxy artists for Gain/Loss and maybe text labels or a separate legend for Predictors?
    # Actually, standard grouped bar plots color by Category (Predictor).
    # BUT here we color by Meaning (Gain/Loss).
    # So we can't distinguish predictors by color. 
    # WE MUST distinguish predictors by position or pattern, or legend needs to map position.
    # User asked for "grouped by model size, all predictors side-by-side".
    # Usually this implies Predictor A is always Left, Predictor B is Right.
    # Without color distinction, it's hard to tell which bar is which.
    # CORRECT FIX: We should use different "Base" colors for predictors? 
    # Or satisfy the "Stacked" request but distinct colors for predictors?
    # User Plan said: "Bottom: Neutral/Blue. Top: Green/Red". This implies uniform color for bottom.
    # To distinguish predictors, we can add labels below the x-axis or use different shades of neutral.
    # Let's try iterating shades of gray/blue for the base for different predictors?
    # Or just rely on the x-axis labels if we label each bar? 
    # Labelling each bar might be too crowded.
    # Let's use a subtle color variation for the base bars per predictor.
    
    # Re-draw with predictor-specific base colors to distinguish them
    base_palette = plt.cm.Blues(np.linspace(0.3, 0.8, n_predictors))
    
    # Clear and redo with new base colors
    ax.clear()
    
    for i, predictor in enumerate(predictor_names):
        df = dfs_by_predictor[predictor].copy()
        offset = (i - (n_predictors - 1) / 2) * bar_width
        col_base = base_palette[i]
        
        # ... (Same data prep logic) ...
        base_heights = []
        stack_heights = []
        stack_colors = []
        x_positions = []
        
        for size in sorted_sizes:
            row = df[df['model_size_b'] == size] if not df[df['model_size_b'] == size].empty else None
            if row is None: continue
            row = row.iloc[0]
            
            w, wo = row['with_explanation_accuracy'], row['without_explanation_accuracy']
            base_heights.append(min(w, wo))
            stack = abs(w - wo)
            stack_heights.append(stack)
            stack_colors.append(color_gain if w >= wo else color_loss)
            x_positions.append(size_to_idx[size] + offset)
            
        # Plot Base (Predictor specific color)
        ax.bar(x_positions, base_heights, width=bar_width, color=col_base, 
               edgecolor='black', linewidth=0.5, label=predictor)
        
        # Plot Stack (Gain/Loss overlay)
        ax.bar(x_positions, stack_heights, width=bar_width, bottom=base_heights,
               color=stack_colors, edgecolor='black', linewidth=0.5)

    # X-Axis Labels
    ax.set_xticks(range(n_groups))
    ax.set_xticklabels([f"{s}B" for s in sorted_sizes], fontsize=11)
    ax.set_xlabel("Reference Model Size", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    
    # Title
    ax.set_title("Simulatability Gain by Predictor & Model Size\n(Base = Min Accuracy, Top Segment = Gain/Loss)", 
                 fontsize=14, fontweight='bold')
    
    # Grid
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Legend
    # 1. Predictor Colors (Base bars)
    handles, labels = ax.get_legend_handles_labels()
    # 2. Add Dummy handles for Gain/Loss meaning
    from matplotlib.patches import Patch
    handles.append(Patch(facecolor=color_gain, edgecolor='black', label='Gain (With > Without)'))
    handles.append(Patch(facecolor=color_loss, edgecolor='black', label='Loss (Without > With)'))
    
    ax.legend(handles=handles, title="Predictors & Metrics", loc='lower right', fontsize=9, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"\n✓ Grouped accuracy plot saved to: {output_path}")
    plt.close()


def plot_simulatability_vs_size(
    df: pd.DataFrame,
    output_path: Path,
    use_normalized: bool = False,
    df_filtered: pd.DataFrame = None
):
    """
    Plot simulatability gain vs model size with error bars.

    Args:
        df: DataFrame with results
        output_path: Path to save plot
        use_normalized: If True, plot normalized simulatability gain instead of raw gain
        df_filtered: Optional filtered DataFrame for comparison (plots second curve)
    """
    # Extract model sizes
    df = df.copy()
    df['model_size_b'] = df['model'].apply(extract_model_size)

    # Filter to models with parseable sizes
    df_plot = df[df['model_size_b'].notna()].copy()

    if len(df_plot) == 0:
        print("\nWarning: Could not extract model sizes for plotting")
        return

    # Sort by size
    df_plot = df_plot.sort_values('model_size_b')

    # Calculate error bars (distance from mean to CI bounds)
    if use_normalized:
        y_col = 'normalized_gain'
        y_label = 'Normalized Simulatability Gain (%)'
        ci_lower = 'norm_gain_ci_lower'
        ci_upper = 'norm_gain_ci_upper'
        title = 'Normalized Simulatability Gain vs Reference Model Size'
    else:
        y_col = 'simulatability_gain'
        y_label = 'Simulatability Gain (%)'
        ci_lower = 'gain_ci_lower'
        ci_upper = 'gain_ci_upper'
        title = 'Simulatability Gain vs Reference Model Size'

    df_plot['error_lower'] = df_plot[y_col] - df_plot[ci_lower]
    df_plot['error_upper'] = df_plot[ci_upper] - df_plot[y_col]

    # Create plot
    plt.figure(figsize=(10, 6))

    # Determine if we're plotting comparison
    is_comparison = df_filtered is not None and len(df_filtered) > 0

    # Plot main curve (all data)
    plt.errorbar(
        df_plot['model_size_b'],
        df_plot[y_col],
        yerr=[df_plot['error_lower'], df_plot['error_upper']],
        fmt='o-',
        capsize=5,
        capthick=2,
        markersize=8,
        linewidth=2,
        color='#1f77b4',
        label='All data' if is_comparison else y_label
    )

    # Plot filtered curve if provided
    if is_comparison:
        df_filt = df_filtered.copy()
        df_filt['model_size_b'] = df_filt['model'].apply(extract_model_size)
        df_filt_plot = df_filt[df_filt['model_size_b'].notna()].copy()

        if len(df_filt_plot) > 0:
            df_filt_plot = df_filt_plot.sort_values('model_size_b')
            df_filt_plot['error_lower'] = df_filt_plot[y_col] - df_filt_plot[ci_lower]
            df_filt_plot['error_upper'] = df_filt_plot[ci_upper] - df_filt_plot[y_col]

            plt.errorbar(
                df_filt_plot['model_size_b'],
                df_filt_plot[y_col],
                yerr=[df_filt_plot['error_lower'], df_filt_plot['error_upper']],
                fmt='s--',
                capsize=5,
                capthick=2,
                markersize=8,
                linewidth=2,
                color='#ff7f0e',
                label='Consistent predictions only'
            )

        # Update title for comparison
        title = title + ' (Consistency Comparison)'

    # Add horizontal line at y=0
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    plt.xlabel('Model Size (Billions of Parameters)', fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)

    # Set exact xticks at the model sizes
    unique_sizes = df_plot['model_size_b'].unique()
    plt.xticks(unique_sizes, [str(s) for s in unique_sizes])

    # Use log scale for x-axis if range is large
    if df_plot['model_size_b'].max() / df_plot['model_size_b'].min() > 10:
        plt.xscale('log')
        plt.xlabel('Model Size (Billions of Parameters, log scale)', fontsize=12)
        # For log scale, still set the ticks
        plt.xticks(unique_sizes, [str(s) for s in unique_sizes])

    plt.tight_layout()
    
    # Save plot
    if use_normalized:
        if '_simulatability' in output_path.stem:
            plot_path = output_path.parent / f"{output_path.stem.replace('_simulatability', '_normalized_simulatability')}.png"
        else:
            plot_path = output_path.parent / f"{output_path.stem}_normalized_simulatability.png"
    else:
        if '_simulatability' in output_path.stem:
            plot_path = output_path.parent / f"{output_path.stem}.png"
        else:
            plot_path = output_path.parent / f"{output_path.stem}_simulatability.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {plot_path}")
    plt.close()


def plot_precision_recall_vs_size(df: pd.DataFrame, output_path: Path):
    """
    Plot Precision and Recall vs model size.
    """
    df = df.copy()
    df['model_size_b'] = df['model'].apply(extract_model_size)
    df_plot = df[df['model_size_b'].notna()].sort_values('model_size_b')
    
    if len(df_plot) == 0:
        return

    # Calculate error bars
    df_plot['prec_err_low'] = df_plot['precision'] - df_plot['precision_ci_lower']
    df_plot['prec_err_high'] = df_plot['precision_ci_upper'] - df_plot['precision']
    df_plot['rec_err_low'] = df_plot['recall'] - df_plot['recall_ci_lower']
    df_plot['rec_err_high'] = df_plot['recall_ci_upper'] - df_plot['recall']
    
    # Calculate error bars for Without
    df_plot['prec_wo_err_low'] = df_plot['precision_without'] - df_plot['precision_without_ci_lower']
    df_plot['prec_wo_err_high'] = df_plot['precision_without_ci_upper'] - df_plot['precision_without']
    df_plot['rec_wo_err_low'] = df_plot['recall_without'] - df_plot['recall_without_ci_lower']
    df_plot['rec_wo_err_high'] = df_plot['recall_without_ci_upper'] - df_plot['recall_without']
    
    # Calculate Data Percentages
    mean_diff_pct = df_plot['diff_pct'].mean()
    mean_same_pct = 100.0 - mean_diff_pct

    plt.figure(figsize=(10, 6))
    
    # Precision colors
    color_prec = '#1f77b4'  # Blue
    color_rec = '#ff7f0e'   # Orange
    
    # Plot Precision (With) - Solid
    plt.errorbar(
        df_plot['model_size_b'],
        df_plot['precision'],
        yerr=[df_plot['prec_err_low'], df_plot['prec_err_high']],
        fmt='o-',
        label=f'Precision (With Explanation) - Acc on Changed (~{mean_diff_pct:.0f}%)',
        color=color_prec, capsize=5, linewidth=2
    )
    
    # Plot Precision (Without) - Dashed
    plt.errorbar(
        df_plot['model_size_b'],
        df_plot['precision_without'],
        yerr=[df_plot['prec_wo_err_low'], df_plot['prec_wo_err_high']],
        fmt='o--',
        label='Precision (Without Explanation)',
        color=color_prec, capsize=5, linewidth=2, alpha=0.7
    )
    
    # Plot Recall (With) - Solid
    plt.errorbar(
        df_plot['model_size_b'],
        df_plot['recall'],
        yerr=[df_plot['rec_err_low'], df_plot['rec_err_high']],
        fmt='s-',
        label=f'Recall (With Explanation) - Acc on Unchanged (~{mean_same_pct:.0f}%)',
        color=color_rec, capsize=5, linewidth=2
    )
    
    # Plot Recall (Without) - Dashed
    plt.errorbar(
        df_plot['model_size_b'],
        df_plot['recall_without'],
        yerr=[df_plot['rec_wo_err_low'], df_plot['rec_wo_err_high']],
        fmt='s--',
        label='Recall (Without Explanation)',
        color=color_rec, capsize=5, linewidth=2, alpha=0.7
    )
    
    plt.xlabel('Model Size (Billions of Parameters)', fontsize=12)
    plt.ylabel('Score (%)', fontsize=12)
    plt.title('Simulatability Precision & Recall vs Model Size', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc='best')
    plt.ylim(0, 105)
    
    unique_sizes = df_plot['model_size_b'].unique()
    plt.xticks(unique_sizes, [str(s) for s in unique_sizes])
    
    if df_plot['model_size_b'].max() / df_plot['model_size_b'].min() > 10:
        plt.xscale('log')
        plt.xlabel('Model Size (Billions, log scale)', fontsize=12)
        plt.xticks(unique_sizes, [str(s) for s in unique_sizes])
        
    plt.tight_layout()
    if '_simulatability' in output_path.stem:
        plot_path = output_path.parent / f"{output_path.stem.replace('_simulatability', '_precision_recall')}.png"
    else:
        plot_path = output_path.parent / f"{output_path.stem}_precision_recall.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Precision/Recall plot saved to: {plot_path}")
    plt.close()


def print_bootstrap_diagnostics_summary():
    """Print summary of bootstrap diagnostics collected during the run."""
    if not _bootstrap_diagnostics:
        return

    print("\n" + "=" * 80)
    print("BOOTSTRAP DIAGNOSTICS SUMMARY")
    print("=" * 80)

    # Per-model breakdown
    print("\nPer-model cluster statistics:")
    print(f"  {'Model':<40} {'Clusters':>10} {'Observations':>14} {'Avg Size':>10}")
    print("  " + "-" * 76)
    for diag in _bootstrap_diagnostics:
        print(f"  {diag['model']:<40} {diag['n_clusters']:>10} {diag['total_observations']:>14} {diag['avg_cluster_size']:>10.1f}")

    # Aggregate statistics
    total_clusters = sum(d['n_clusters'] for d in _bootstrap_diagnostics)
    total_obs = sum(d['total_observations'] for d in _bootstrap_diagnostics)
    avg_clusters_per_model = total_clusters / len(_bootstrap_diagnostics)
    avg_obs_per_cluster = total_obs / total_clusters if total_clusters > 0 else 0

    print("\n  " + "-" * 76)
    print(f"  {'TOTAL/AVERAGE':<40} {avg_clusters_per_model:>10.1f} {total_obs:>14} {avg_obs_per_cluster:>10.1f}")

    # Interpretation guidance
    print("\n  Interpretation:")
    min_clusters = min(d['n_clusters'] for d in _bootstrap_diagnostics)
    if min_clusters < 30:
        print(f"  ⚠ WARNING: Minimum clusters = {min_clusters}. Recommend ≥50 for stable CIs.")
    elif min_clusters < 50:
        print(f"  ⚠ CAUTION: Minimum clusters = {min_clusters}. CIs may be somewhat unstable.")
    else:
        print(f"  ✓ Minimum clusters = {min_clusters}. Cluster count appears adequate.")

    print("=" * 80 + "\n")


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description="Analyze simulatability gain from predictor answers"
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to parquet file with predictor answers, or directory containing multiple predictor files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: same folder as input with '_simulatability_analysis.csv' suffix)"
    )
    parser.add_argument(
        "--multi-predictor",
        action="store_true",
        help="Search for all predictor parquet files in the directory and create comparison plot"
    )
    parser.add_argument(
        "--normalized",
        action="store_true",
        help="Use normalized simulatability gain for plots"
    )
    parser.add_argument(
        "--consistency-filter",
        action="store_true",
        help="Show comparison of all data vs consistent predictions only (where all predictors agree)"
    )
    parser.add_argument(
        "--by-dataset",
        action="store_true",
        help="Create per-dataset breakdown in a 'by_dataset/' subdirectory"
    )
    parser.add_argument(
        "--average",
        action="store_true",
        help="Average results across predictor model"
    )
    parser.add_argument(
        "--exclude-self-predictors",
        choices=["none", "family", "exact"],
        default="none",
        help="Exclude predictors matching reference model: 'none' (disabled, default), 'family' (same model family), 'exact' (same model string)"
    )

    args = parser.parse_args()

    if args.exclude_self_predictors != "none":
        print(f"Excluding self-predictors by: {args.exclude_self_predictors}")

    input_path = Path(args.path)
    np.random.seed(42)
    # Check if path is a directory
    if input_path.is_dir() or args.multi_predictor:
        # Multi-predictor mode
        if input_path.is_file():
            search_dir = input_path.parent
        else:
            search_dir = input_path
            
        # Find all predictor parquet files
        parquet_files = list(search_dir.glob("*predictor_answers*.parquet"))
        
        if len(parquet_files) == 0:
            print(f"ERROR: No predictor parquet files found in: {search_dir}")
            return
        
        print(f"Found {len(parquet_files)} predictor parquet files:")
        for f in parquet_files:
            print(f"  - {f.name}")
        print()
        
        # Analyze each file
        dfs_by_predictor = {}
        dfs_filtered_by_predictor = {} if args.consistency_filter else None

        for parquet_file in parquet_files:
            predictor_name = extract_predictor_name_from_parquet(str(parquet_file))
            print(f"\n{'='*80}")
            print(f"Analyzing predictor: {predictor_name}")
            print(f"{'='*80}")

            # Load database once per file
            db = CounterfactualDatabase.load_parquet(str(parquet_file))

            # Analyze full dataset
            df = analyze_simulatability(db=db)
            dfs_by_predictor[predictor_name] = df

            # Print results
            print_results(df)

            # Save individual CSV
            base_name = parquet_file.stem.replace('_with_predictor_answers', '')
            output_csv = parquet_file.parent / f"{base_name}_simulatability_analysis.csv"
            df.to_csv(output_csv, index=False)
            print(f"\n✓ Results saved to: {output_csv}")

            # Handle consistency filtering for this predictor
            df_filtered = None
            if args.consistency_filter:
                n_original = len(db.records)
                filtered_db = filter_to_consistent_records(db)
                n_filtered = len(filtered_db.records)

                print(f"\n--- Consistency Filtering ---")
                print(f"Records retained: {n_filtered}/{n_original} ({100*n_filtered/n_original:.1f}%)")

                if n_filtered > 0:
                    df_filtered = analyze_simulatability(db=filtered_db)
                    dfs_filtered_by_predictor[predictor_name] = df_filtered

                    # Save filtered results
                    filtered_csv = output_csv.parent / f"{output_csv.stem}_consistent_only.csv"
                    df_filtered.to_csv(filtered_csv, index=False)
                    print(f"✓ Filtered results saved to: {filtered_csv}")
                else:
                    print("Warning: No records remain after consistency filtering.")

            # Create individual plots (with optional filtered data)
            plot_simulatability_vs_size(df, output_csv, use_normalized=args.normalized, df_filtered=df_filtered)
            plot_precision_recall_vs_size(df, output_csv)

        # Create multi-predictor comparison plot
        plot_name = "normalized_simulatability_comparison_all_predictors.png" if args.normalized else "simulatability_comparison_all_predictors.png"
        output_plot = search_dir / plot_name
        plot_multi_predictor_comparison(dfs_by_predictor, output_plot, use_normalized=args.normalized)

        # Create grouped accuracy bars plot
        output_bars = search_dir / "simulatability_accuracy_bars_all_predictors.png"
        plot_accuracy_comparison_bars(dfs_by_predictor, output_bars)

        # If consistency filtering is active, also create comparison plots for filtered data
        if args.consistency_filter and len(dfs_filtered_by_predictor) > 0:
            plot_name_filtered = "normalized_simulatability_comparison_all_predictors_consistent_only.png" if args.normalized else "simulatability_comparison_all_predictors_consistent_only.png"
            output_plot_filtered = search_dir / plot_name_filtered
            plot_multi_predictor_comparison(dfs_filtered_by_predictor, output_plot_filtered, use_normalized=args.normalized)

        # Dataset breakdown
        if args.by_dataset:
            # Load all databases to get unique datasets
            dbs_by_predictor = {}
            all_datasets = set()
            for parquet_file in parquet_files:
                predictor_name = extract_predictor_name_from_parquet(str(parquet_file))
                db = CounterfactualDatabase.load_parquet(str(parquet_file))
                dbs_by_predictor[predictor_name] = db
                all_datasets.update(r.original_question.dataset for r in db.records)

            unique_datasets = sorted(all_datasets)

            # Create output directory
            breakdown_dir = search_dir / "by_dataset"
            breakdown_dir.mkdir(exist_ok=True)

            print(f"\n{'='*80}")
            print(f"DATASET BREAKDOWN (MULTI-PREDICTOR)")
            print(f"{'='*80}")
            print(f"Found {len(unique_datasets)} datasets: {unique_datasets}")
            print(f"Output directory: {breakdown_dir}")

            # Collect all results for overview
            all_dataset_dfs = []

            for dataset_name in unique_datasets:
                print(f"\n{'='*80}")
                print(f"DATASET: {dataset_name}")
                print(f"{'='*80}")

                for predictor_name, db in dbs_by_predictor.items():
                    print(f"\n--- {predictor_name} ---")

                    # Filter to this dataset
                    dataset_db = filter_db_by_dataset(db, dataset_name)
                    print(f"Records: {len(dataset_db.records)}")

                    if len(dataset_db.records) == 0:
                        continue

                    # Run analysis
                    df_dataset = analyze_simulatability(db=dataset_db)

                    # Add dataset and predictor columns, collect for overview
                    df_dataset_copy = df_dataset.copy()
                    df_dataset_copy['dataset'] = dataset_name
                    df_dataset_copy['predictor'] = predictor_name
                    all_dataset_dfs.append(df_dataset_copy)

                    # Print results
                    print_results(df_dataset)

                    # Save CSV with predictor name in filename
                    predictor_short = predictor_name.replace("/", "_")
                    dataset_csv = breakdown_dir / f"{dataset_name}_{predictor_short}_simulatability_analysis.csv"
                    df_dataset.to_csv(dataset_csv, index=False)
                    print(f"✓ Saved: {dataset_csv}")

                    # Create plots
                    plot_simulatability_vs_size(df_dataset, dataset_csv, use_normalized=args.normalized)
                    plot_precision_recall_vs_size(df_dataset, dataset_csv)

            # Save multi-indexed overview CSV
            if all_dataset_dfs:
                overview_df = pd.concat(all_dataset_dfs, ignore_index=True)
                overview_df = overview_df.set_index(['dataset', 'predictor', 'model'])
                overview_df = overview_df.sort_index()
                overview_csv = breakdown_dir / "overview.csv"
                overview_df.to_csv(overview_csv)
                print(f"\n✓ Overview saved to: {overview_csv}")

    else:
        # Single file mode
        parquet_path = input_path

        if not parquet_path.exists():
            print(f"ERROR: Parquet file not found: {parquet_path}")
            return

        # Determine output path
        if args.output:
            output_csv = Path(args.output)
        else:
            # Default: same folder, replace suffix
            base_name = parquet_path.stem.replace('_with_predictor_answers', '')
            output_csv = parquet_path.parent / f"{base_name}_simulatability_analysis.csv"

        # Load database once
        db = CounterfactualDatabase.load_parquet(str(parquet_path))

        # Average mode: analyze with all predictors pooled, correctly bootstrapped CIs
        if args.average:
            # Apply consistency filtering if requested
            if args.consistency_filter:
                n_original = len(db.records)
                db = filter_to_consistent_records(db)
                n_filtered = len(db.records)
                print(f"Consistency filtering: {n_filtered}/{n_original} records retained ({100*n_filtered/n_original:.1f}%)\n")
                if n_filtered == 0:
                    print("ERROR: No records remain after consistency filtering")
                    return

            # Get predictor names from first record
            first_record = db.records[0]
            pred_with = first_record.counterfactual.predictor_response_with_explanation
            if pred_with is None or pred_with.predictor_names is None:
                print("ERROR: No predictor_names found in database")
                return

            predictor_names = pred_with.predictor_names
            n_predictors = len(predictor_names)
            predictor_indices = list(range(n_predictors))
            print(f"Found {n_predictors} predictors: {predictor_names}\n")

            # Optional: show per-predictor breakdown for transparency
            all_dfs = []
            for i, pred_name in enumerate(predictor_names):
                print(f"\n{'='*80}")
                print(f"Analyzing predictor {i+1}/{n_predictors}: {pred_name}")
                print(f"{'='*80}")

                df_i = analyze_simulatability_for_predictor_index(
                    db, predictor_index=i, silent=False,
                    exclude_self_predictors=args.exclude_self_predictors
                )

                # Print results for this predictor
                print_results(df_i)

                # Only add to breakdown if there are results
                if not df_i.empty:
                    df_i['predictor'] = pred_name
                    df_i['predictor_index'] = i
                    all_dfs.append(df_i)

            # Combine into breakdown DataFrame
            base_name = parquet_path.stem.replace('_with_predictor_answers', '')
            suffix = ""
            if args.consistency_filter:
                suffix += "_consistent_only"
            if args.exclude_self_predictors != "none":
                suffix += f"_exclude_{args.exclude_self_predictors}"

            if all_dfs:
                breakdown_df = pd.concat(all_dfs, ignore_index=True)

                # Reorder columns to put predictor first
                cols = ['predictor', 'predictor_index'] + [c for c in breakdown_df.columns if c not in ['predictor', 'predictor_index']]
                breakdown_df = breakdown_df[cols]

                # Save breakdown CSV
                breakdown_csv = parquet_path.parent / f"{base_name}_simulatability_breakdown_by_predictor{suffix}.csv"
                breakdown_df.to_csv(breakdown_csv, index=False)
                print(f"\n✓ Per-predictor breakdown saved to: {breakdown_csv}")
            else:
                print("\n⚠ No per-predictor breakdown to save (all predictors were excluded)")

            # Calculate averaged results with correct bootstrap CIs
            # This pools all predictors into the same clusters, then bootstraps by original_question_idx
            print(f"\n{'='*80}")
            print("AVERAGED ANALYSIS (with correct bootstrap CIs)")
            print(f"{'='*80}")
            avg_df = analyze_simulatability_averaged(
                db, predictor_indices, silent=False,
                exclude_self_predictors=args.exclude_self_predictors
            )

            # Print averaged results
            print_results(avg_df)

            # Save averaged CSV and create plots only if there are results
            if not avg_df.empty:
                avg_csv = parquet_path.parent / f"{base_name}_simulatability_averaged{suffix}.csv"
                avg_df.to_csv(avg_csv, index=False)
                print(f"\n✓ Averaged results saved to: {avg_csv}")

                # Create plots for averaged results
                plot_simulatability_vs_size(avg_df, avg_csv, use_normalized=args.normalized)
                plot_precision_recall_vs_size(avg_df, avg_csv)
            else:
                print("\n⚠ No averaged results to save (all records were excluded)")

        # Analyze full dataset (non-average mode only)
        if not args.average:
            df = analyze_simulatability(db=db)

            # Print results
            print_results(df)

            # Save to CSV
            df.to_csv(output_csv, index=False)
            print(f"\nResults saved to: {output_csv}")

            # Handle consistency filtering
            df_filtered = None
            if args.consistency_filter:
                n_original = len(db.records)
                filtered_db = filter_to_consistent_records(db)
                n_filtered = len(filtered_db.records)

                print(f"\n{'='*80}")
                print(f"CONSISTENCY FILTERING")
                print(f"{'='*80}")
                print(f"Records retained: {n_filtered}/{n_original} ({100*n_filtered/n_original:.1f}%)")

                if n_filtered > 0:
                    df_filtered = analyze_simulatability(db=filtered_db)
                    print_results(df_filtered)

                    # Save filtered results
                    filtered_csv = output_csv.parent / f"{output_csv.stem}_consistent_only.csv"
                    df_filtered.to_csv(filtered_csv, index=False)
                    print(f"\n✓ Filtered results saved to: {filtered_csv}")
                else:
                    print("\nWarning: No records remain after consistency filtering. Skipping filtered analysis.")

            # Create plot (with optional filtered data for comparison)
            plot_simulatability_vs_size(df, output_csv, use_normalized=args.normalized, df_filtered=df_filtered)

            # Create Precision/Recall plot
            plot_precision_recall_vs_size(df, output_csv)

        # Dataset breakdown
        if args.by_dataset:
            unique_datasets = sorted(set(r.original_question.dataset for r in db.records))

            # Create output directory
            breakdown_dir = output_csv.parent / "by_dataset"
            breakdown_dir.mkdir(exist_ok=True)

            print(f"\n{'='*80}")
            print(f"DATASET BREAKDOWN{' (AVERAGED)' if args.average else ''}")
            print(f"{'='*80}")
            print(f"Found {len(unique_datasets)} datasets: {unique_datasets}")
            print(f"Output directory: {breakdown_dir}")

            # Get predictor_indices for average mode
            if args.average:
                first_record = db.records[0]
                pred_with = first_record.counterfactual.predictor_response_with_explanation
                if pred_with is None or pred_with.predictor_names is None:
                    print("ERROR: No predictor_names found in database")
                    return
                predictor_indices = list(range(len(pred_with.predictor_names)))

            # Collect all dataset results for overview
            all_dataset_dfs = []

            for dataset_name in unique_datasets:
                print(f"\n--- {dataset_name} ---")

                # Filter to this dataset
                dataset_db = filter_db_by_dataset(db, dataset_name)
                print(f"Records: {len(dataset_db.records)}")

                if len(dataset_db.records) == 0:
                    continue

                # Run analysis (averaged or standard)
                if args.average:
                    df_dataset = analyze_simulatability_averaged(
                        dataset_db, predictor_indices, silent=True,
                        exclude_self_predictors=args.exclude_self_predictors
                    )
                else:
                    df_dataset = analyze_simulatability(db=dataset_db)

                # Print results
                print_results(df_dataset)

                # Skip saving/plotting if no results
                if df_dataset.empty:
                    print(f"  (No results for {dataset_name} - all records were excluded)")
                    continue

                # Add dataset column and collect for overview
                df_dataset_copy = df_dataset.copy()
                df_dataset_copy['dataset'] = dataset_name
                all_dataset_dfs.append(df_dataset_copy)

                # Save CSV
                dataset_csv = breakdown_dir / f"{dataset_name}_simulatability_analysis.csv"
                df_dataset.to_csv(dataset_csv, index=False)
                print(f"✓ Saved: {dataset_csv}")

                # Create plots
                plot_simulatability_vs_size(df_dataset, dataset_csv, use_normalized=args.normalized)
                plot_precision_recall_vs_size(df_dataset, dataset_csv)

            # Save multi-indexed overview CSV
            if all_dataset_dfs:
                overview_df = pd.concat(all_dataset_dfs, ignore_index=True)
                overview_df = overview_df.set_index(['dataset', 'model'])
                overview_df = overview_df.sort_index()
                overview_csv = breakdown_dir / "overview.csv"
                overview_df.to_csv(overview_csv)
                print(f"\n✓ Overview saved to: {overview_csv}")

    # Print bootstrap diagnostics summary at the very end
    print_bootstrap_diagnostics_summary()


if __name__ == "__main__":
    main()
