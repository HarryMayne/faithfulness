"""
Analyze egregious unfaithfulness by feature.

Identifies which feature changes are most associated with
egregious errors (all predictors unanimously wrong).
"""

import pandas as pd
from collections import defaultdict

from .utils import (
    load_raw_datasets,
    load_prediction_files,
    extract_feature_changes,
    load_moral_machines_raw,
    extract_moral_machines_dimensions,
    bootstrap_rr_ci,
    DATASET_CLASSES,
    DATASET_DISPLAY_NAMES,
)


def add_egregious_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add egregious unfaithfulness flag."""
    def is_egregious(row):
        answers = row['counterfactual_predictor_response_with_explanation_predictor_answers']
        ref_answer = row['counterfactual_reference_response_answer']
        if answers is None or ref_answer is None:
            return False
        unique_answers = set(answers)
        return len(unique_answers) == 1 and answers[0] != ref_answer
    
    df['egregious'] = df.apply(is_egregious, axis=1)
    return df


def analyze_egregious_by_feature(feature_df: pd.DataFrame, n_bootstrap: int = 10000) -> pd.DataFrame:
    """Compute egregious rate statistics by feature with bootstrap CIs."""
    results = []
    
    for dataset in feature_df['dataset'].unique():
        ds_df = feature_df[feature_df['dataset'] == dataset]
        overall_egregious = ds_df.groupby(['dataset'])['egregious'].mean().iloc[0]
        
        for feature in ds_df['feature'].unique():
            feat_df = ds_df[ds_df['feature'] == feature]
            
            changed_df = feat_df[feat_df['changed'] == True]
            unchanged_df = feat_df[feat_df['changed'] == False]
            
            n_changed = len(changed_df)
            n_unchanged = len(unchanged_df)
            
            if n_changed > 0 and n_unchanged > 0:
                # Compute bootstrap CI for RR
                rr, rr_ci_low, rr_ci_high = bootstrap_rr_ci(
                    changed_df['egregious'].values,
                    unchanged_df['egregious'].values,
                    n_bootstrap=n_bootstrap
                )
                rate_changed = changed_df['egregious'].mean()
                rate_unchanged = unchanged_df['egregious'].mean()
            else:
                rr = None
                rr_ci_low = None
                rr_ci_high = None
                rate_changed = changed_df['egregious'].mean() if n_changed > 0 else None
                rate_unchanged = unchanged_df['egregious'].mean() if n_unchanged > 0 else None
            
            results.append({
                'dataset': dataset,
                'feature': feature,
                'overall_egregious_rate': overall_egregious,
                'egregious_when_changed': rate_changed,
                'egregious_when_unchanged': rate_unchanged,
                'n_changed': n_changed,
                'n_unchanged': n_unchanged,
                'relative_risk': rr,
                'rr_ci_low': rr_ci_low,
                'rr_ci_high': rr_ci_high,
                'change_effect': (rate_changed - overall_egregious) if rate_changed else None,
            })
    
    return pd.DataFrame(results).sort_values(['dataset', 'change_effect'], ascending=[True, False])


def print_summary(analysis_df: pd.DataFrame):
    """Print summary of results."""
    print("\n" + "="*80)
    print("EGREGIOUS UNFAITHFULNESS BY FEATURE CHANGE")
    print("="*80)
    
    for dataset in analysis_df['dataset'].unique():
        ds_df = analysis_df[analysis_df['dataset'] == dataset]
        overall = ds_df['overall_egregious_rate'].iloc[0]
        
        print(f"\n--- {dataset.upper()} (overall egregious rate: {overall*100:.1f}%) ---")
        print(f"{'Feature':<25} {'When Changed':>14} {'When Unchanged':>14} {'Effect':>10}")
        print("-" * 65)
        
        for _, row in ds_df.iterrows():
            feat = row['feature'][:24]
            changed = f"{row['egregious_when_changed']*100:.1f}%" if row['egregious_when_changed'] is not None else "N/A"
            unchanged = f"{row['egregious_when_unchanged']*100:.1f}%" if row['egregious_when_unchanged'] is not None else "N/A"
            effect = f"{row['change_effect']*100:+.1f}%" if row['change_effect'] is not None else "N/A"
            
            print(f"{feat:<25} {changed:>14} {unchanged:>14} {effect:>10}")


def analyze_mm_by_dimension(mm_df: pd.DataFrame, n_bootstrap: int = 10000) -> pd.DataFrame:
    """
    Compute egregious rate statistics by moral dimension with bootstrap CIs.
    
    RR = P(egregious | dimension) / P(egregious | all dimensions)
    Bootstrap CI is computed by resampling the full dataset.
    """
    import numpy as np
    
    if len(mm_df) == 0:
        return pd.DataFrame()
    
    overall_egregious = mm_df['egregious'].mean()
    results = []
    
    # Use local RandomState
    rng = np.random.RandomState(42)
    
    for dim in mm_df['dimension'].unique():
        dim_mask = mm_df['dimension'] == dim
        dim_outcomes = mm_df.loc[dim_mask, 'egregious'].values
        all_outcomes = mm_df['egregious'].values
        
        rate = dim_outcomes.mean()
        n = len(dim_outcomes)
        rr = rate / overall_egregious if overall_egregious > 0 else float('nan')
        
        # Bootstrap CI for RR vs overall
        bootstrap_rrs = []
        for _ in range(n_bootstrap):
            # Resample entire dataset with replacement
            sample_idx = rng.choice(len(mm_df), size=len(mm_df), replace=True)
            sample_df = mm_df.iloc[sample_idx]
            
            # Compute dimension rate and overall rate in bootstrap sample
            sample_dim = sample_df[sample_df['dimension'] == dim]['egregious']
            sample_overall = sample_df['egregious'].mean()
            
            if len(sample_dim) > 0 and sample_overall > 0:
                sample_rate = sample_dim.mean()
                bootstrap_rrs.append(sample_rate / sample_overall)
        
        if len(bootstrap_rrs) > 0:
            rr_ci_low = np.percentile(bootstrap_rrs, 2.5)
            rr_ci_high = np.percentile(bootstrap_rrs, 97.5)
        else:
            rr_ci_low = np.nan
            rr_ci_high = np.nan
        
        results.append({
            'dataset': 'moral_machines',
            'feature': dim,  # Using 'feature' for consistency with tabular output
            'overall_egregious_rate': overall_egregious,
            'egregious_rate': rate,
            'n': n,
            'effect': rate - overall_egregious,
            'relative_risk': rr,
            'rr_ci_low': rr_ci_low,
            'rr_ci_high': rr_ci_high,
        })
    
    return pd.DataFrame(results).sort_values('relative_risk', ascending=False)


def print_mm_summary(mm_analysis_df: pd.DataFrame):
    """Print summary of Moral Machines results."""
    if len(mm_analysis_df) == 0:
        return
    
    overall = mm_analysis_df['overall_egregious_rate'].iloc[0]
    
    print(f"\n--- MORAL MACHINES (overall egregious rate: {overall*100:.1f}%) ---")
    print(f"{'Dimension':<20} {'Rate':>8} {'N':>6} {'RR':>8}")
    print("-" * 45)
    
    for _, row in mm_analysis_df.iterrows():
        dim = row['feature']
        rate = f"{row['egregious_rate']*100:.1f}%"
        n = str(int(row['n']))
        rr = f"{row['relative_risk']:.2f}x"
        print(f"{dim:<20} {rate:>8} {n:>6} {rr:>8}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze egregious unfaithfulness by feature')
    parser.add_argument('--input', '-i', nargs='+', 
                        default=['gpt_5_predictions.parquet', 'claude_4_5_predictions.parquet', 'gemini_3_predictions.parquet'],
                        help='Input parquet files')
    parser.add_argument('--output', '-o', default='egregious_by_feature.csv',
                        help='Output CSV file')
    parser.add_argument('--include-mm', action='store_true',
                        help='Include Moral Machines analysis')
    args = parser.parse_args()
    
    print("Loading prediction files...")
    predictions_df = load_prediction_files(files=args.input, filter_tabular=not args.include_mm)
    
    print("\nComputing egregious flags...")
    predictions_df = add_egregious_column(predictions_df)
    print(f"Overall egregious rate: {predictions_df['egregious'].mean()*100:.2f}%")
    
    # --- Tabular datasets analysis ---
    print("\nLoading raw tabular datasets...")
    raw_datasets = load_raw_datasets()
    
    print("\nExtracting feature changes...")
    feature_df = extract_feature_changes(
        predictions_df, 
        raw_datasets, 
        additional_columns=['egregious']
    )
    print(f"Total feature observations: {len(feature_df)}")
    
    print("\nAnalyzing egregious rates by feature...")
    analysis_df = analyze_egregious_by_feature(feature_df)
    
    print_summary(analysis_df)
    
    # --- Moral Machines analysis ---
    mm_analysis_df = pd.DataFrame()
    if args.include_mm:
        print("\n" + "="*60)
        print("MORAL MACHINES ANALYSIS")
        print("="*60)
        
        print("\nLoading Moral Machines raw data...")
        mm_raw = load_moral_machines_raw()
        
        print("\nExtracting dimensions...")
        mm_df = extract_moral_machines_dimensions(
            predictions_df,
            mm_raw,
            additional_columns=['egregious']
        )
        
        if len(mm_df) > 0:
            mm_analysis_df = analyze_mm_by_dimension(mm_df)
            print_mm_summary(mm_analysis_df)
    
    # Save combined results
    all_results = pd.concat([analysis_df, mm_analysis_df], ignore_index=True)
    all_results.to_csv(args.output, index=False)
    print(f"\n✓ Saved analysis to {args.output}")


if __name__ == "__main__":
    main()
