"""
Analyze answer change rate by feature.

Computes P(model_answer_changed | feature X changed)
to measure how "impactful" each feature is for model predictions.
"""

import pandas as pd

from .utils import (
    load_raw_datasets,
    load_prediction_files,
    extract_feature_changes,
    bootstrap_rr_ci,
)


def add_answer_change_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add column indicating if model answer changed from original to counterfactual."""
    df['answer_changed'] = (
        df['original_reference_response_answer'] != 
        df['counterfactual_reference_response_answer']
    )
    return df


def analyze_answer_change_by_feature(feature_df: pd.DataFrame, n_bootstrap: int = 10000) -> pd.DataFrame:
    """
    Compute answer change rate statistics by feature with bootstrap CIs.
    
    For each feature, computes:
    - P(model_answer_changed | feature_value_changed)
    - P(model_answer_changed | feature_value_unchanged)
    - Relative risk = ratio of the above (with bootstrap 95% CI)
    """
    results = []
    
    for dataset in feature_df['dataset'].unique():
        dataset_df = feature_df[feature_df['dataset'] == dataset]
        overall_answer_change_rate = dataset_df['answer_changed'].mean()
        
        for feature in dataset_df['feature'].unique():
            feature_rows = dataset_df[dataset_df['feature'] == feature]
            
            # Split by whether the FEATURE VALUE changed
            rows_where_feature_changed = feature_rows[feature_rows['changed'] == True]
            rows_where_feature_unchanged = feature_rows[feature_rows['changed'] == False]
            
            n_feature_changed = len(rows_where_feature_changed)
            n_feature_unchanged = len(rows_where_feature_unchanged)
            
            if n_feature_changed > 0 and n_feature_unchanged > 0:
                # Compute bootstrap CI for RR
                rr, rr_ci_low, rr_ci_high = bootstrap_rr_ci(
                    rows_where_feature_changed['answer_changed'].values,
                    rows_where_feature_unchanged['answer_changed'].values,
                    n_bootstrap=n_bootstrap
                )
                answer_change_rate_when_feature_changed = rows_where_feature_changed['answer_changed'].mean()
                answer_change_rate_when_feature_unchanged = rows_where_feature_unchanged['answer_changed'].mean()
            else:
                rr = None
                rr_ci_low = None
                rr_ci_high = None
                answer_change_rate_when_feature_changed = rows_where_feature_changed['answer_changed'].mean() if n_feature_changed > 0 else None
                answer_change_rate_when_feature_unchanged = rows_where_feature_unchanged['answer_changed'].mean() if n_feature_unchanged > 0 else None
            
            results.append({
                'dataset': dataset,
                'feature': feature,
                'overall_answer_change_rate': overall_answer_change_rate,
                'answer_change_when_changed': answer_change_rate_when_feature_changed,
                'answer_change_when_unchanged': answer_change_rate_when_feature_unchanged,
                'n_changed': n_feature_changed,
                'n_unchanged': n_feature_unchanged,
                'relative_risk': rr,
                'rr_ci_low': rr_ci_low,
                'rr_ci_high': rr_ci_high,
            })
    
    return pd.DataFrame(results).sort_values(['dataset', 'relative_risk'], ascending=[True, False])


def print_summary(analysis_df: pd.DataFrame):
    """Print summary."""
    print("\n" + "="*80)
    print("ANSWER CHANGE RATE BY FEATURE CHANGE")
    print("="*80)
    
    for dataset in analysis_df['dataset'].unique():
        ds_df = analysis_df[analysis_df['dataset'] == dataset]
        overall = ds_df['overall_answer_change_rate'].iloc[0]
        
        print(f"\n--- {dataset.upper()} (overall: {overall*100:.1f}%) ---")
        print(f"{'Feature':<25} {'When Changed':>14} {'When Unchanged':>16} {'RR':>8}")
        print("-" * 70)
        
        for _, row in ds_df.iterrows():
            feat = row['feature'][:24]
            changed = f"{row['answer_change_when_changed']*100:.1f}%" if row['answer_change_when_changed'] is not None else "N/A"
            unchanged = f"{row['answer_change_when_unchanged']*100:.1f}%" if row['answer_change_when_unchanged'] is not None else "N/A"
            rr = f"{row['relative_risk']:.2f}x" if row['relative_risk'] is not None else "N/A"
            
            print(f"{feat:<25} {changed:>14} {unchanged:>16} {rr:>8}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze answer change rate by feature')
    parser.add_argument('--input', '-i', nargs='+', 
                        default=['gpt_5_predictions.parquet', 'claude_4_5_predictions.parquet', 'gemini_3_predictions.parquet'],
                        help='Input parquet files')
    parser.add_argument('--output', '-o', default='answer_change_by_feature.csv',
                        help='Output CSV file')
    args = parser.parse_args()
    
    print("Loading prediction files...")
    predictions_df = load_prediction_files(files=args.input)
    
    print("\nAdding answer change column...")
    predictions_df = add_answer_change_column(predictions_df)
    print(f"Overall answer change rate: {predictions_df['answer_changed'].mean()*100:.2f}%")
    
    print("\nLoading raw tabular datasets...")
    raw_datasets = load_raw_datasets()
    
    print("\nExtracting feature changes...")
    feature_df = extract_feature_changes(
        predictions_df, 
        raw_datasets, 
        additional_columns=['answer_changed']
    )
    print(f"Total feature observations: {len(feature_df)}")
    
    print("\nAnalyzing answer change rates by feature...")
    analysis_df = analyze_answer_change_by_feature(feature_df)
    
    print_summary(analysis_df)
    
    analysis_df.to_csv(args.output, index=False)
    print(f"\n✓ Saved analysis to {args.output}")


if __name__ == "__main__":
    main()
