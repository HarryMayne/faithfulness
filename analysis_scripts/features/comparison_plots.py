"""
Compare egregious error rate vs answer change rate by feature.
Creates scatter plots showing which features cause disproportionately
high egregious errors relative to their impact on model predictions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from .utils import (
    DATASET_DISPLAY_NAMES,
    FEATURE_DISPLAY_NAMES,
)


MIN_SAMPLE_SIZE = 200
OUTPUT_DIR = Path('figures')


def load_and_merge_data(egregious_csv='egregious_by_feature.csv', 
                        answer_csv='answer_change_by_feature.csv',
                        min_sample_size=200):
    """Load both CSVs and merge on dataset + feature.
    
    CIs are pre-computed in the CSVs using bootstrap.
    """
    egregious_df = pd.read_csv(egregious_csv)
    answer_df = pd.read_csv(answer_csv)
    
    # Filter to tabular datasets (those with n_changed)
    egregious_df = egregious_df[egregious_df['n_changed'].notna()]
    egregious_df = egregious_df[egregious_df['n_changed'] >= min_sample_size]
    answer_df = answer_df[answer_df['n_changed'] >= min_sample_size]
    
    # Rename columns for merge
    egregious_df = egregious_df.rename(columns={
        'relative_risk': 'egregious_rr',
        'rr_ci_low': 'egr_rr_ci_low',
        'rr_ci_high': 'egr_rr_ci_high',
    })
    answer_df = answer_df.rename(columns={
        'relative_risk': 'answer_change_rr',
        'rr_ci_low': 'ans_rr_ci_low',
        'rr_ci_high': 'ans_rr_ci_high',
    })
    
    merged = pd.merge(
        egregious_df[['dataset', 'feature', 'egregious_rr', 'egr_rr_ci_low', 'egr_rr_ci_high', 'n_changed']],
        answer_df[['dataset', 'feature', 'answer_change_rr', 'ans_rr_ci_low', 'ans_rr_ci_high']],
        on=['dataset', 'feature'],
        how='inner'
    )
    
    return merged


def plot_combined_scatter(df, output_dir=OUTPUT_DIR):
    """Create combined scatter plot."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    dataset_colors = {
        'heart_disease': '#e74c3c',
        'pima_diabetes': '#3498db',
        'breast_cancer_recurrence': '#2ecc71',
        'income': '#9b59b6',
        'attrition': '#f39c12',
        'bank_marketing': '#1abc9c',
    }
    
    for dataset in df['dataset'].unique():
        ds_df = df[df['dataset'] == dataset]
        ax.scatter(
            ds_df['answer_change_rr'], 
            ds_df['egregious_rr'],
            c=dataset_colors.get(dataset, '#333'),
            s=80, alpha=0.7,
            label=DATASET_DISPLAY_NAMES.get(dataset, dataset),
            edgecolors='white', linewidth=0.5
        )
    
    max_val = max(df['answer_change_rr'].max(), df['egregious_rr'].max())
    min_val = min(df['answer_change_rr'].min(), df['egregious_rr'].min())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=1, label='y = x')
    ax.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=1, color='gray', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Answer change RR (impactfulness)', fontsize=14)
    ax.set_ylabel('Egregious error RR', fontsize=14)
    ax.set_title('Feature impact vs egregious error rate', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    corr = df['answer_change_rr'].corr(df['egregious_rr'])
    ax.text(0.95, 0.05, f'r = {corr:.2f}', transform=ax.transAxes,
            fontsize=12, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / 'egregious_vs_answer_change.pdf'
    plt.savefig(output_file, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"Saved: {output_file}")
    return corr


def plot_per_dataset_scatter(df, output_dir=OUTPUT_DIR):
    """Create per-dataset scatter plots with error bars."""
    for dataset in df['dataset'].unique():
        ds_df = df[df['dataset'] == dataset].copy()
        if len(ds_df) < 3:
            continue
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.errorbar(
            ds_df['answer_change_rr'], ds_df['egregious_rr'],
            xerr=[ds_df['answer_change_rr'] - ds_df['ans_rr_ci_low'],
                  ds_df['ans_rr_ci_high'] - ds_df['answer_change_rr']],
            yerr=[ds_df['egregious_rr'] - ds_df['egr_rr_ci_low'],
                  ds_df['egr_rr_ci_high'] - ds_df['egregious_rr']],
            fmt='o', color='#3498db', ecolor='#7fb3d5',
            markersize=8, capsize=3, capthick=1, alpha=0.8
        )
        
        max_val = max(ds_df['answer_change_rr'].max(), ds_df['egregious_rr'].max()) * 1.1
        min_val = min(ds_df['answer_change_rr'].min(), ds_df['egregious_rr'].min()) * 0.9
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=1)
        ax.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
        ax.axvline(x=1, color='gray', linestyle=':', alpha=0.5)
        
        for _, row in ds_df.iterrows():
            feat_name = FEATURE_DISPLAY_NAMES.get(row['feature'], row['feature'][:15])
            ax.annotate(feat_name, (row['answer_change_rr'], row['egregious_rr']),
                       xytext=(5, 5), textcoords='offset points', fontsize=9, alpha=0.9)
        
        ax.set_xlabel('Answer change RR (impactfulness)', fontsize=14)
        ax.set_ylabel('Egregious error RR', fontsize=14)
        ax.set_title(f'{DATASET_DISPLAY_NAMES.get(dataset, dataset)}: Feature impact vs egregious errors', 
                     fontsize=14, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3)
        
        corr = ds_df['answer_change_rr'].corr(ds_df['egregious_rr'])
        ax.text(0.95, 0.05, f'r = {corr:.2f}', transform=ax.transAxes,
                fontsize=12, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        output_file = output_dir / f'egregious_vs_answer_{dataset}.pdf'
        plt.savefig(output_file, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Saved: {output_file} (r = {corr:.2f})")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate comparison scatter plots')
    parser.add_argument('--egregious', default='egregious_by_feature.csv',
                        help='Input CSV from egregious analysis')
    parser.add_argument('--answer-change', default='answer_change_by_feature.csv',
                        help='Input CSV from answer change analysis')
    parser.add_argument('--output-dir', '-o', default='figures',
                        help='Output directory for plots')
    args = parser.parse_args()
    
    print("Loading and merging data...")
    df = load_and_merge_data(
        egregious_csv=args.egregious,
        answer_csv=args.answer_change
    )
    print(f"Merged {len(df)} features")
    
    output_dir = Path(args.output_dir)
    
    print("\nCreating combined scatter plot...")
    corr = plot_combined_scatter(df, output_dir=output_dir)
    
    print("\nCreating per-dataset scatter plots...")
    plot_per_dataset_scatter(df, output_dir=output_dir)
    
    print(f"\nOverall correlation: r = {corr:.3f}")


if __name__ == "__main__":
    main()

