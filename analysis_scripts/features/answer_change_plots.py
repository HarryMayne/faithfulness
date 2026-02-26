"""
Visualize answer change rate by feature for each dataset.
Generates vertical bar charts with error bars.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from .utils import (
    DATASET_DISPLAY_NAMES,
    wrap_text,
    get_display_name,
)


MIN_SAMPLE_SIZE = 200
OUTPUT_DIR = Path('figures')


def load_and_prepare_data(csv_path='answer_change_by_feature.csv'):
    """Load data with pre-computed bootstrap CIs from CSV."""
    df = pd.read_csv(csv_path)
    df = df[df['n_changed'] >= MIN_SAMPLE_SIZE].copy()
    
    # CIs for rates - compute from RR CIs and overall rate
    # For now, use the rate directly with bootstrap RR CIs as a proxy
    # RR CI comes from CSV (pre-computed by analysis script using bootstrap)
    
    return df


def plot_dataset_features(df, dataset_name, output_dir=OUTPUT_DIR):
    """Create forest plot for one dataset showing RR with CIs."""
    ds_df = df[df['dataset'] == dataset_name].copy()
    
    if len(ds_df) == 0:
        return
    
    # Filter to rows with valid RR
    ds_df = ds_df[ds_df['relative_risk'].notna()].copy()
    ds_df = ds_df.sort_values('relative_risk', ascending=True)
    ds_df['display_name'] = ds_df['feature'].apply(
        lambda x: wrap_text(get_display_name(x))
    )
    
    n_features = len(ds_df)
    fig, ax = plt.subplots(figsize=(8, max(6, n_features * 0.4)))
    
    y_pos = np.arange(n_features)
    colors = ['#e74c3c' if v > 1.0 else '#2ecc71' for v in ds_df['relative_risk']]
    
    # Plot points
    ax.scatter(ds_df['relative_risk'], y_pos, color=colors, s=100, zorder=3,
               edgecolors='#333333', linewidth=0.5)
    
    # Plot CIs
    for i, (_, row) in enumerate(ds_df.iterrows()):
        if pd.notna(row['rr_ci_low']) and pd.notna(row['rr_ci_high']):
            ax.hlines(y=i, xmin=row['rr_ci_low'], xmax=row['rr_ci_high'],
                     color=colors[i], linewidth=2.5, zorder=2)
            ax.vlines(x=row['rr_ci_low'], ymin=i-0.15, ymax=i+0.15,
                     color=colors[i], linewidth=2, zorder=2)
            ax.vlines(x=row['rr_ci_high'], ymin=i-0.15, ymax=i+0.15,
                     color=colors[i], linewidth=2, zorder=2)
    
    ax.axvline(x=1.0, color='#333333', linestyle='--', linewidth=1.5, alpha=0.7)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ds_df['display_name'], fontsize=16)
    ax.set_xlabel('Relative Risk (feature changed vs unchanged)', fontsize=14)
    
    display_name = DATASET_DISPLAY_NAMES.get(dataset_name, dataset_name)
    ax.set_title(f'{display_name}: Answer Change RR', fontsize=14, fontweight='bold')
    
    ax.xaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    x_max_data = ds_df['rr_ci_high'].max() if ds_df['rr_ci_high'].notna().any() else ds_df['relative_risk'].max()
    x_min_data = ds_df['rr_ci_low'].min() if ds_df['rr_ci_low'].notna().any() else ds_df['relative_risk'].min()
    ax.set_xlim(min(x_min_data * 0.8, 0.8), x_max_data * 1.05)
    ax.tick_params(axis='x', labelsize=11)
    
    plt.tight_layout()
    
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f'answer_change_{dataset_name}.pdf'
    plt.savefig(output_file, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"Saved: {output_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate answer change rate bar plots')
    parser.add_argument('--input', '-i', default='answer_change_by_feature.csv',
                        help='Input CSV file from answer_change_analysis')
    parser.add_argument('--output-dir', '-o', default='figures',
                        help='Output directory for plots')
    args = parser.parse_args()
    
    print("Generating answer change rate plots...")
    
    df = load_and_prepare_data(csv_path=args.input)
    print(f"Loaded {len(df)} features across {df['dataset'].nunique()} datasets")
    
    output_dir = Path(args.output_dir)
    for dataset in df['dataset'].unique():
        plot_dataset_features(df, dataset, output_dir=output_dir)
    
    print(f"\nDone! Plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
