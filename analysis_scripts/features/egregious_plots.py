"""
Visualize egregious unfaithfulness by feature for each dataset.
Generates forest plots showing relative risk.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from .utils import (
    DATASET_DISPLAY_NAMES,
    FEATURE_DISPLAY_NAMES,
    wrap_text,
    get_display_name,
)


# ============== CONFIGURATION ==============
METRIC = 'relative_risk'  # Options: 'relative_risk', 'effect_unchanged', 'effect_baseline', 'lift'

METRIC_CONFIG = {
    'relative_risk': {
        'label': 'Relative Risk',
        'xlabel': 'Relative Risk (RR)',
        'ref_line': 1.0,
        'format': lambda x: f'{x:.2f}x',
    },
    'effect_unchanged': {
        'label': 'Effect vs Unchanged',
        'xlabel': 'Effect (percentage points)',
        'ref_line': 0.0,
        'format': lambda x: f'{x*100:+.1f}%',
    },
}

MIN_SAMPLE_SIZE = 200
OUTPUT_DIR = Path('figures')

def load_and_compute_metrics(csv_path='egregious_by_feature.csv'):
    """Load tabular feature data with pre-computed bootstrap CIs from CSV."""
    df = pd.read_csv(csv_path)
    
    # Filter to tabular datasets (those with n_changed column)
    tabular_df = df[df['n_changed'].notna()].copy()
    tabular_df = tabular_df[tabular_df['n_changed'] >= MIN_SAMPLE_SIZE]
    
    # Derived metrics (not CIs - those come from CSV)
    tabular_df['effect_baseline'] = tabular_df['egregious_when_changed'] - tabular_df['overall_egregious_rate']
    tabular_df['effect_unchanged'] = tabular_df['egregious_when_changed'] - tabular_df['egregious_when_unchanged']
    tabular_df['lift'] = tabular_df['egregious_when_changed'] / tabular_df['overall_egregious_rate']
    
    # CIs come from CSV (pre-computed by analysis script using bootstrap)
    # relative_risk, rr_ci_low, rr_ci_high should already be in the CSV
    
    return tabular_df


def load_mm_metrics(csv_path='egregious_by_feature.csv'):
    """Load Moral Machines metrics with pre-computed bootstrap CIs from CSV."""
    df = pd.read_csv(csv_path)
    
    # MM rows have 'relative_risk' and CIs from bootstrap
    mm_df = df[(df['dataset'] == 'moral_machines') & df['relative_risk'].notna()].copy()
    
    # CIs come from CSV (pre-computed by analysis script using bootstrap)
    # rr_ci_low, rr_ci_high should already be in the CSV
    
    return mm_df



def plot_dataset_features(df, dataset_name, metric=METRIC, output_dir=OUTPUT_DIR):
    """Create forest plot for one dataset."""
    ds_df = df[df['dataset'] == dataset_name].copy()
    
    if len(ds_df) == 0:
        return
    
    ds_df = ds_df.sort_values(metric, ascending=True)
    ds_df['display_name'] = ds_df['feature'].apply(
        lambda x: wrap_text(get_display_name(x))
    )
    
    config = METRIC_CONFIG[metric]
    n_features = len(ds_df)
    fig, ax = plt.subplots(figsize=(8, 8))
    
    y_pos = np.arange(n_features)
    color_red = '#e74c3c'
    color_green = '#27ae60'  # Lighter green, less prominent than red
    colors = [color_red if v > config['ref_line'] else color_green
              for v in ds_df[metric]]
    alphas = [1.0 for _ in ds_df[metric]]

    for i, (x_val, y_val) in enumerate(zip(ds_df[metric], y_pos)):
        ax.scatter(x_val, y_val, color=colors[i], s=100, zorder=3,
                   edgecolors='#333333', linewidth=0.5, alpha=alphas[i])
    
    if metric == 'relative_risk':
        for i, (_, row) in enumerate(ds_df.iterrows()):
            ax.hlines(y=i, xmin=row['rr_ci_low'], xmax=row['rr_ci_high'],
                     color=colors[i], linewidth=2.5, zorder=2, alpha=alphas[i])
            ax.vlines(x=row['rr_ci_low'], ymin=i-0.15, ymax=i+0.15,
                     color=colors[i], linewidth=2, zorder=2, alpha=alphas[i])
            ax.vlines(x=row['rr_ci_high'], ymin=i-0.15, ymax=i+0.15,
                     color=colors[i], linewidth=2, zorder=2, alpha=alphas[i])
    
    ax.axvline(x=config['ref_line'], color='#333333', linestyle='--', 
               linewidth=1.5, alpha=0.7)
    
    fontsize = 16  # Unified font size for all text

    ax.set_yticks(y_pos)
    ax.set_yticklabels(ds_df['display_name'], fontsize=fontsize, ha='right')
    ax.set_xlabel(config['xlabel'], fontsize=fontsize)

    display_name = DATASET_DISPLAY_NAMES.get(dataset_name, dataset_name)
    ax.set_title(f'{display_name} dataset',
                 fontsize=fontsize, fontstyle='italic')

    ax.xaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    # Set x-axis limits: fixed range for income and breast_cancer, auto for others
    if dataset_name in ['income', 'breast_cancer']:
        ax.set_xlim(0.35, 1.65)
    else:
        x_max_data = ds_df['rr_ci_high'].max() if metric == 'relative_risk' else ds_df[metric].max()
        x_min_data = ds_df['rr_ci_low'].min() if metric == 'relative_risk' else ds_df[metric].min()
        ax.set_xlim(min(x_min_data * 0.8, config['ref_line'] * 0.8), x_max_data * 1.05)
    ax.tick_params(axis='x', labelsize=fontsize)

    # Add legend only for income dataset
    if dataset_name == 'income':
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color_red,
                   markersize=10, markeredgecolor='#333333', markeredgewidth=0.5,
                   label='Associated with unfaithfulness'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color_green,
                   markersize=10, markeredgecolor='#333333', markeredgewidth=0.5,
                   label='Associated with faithfulness'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=fontsize)
    
    plt.tight_layout()
    
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f'egregious_{dataset_name}_{metric}.pdf'
    plt.savefig(output_file, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"Saved: {output_file}")


def plot_mm_dimensions(mm_df, output_dir=OUTPUT_DIR):
    """Create forest plot for Moral Machines dimensions (RR vs overall)."""
    if len(mm_df) == 0:
        return
    
    mm_df = mm_df.sort_values('relative_risk', ascending=True)
    mm_df['display_name'] = mm_df['feature'].apply(
        lambda x: wrap_text(get_display_name(x))
    )
    
    n_dimensions = len(mm_df)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    y_pos = np.arange(n_dimensions)
    colors = ['#e74c3c' if v > 1.0 else '#2ecc71' for v in mm_df['relative_risk']]
    
    # Plot points
    ax.scatter(mm_df['relative_risk'], y_pos, color=colors, s=100, zorder=3,
               edgecolors='#333333', linewidth=0.5)
    
    # Plot bootstrap CI whiskers from rr_ci_low/rr_ci_high
    for i, (_, row) in enumerate(mm_df.iterrows()):
        ax.hlines(y=i, xmin=row['rr_ci_low'], xmax=row['rr_ci_high'],
                 color=colors[i], linewidth=2.5, zorder=2)
        ax.vlines(x=row['rr_ci_low'], ymin=i-0.15, ymax=i+0.15,
                 color=colors[i], linewidth=2, zorder=2)
        ax.vlines(x=row['rr_ci_high'], ymin=i-0.15, ymax=i+0.15,
                 color=colors[i], linewidth=2, zorder=2)
    
    # Reference line at RR=1
    ax.axvline(x=1.0, color='#333333', linestyle='--', linewidth=1.5, alpha=0.7)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(mm_df['display_name'], fontsize=16)
    ax.set_xlabel('Relative Risk (vs overall)', fontsize=14)
    ax.set_title('Moral Machines: Egregious Error Rate by Dimension', 
                 fontsize=14, fontweight='bold')
    
    ax.xaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    x_max_data = mm_df['rr_ci_high'].max()
    x_min_data = mm_df['rr_ci_low'].min()
    ax.set_xlim(min(x_min_data * 0.8, 0.8), x_max_data * 1.05)
    ax.tick_params(axis='x', labelsize=11)
    
    plt.tight_layout()
    
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / 'egregious_moral_machines_relative_risk.pdf'
    plt.savefig(output_file, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"Saved: {output_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate egregious feature forest plots')
    parser.add_argument('--input', '-i', default='egregious_by_feature.csv',
                        help='Input CSV file from egregious_analysis')
    parser.add_argument('--output-dir', '-o', default='figures',
                        help='Output directory for plots')
    parser.add_argument('--min-sample-size', type=int, default=200,
                        help='Minimum sample size to include feature')
    args = parser.parse_args()
    
    print(f"Generating egregious feature plots using metric: {METRIC}")
    
    # Load and plot tabular datasets
    df = load_and_compute_metrics(csv_path=args.input)
    print(f"Loaded {len(df)} tabular features across {df['dataset'].nunique()} datasets")
    
    output_dir = Path(args.output_dir)
    for dataset in df['dataset'].unique():
        plot_dataset_features(df, dataset, output_dir=output_dir)
    
    # Load and plot Moral Machines
    mm_df = load_mm_metrics(csv_path=args.input)
    if len(mm_df) > 0:
        print(f"Loaded {len(mm_df)} Moral Machines dimensions")
        plot_mm_dimensions(mm_df, output_dir=output_dir)
    
    print(f"\nDone! Plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
