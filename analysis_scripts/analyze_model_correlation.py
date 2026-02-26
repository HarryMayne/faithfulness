"""
Model Correlation Analyzer for Scaling Laws

Analyzes agreement between different Qwen model sizes on the same questions.
Works with parquet files from scaling laws experiments.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import matthews_corrcoef


def compute_phi_coefficient(y1, y2):
    """
    Compute phi coefficient (Matthews correlation) between two binary predictions.
    
    Phi coefficient ranges from -1 to 1:
    - 1: Perfect positive correlation
    - 0: No correlation
    - -1: Perfect negative correlation
    
    Args:
        y1, y2: Binary prediction arrays (must be same length)
    
    Returns:
        Phi coefficient, or NaN if cannot be computed
    """
    # Filter to only cases where both predicted
    mask = pd.notna(y1) & pd.notna(y2)
    if mask.sum() < 2:
        return np.nan
    
    y1_valid = y1[mask]
    y2_valid = y2[mask]
    
    try:
        return matthews_corrcoef(y1_valid, y2_valid)
    except:
        return np.nan


def compute_marginal_distributions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute answer distribution for each model.
    
    Args:
        df: DataFrame with 'model' and 'answer' columns
    
    Returns:
        DataFrame with model as index and YES percentage
    """
    results = []
    
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        total = len(model_data)
        yes_count = (model_data['answer'] == 'YES').sum()
        yes_pct = (yes_count / total * 100) if total > 0 else 0
        
        # Simplify model name
        model_name = model.split('/')[-1] if '/' in model else model
        
        results.append({
            'model': model_name,
            'yes_pct': yes_pct,
            'total': total
        })
    
    dist_df = pd.DataFrame(results)
    dist_df = dist_df.set_index('model')
    
    # Sort by model size
    def get_model_size(model_name: str) -> float:
        try:
            if 'Qwen3-' in model_name:
                size_str = model_name.split('Qwen3-')[1].split('B')[0]
                return float(size_str)
        except:
            pass
        return 0.0
    
    dist_df['sort_key'] = dist_df.index.map(get_model_size)
    dist_df = dist_df.sort_values('sort_key')
    dist_df = dist_df.drop('sort_key', axis=1)
    
    return dist_df


def prepare_pivot_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot DataFrame for agreement analysis.
    
    Args:
        df: DataFrame with 'original_question_idx', 'model', and 'answer' columns
        
    Returns:
        Pivoted DataFrame (index=questions, columns=models, values=answers)
    """
    # Pivot: rows = questions, columns = models, values = answers
    pivot_df = df.pivot_table(
        index='original_question_idx',
        columns='model',
        values='answer',
        aggfunc='first'
    )
    
    print(f"Pivoted to {len(pivot_df)} questions x {len(pivot_df.columns)} models")
    return pivot_df


def compute_agreement_matrix_from_pivot(pivot_df: pd.DataFrame, use_phi: bool = False) -> pd.DataFrame:
    """
    Compute pairwise agreement or phi correlation from pivoted data.
    
    Args:
        pivot_df: Pivoted DataFrame (questions x models)
        use_phi: If True, compute phi coefficient. If False, agreement.
        
    Returns:
        Agreement/correlation matrix
    """
    models = pivot_df.columns.tolist()
    
    # Sort models by size (extract number from model name)
    def get_model_size(model_name: str) -> float:
        """Extract model size in billions from name like Qwen/Qwen3-14B"""
        try:
            if 'Qwen3-' in model_name:
                size_str = model_name.split('Qwen3-')[1].split('B')[0]
                return float(size_str)
        except:
            pass
        return 0.0
    
    models = sorted(models, key=get_model_size)
    n_models = len(models)
    matrix = np.zeros((n_models, n_models))
    
    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models):
            if i == j:
                matrix[i, j] = 1.0
            else:
                preds1 = pivot_df[model1]
                preds2 = pivot_df[model2]
                
                if use_phi:
                    # Compute phi coefficient (Matthews correlation)
                    matrix[i, j] = compute_phi_coefficient(preds1, preds2)
                else:
                    # Calculate simple agreement where both predicted successfully
                    mask = preds1.notna() & preds2.notna()
                    if mask.sum() > 0:
                        agreement = (preds1[mask] == preds2[mask]).sum() / mask.sum()
                        matrix[i, j] = agreement
                    else:
                        matrix[i, j] = np.nan
    
    # Create DataFrame with simplified model names
    model_names = [m.split('/')[-1] if '/' in m else m for m in models]
    
    return pd.DataFrame(
        matrix,
        index=model_names,
        columns=model_names
    )


def compute_agreement_matrix(df: pd.DataFrame, use_phi: bool = False) -> pd.DataFrame:
    """Wrapper for backward compatibility"""
    pivot_df = prepare_pivot_data(df)
    return compute_agreement_matrix_from_pivot(pivot_df, use_phi)


def compute_bootstrap_ci_matrix(pivot_df: pd.DataFrame, use_phi: bool = False, n_boot: int = 1000, ci: float = 0.95):
    """
    Compute bootstrap confidence interval for each cell in the correlation matrix.
    
    Args:
        pivot_df: Pivoted DataFrame (questions x models)
        use_phi: If True, compute phi coefficient. If False, agreement.
        n_boot: Number of bootstrap iterations
        ci: Confidence level (e.g. 0.95)
        
    Returns:
        Tuple (lower_bound_df, upper_bound_df)
    """
    print(f"Computing {int(ci*100)}% CI per cell with {n_boot} bootstrap samples...")
    
    n_samples = len(pivot_df)
    
    # Get model list sorted (to match main matrix)
    models = pivot_df.columns.tolist()
    
    # Helper to sort models same way as main function
    def get_model_size(model_name: str) -> float:
        try:
            if 'Qwen3-' in model_name:
                size_str = model_name.split('Qwen3-')[1].split('B')[0]
                return float(size_str)
        except:
            pass
        return 0.0
    
    models = sorted(models, key=get_model_size)
    n_models = len(models)
    
    # Store all bootstrap matrices
    boot_matrices = np.zeros((n_boot, n_models, n_models))
    
    # Pre-convert to numpy for speed
    # We need to map string answers to integers for faster comparison if not phi
    # For phi we need binary 0/1
    
    # Actually, let's just resample the dataframe indices
    indices = np.arange(n_samples)
    
    for b in range(n_boot):
        # Resample indices with replacement
        resample_idx = np.random.choice(indices, size=n_samples, replace=True)
        resampled_df = pivot_df.iloc[resample_idx]
        
        for i in range(n_models):
            for j in range(n_models):
                if i == j:
                    boot_matrices[b, i, j] = 1.0
                else:
                    m1 = models[i]
                    m2 = models[j]
                    
                    p1 = resampled_df[m1]
                    p2 = resampled_df[m2]
                    
                    if use_phi:
                        val = compute_phi_coefficient(p1, p2)
                    else:
                        mask = p1.notna() & p2.notna()
                        if mask.sum() > 0:
                            val = (p1[mask] == p2[mask]).sum() / mask.sum()
                        else:
                            val = np.nan
                    
                    boot_matrices[b, i, j] = val
            
    # Compute percentiles
    lower_matrix = np.nanpercentile(boot_matrices, (1 - ci) / 2 * 100, axis=0)
    upper_matrix = np.nanpercentile(boot_matrices, (1 + ci) / 2 * 100, axis=0)
    
    # Create DataFrames
    model_names = [m.split('/')[-1] if '/' in m else m for m in models]
    
    lower_df = pd.DataFrame(lower_matrix, index=model_names, columns=model_names)
    upper_df = pd.DataFrame(upper_matrix, index=model_names, columns=model_names)
    
    return lower_df, upper_df


def plot_agreement_heatmap(agreement_matrix: pd.DataFrame, output_path: Path, title: str, use_phi: bool = False, marginal_distributions: pd.DataFrame = None, ci_matrices: tuple = None):
    """Generate and save agreement heatmap with optional marginal distributions"""
    
    # Create annotation matrix
    annot = agreement_matrix.applymap(lambda x: f"{x:.3f}")
    
    if ci_matrices:
        lower_df, upper_df = ci_matrices
        for i in range(len(agreement_matrix)):
            for j in range(len(agreement_matrix)):
                if i != j:
                    val = agreement_matrix.iloc[i, j]
                    low = lower_df.iloc[i, j]
                    high = upper_df.iloc[i, j]
                    if not np.isnan(val):
                        annot.iloc[i, j] = f"{val:.3f}\n[{low:.3f}, {high:.3f}]"
    
    if marginal_distributions is not None:
        # Create figure with space for marginals on the right only
        fig = plt.figure(figsize=(14, 10)) # Increased size for CIs
        
        # Create grid for heatmap and right marginal (wider marginal space)
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.2], wspace=0.08)
        
        ax_main = fig.add_subplot(gs[0, 0])
        ax_right = fig.add_subplot(gs[0, 1], sharey=ax_main)
        
        # Plot main heatmap
        # Get min/max for color scale (excluding diagonal and NaN)
        values = []
        for i in range(len(agreement_matrix)):
            for j in range(len(agreement_matrix)):
                if i != j:
                    val = agreement_matrix.iloc[i, j]
                    if not np.isnan(val):
                        values.append(val)
        
        if values:
            if use_phi:
                vmin = max(-1.0, min(values) - 0.1)
                vmax = min(1.0, max(values) + 0.1)
                cbar_label = 'Phi Coefficient'
                cmap = 'RdBu_r'
            else:
                vmin = max(0.0, min(values) - 0.05)
                vmax = min(1.0, max(values) + 0.05)
                cbar_label = 'Agreement Rate'
                cmap = 'RdYlGn'
        else:
            if use_phi:
                vmin, vmax = -1.0, 1.0
                cbar_label = 'Phi Coefficient'
                cmap = 'RdBu_r'
            else:
                vmin, vmax = 0.0, 1.0
                cbar_label = 'Agreement Rate'
                cmap = 'RdYlGn'
        
        sns.heatmap(
            agreement_matrix,
            annot=annot,
            fmt='',
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            center=0.0 if use_phi else None,
            square=True,
            cbar_kws={'label': cbar_label},
            linewidths=0.5,
            ax=ax_main,
            annot_kws={"size": 8} # Smaller font for CI
        )
        
        ax_main.set_xlabel('Model', fontsize=12)
        ax_main.set_ylabel('Model', fontsize=12)
        ax_main.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Plot right marginal (YES percentage, horizontal bars)
        model_names = agreement_matrix.index.tolist()
        yes_pcts = [marginal_distributions.loc[m, 'yes_pct'] for m in model_names]
        
        # Position bars to align with heatmap squares
        n_models = len(model_names)
        y_pos = np.arange(n_models) + 0.5
        
        ax_right.barh(y_pos, yes_pcts, color='steelblue', alpha=0.7, height=0.6)
        ax_right.set_xlabel('YES %', fontsize=10)
        ax_right.set_xlim([0, 100])
        ax_right.set_ylim([0, n_models])
        ax_right.tick_params(axis='y', labelleft=False, left=False)
        ax_right.invert_yaxis()
        ax_right.spines['left'].set_visible(False)
        ax_right.spines['right'].set_visible(False)
        ax_right.spines['top'].set_visible(False)
        ax_right.grid(axis='x', alpha=0.3)
        
    else:
        # Original heatmap without marginals
        plt.figure(figsize=(12, 10))
        
        # Get min/max for color scale (excluding diagonal and NaN)
        values = []
        for i in range(len(agreement_matrix)):
            for j in range(len(agreement_matrix)):
                if i != j:
                    val = agreement_matrix.iloc[i, j]
                    if not np.isnan(val):
                        values.append(val)
        
        if values:
            if use_phi:
                vmin = max(-1.0, min(values) - 0.1)
                vmax = min(1.0, max(values) + 0.1)
                cbar_label = 'Phi Coefficient'
                cmap = 'RdBu_r'
            else:
                vmin = max(0.0, min(values) - 0.05)
                vmax = min(1.0, max(values) + 0.05)
                cbar_label = 'Agreement Rate'
                cmap = 'RdYlGn'
        else:
            if use_phi:
                vmin, vmax = -1.0, 1.0
                cbar_label = 'Phi Coefficient'
                cmap = 'RdBu_r'
            else:
                vmin, vmax = 0.0, 1.0
                cbar_label = 'Agreement Rate'
                cmap = 'RdYlGn'
        
        sns.heatmap(
            agreement_matrix,
            annot=annot,
            fmt='',
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            center=0.0 if use_phi else None,
            square=True,
            cbar_kws={'label': cbar_label},
            linewidths=0.5,
            annot_kws={"size": 8}
        )
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Model', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved heatmap to: {output_path}")


def print_statistics(agreement_matrix: pd.DataFrame, use_phi: bool = False, ci_range: tuple = None):
    """Print agreement statistics"""
    # Get upper triangle (excluding diagonal)
    n = len(agreement_matrix)
    upper_triangle = []
    for i in range(n):
        for j in range(i+1, n):
            val = agreement_matrix.iloc[i, j]
            if not np.isnan(val):
                upper_triangle.append(val)
    
    if not upper_triangle:
        print("No valid values to compute statistics")
        return
    
    metric_name = "Phi Coefficient" if use_phi else "Agreement"
    
    print("\n" + "="*80)
    print(f"{metric_name.upper()} STATISTICS")
    print("="*80)
    print(f"Mean {metric_name.lower()}: {np.mean(upper_triangle):.3f}")
    
    if ci_range:
        lower, upper = ci_range
        print(f"95% CI for Mean:     [{lower:.3f}, {upper:.3f}]")
        
    print(f"Std {metric_name.lower()}:  {np.std(upper_triangle):.3f}")
    print(f"Min {metric_name.lower()}:  {np.min(upper_triangle):.3f}")
    print(f"Max {metric_name.lower()}:  {np.max(upper_triangle):.3f}")
    
    # Find most/least similar pairs
    model_names = list(agreement_matrix.index)
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            val = agreement_matrix.iloc[i, j]
            if not np.isnan(val):
                pairs.append({
                    'model1': model_names[i],
                    'model2': model_names[j],
                    'value': val
                })
    
    if pairs:
        pairs.sort(key=lambda x: x['value'], reverse=True)
        
        print(f"\nMost {'correlated' if use_phi else 'similar'} models:")
        for item in pairs[:5]:
            print(f"  {item['model1']:20s} ↔ {item['model2']:20s}: {item['value']:6.3f}")
        
        print(f"\nMost {'anti-correlated' if use_phi else 'different'} models:")
        for item in reversed(pairs[-5:]):
            print(f"  {item['model1']:20s} ↔ {item['model2']:20s}: {item['value']:6.3f}")


def analyze_parquet(parquet_path: Path, output_dir: Path, use_phi: bool = False):
    """
    Analyze model correlations from scaling laws parquet file.
    
    Args:
        parquet_path: Path to the multi-model parquet file
        output_dir: Directory to save outputs
        use_phi: If True, compute phi coefficient. If False, compute agreement.
    """
    metric_name = "Phi Coefficient" if use_phi else "Agreement"
    
    print("="*80)
    print(f"MODEL CORRELATION ANALYSIS ({metric_name.upper()})")
    print("="*80)
    print(f"Input: {parquet_path}")
    print(f"Output: {output_dir}")
    print(f"Metric: {metric_name}")
    print("="*80)
    
    # Load parquet
    df = pd.read_parquet(parquet_path)
    print(f"\nLoaded {len(df)} records")
    
    # Get dataset name
    dataset_name = parquet_path.stem.replace('_multi_model_responses', '')
    
    # Extract model and answer columns
    # Format: original_reference_response_answer and original_reference_response_model_info_model
    if 'original_reference_response_model_info_model' not in df.columns:
        print("ERROR: Expected column 'original_reference_response_model_info_model' not found!")
        print(f"Available columns: {df.columns.tolist()}")
        return
    
    # Prepare data: question_idx, model, answer
    data = df[['original_question_idx', 
               'original_reference_response_model_info_model',
               'original_reference_response_answer']].copy()
    data.columns = ['original_question_idx', 'model', 'answer']
    
    # Remove failed parses (None/NaN answers)
    data = data.dropna(subset=['answer'])
    
    print(f"Valid predictions: {len(data)}")
    print(f"Unique questions: {data['original_question_idx'].nunique()}")
    print(f"Models: {data['model'].nunique()}")
    
    # Check if we have answer_first field
    if 'original_answer_first' in df.columns:
        print("\nAnalyzing separately by answer placement...")
        
        # Analyze answer_first
        answer_first_df = df[df['original_answer_first'] == True][['original_question_idx', 
                                                                     'original_reference_response_model_info_model',
                                                                     'original_reference_response_answer']].copy()
        answer_first_df.columns = ['original_question_idx', 'model', 'answer']
        answer_first_df = answer_first_df.dropna(subset=['answer'])
        
        if len(answer_first_df) > 0:
            print(f"\n{'='*80}")
            print("ANSWER FIRST")
            print('='*80)
            
            # Compute marginal distributions
            marginal_dist = compute_marginal_distributions(answer_first_df)
            print("\nMarginal Distributions (YES %):")
            print(marginal_dist.to_string())
            
            pivot_df = prepare_pivot_data(answer_first_df)
            agreement_matrix = compute_agreement_matrix_from_pivot(pivot_df, use_phi=use_phi)
            ci_matrices = compute_bootstrap_ci_matrix(pivot_df, use_phi=use_phi)
            
            print(f"\n{metric_name} Matrix:")
            print(agreement_matrix.to_string(float_format=lambda x: f"{x:.3f}"))
            
            # Save
            suffix = "phi" if use_phi else "agreement"
            csv_path = output_dir / f"{dataset_name}_answer_first_{suffix}.csv"
            agreement_matrix.to_csv(csv_path)
            print(f"✓ Saved to: {csv_path}")
            
            plot_path = output_dir / f"{dataset_name}_answer_first_{suffix}.png"
            plot_agreement_heatmap(agreement_matrix, plot_path, 
                                  f"{dataset_name} - Answer First {metric_name}", 
                                  use_phi=use_phi,
                                  marginal_distributions=marginal_dist,
                                  ci_matrices=ci_matrices)
            
            print_statistics(agreement_matrix, use_phi=use_phi)
        
        # Analyze answer_last
        answer_last_df = df[df['original_answer_first'] == False][['original_question_idx', 
                                                                     'original_reference_response_model_info_model',
                                                                     'original_reference_response_answer']].copy()
        answer_last_df.columns = ['original_question_idx', 'model', 'answer']
        answer_last_df = answer_last_df.dropna(subset=['answer'])
        
        if len(answer_last_df) > 0:
            print(f"\n{'='*80}")
            print("ANSWER LAST")
            print('='*80)
            
            # Compute marginal distributions
            marginal_dist = compute_marginal_distributions(answer_last_df)
            print("\nMarginal Distributions (YES %):")
            print(marginal_dist.to_string())
            
            pivot_df = prepare_pivot_data(answer_last_df)
            agreement_matrix = compute_agreement_matrix_from_pivot(pivot_df, use_phi=use_phi)
            ci_matrices = compute_bootstrap_ci_matrix(pivot_df, use_phi=use_phi)
            
            print(f"\n{metric_name} Matrix:")
            print(agreement_matrix.to_string(float_format=lambda x: f"{x:.3f}"))
            
            # Save
            suffix = "phi" if use_phi else "agreement"
            csv_path = output_dir / f"{dataset_name}_answer_last_{suffix}.csv"
            agreement_matrix.to_csv(csv_path)
            print(f"✓ Saved to: {csv_path}")
            
            plot_path = output_dir / f"{dataset_name}_answer_last_{suffix}.png"
            plot_agreement_heatmap(agreement_matrix, plot_path, 
                                  f"{dataset_name} - Answer Last {metric_name}",
                                  use_phi=use_phi,
                                  marginal_distributions=marginal_dist,
                                  ci_matrices=ci_matrices)
            
            print_statistics(agreement_matrix, use_phi=use_phi)
    
    # Overall analysis (all together)
    print(f"\n{'='*80}")
    print("OVERALL (ALL SCENARIOS)")
    print('='*80)
    
    # Compute marginal distributions
    marginal_dist = compute_marginal_distributions(data)
    print("\nMarginal Distributions (YES %):")
    print(marginal_dist.to_string())
    
    pivot_df = prepare_pivot_data(data)
    agreement_matrix = compute_agreement_matrix_from_pivot(pivot_df, use_phi=use_phi)
    ci_matrices = compute_bootstrap_ci_matrix(pivot_df, use_phi=use_phi)
    
    print(f"\n{metric_name} Matrix:")
    print(agreement_matrix.to_string(float_format=lambda x: f"{x:.3f}"))
    
    # Save results
    suffix = "phi" if use_phi else "agreement"
    csv_path = output_dir / f"{dataset_name}_{suffix}.csv"
    agreement_matrix.to_csv(csv_path)
    print(f"✓ Saved to: {csv_path}")
    
    # Generate heatmap
    plot_path = output_dir / f"{dataset_name}_{suffix}_heatmap.png"
    plot_agreement_heatmap(agreement_matrix, plot_path, 
                          f"{dataset_name} - Model {metric_name}",
                          use_phi=use_phi,
                          marginal_distributions=marginal_dist,
                          ci_matrices=ci_matrices)
    
    # Print statistics
    print_statistics(agreement_matrix, use_phi=use_phi)
    
    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze model agreement in scaling laws experiments"
    )
    parser.add_argument(
        "parquet_path",
        type=str,
        help="Path to the multi-model parquet file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (defaults to same directory as parquet file)"
    )
    parser.add_argument(
        "--phi",
        action="store_true",
        help="Use phi coefficient (Matthews correlation) instead of simple agreement"
    )
    
    args = parser.parse_args()
    
    parquet_path = Path(args.parquet_path)
    if not parquet_path.exists():
        print(f"ERROR: Path not found: {parquet_path}")
        return
        
    # Handle directory input
    if parquet_path.is_dir():
        print(f"Input is a directory. Searching for parquet file...")
        parquet_files = list(parquet_path.glob("*multi_model_responses.parquet"))
        
        if not parquet_files:
            # Try any parquet file
            parquet_files = list(parquet_path.glob("*.parquet"))
            
        if not parquet_files:
            print(f"ERROR: No parquet files found in {parquet_path}")
            return
            
        if len(parquet_files) > 1:
            print(f"WARNING: Multiple parquet files found. Using: {parquet_files[0].name}")
            
        parquet_path = parquet_files[0]
        print(f"Found: {parquet_path}")
    
    # Default output directory
    output_dir = Path(args.output_dir) if args.output_dir else parquet_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    analyze_parquet(parquet_path, output_dir, use_phi=args.phi)


if __name__ == "__main__":
    main()
