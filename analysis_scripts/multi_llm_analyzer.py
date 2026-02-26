"""
Multi-LLM Experiment Analyzer

Analyzes results from multi-LLM experiments including:
- Model agreement matrices
- Prediction distributions
- Failure rate analysis
- Correlation heatmaps
"""

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.schema import CounterfactualDatabase


class MultiLLMAnalyzer:
    """
    Analyzes multi-model experiment results stored in Parquet files.
    
    Works with parquet files containing responses from multiple models,
    where each model's responses are stored in separate columns.
    """
    
    def __init__(self, experiment_folder: str | Path):
        """
        Initialize analyzer for a specific experiment run.
        
        Args:
            experiment_folder: Path to the experiment run folder containing
                             parquet files and config
        """
        self.experiment_folder = Path(experiment_folder)
        
        if not self.experiment_folder.exists():
            raise ValueError(f"Experiment folder not found: {experiment_folder}")
        
        # Load experiment config
        config_path = self.experiment_folder / "experiment_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = None
            print("⚠️  Warning: No experiment config found")
        
        # Find all parquet files
        self.parquet_files = list(self.experiment_folder.glob("*_multi_model_responses.parquet"))
        
        if not self.parquet_files:
            print("⚠️  Warning: No multi-model parquet files found")
        
        print(f"Found {len(self.parquet_files)} dataset(s) to analyze")
    
    def analyze_all(self):
        """
        Run complete analysis on all datasets.
        Generates summary report, correlation matrices, and visualizations.
        """
        print("="*80)
        print("MULTI-LLM ANALYSIS")
        print("="*80)
        print(f"Experiment: {self.experiment_folder.name}")
        print(f"Datasets: {len(self.parquet_files)}")
        print("="*80)
        
        # Generate summary report
        self._generate_summary_report()
        
        # Analyze correlations for each dataset
        for parquet_file in self.parquet_files:
            dataset_name = self._extract_dataset_name(parquet_file)
            print(f"\n{'='*80}")
            print(f"Analyzing: {dataset_name}")
            print('='*80)
            
            self._analyze_dataset(parquet_file, dataset_name)
        
        print("\n" + "="*80)
        print("✓ ANALYSIS COMPLETE")
        print("="*80)
        print(f"Results saved to: {self.experiment_folder}")
    
    def _generate_summary_report(self):
        """Generate a comprehensive summary report"""
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("MULTI-LLM EXPERIMENT SUMMARY REPORT")
        report_lines.append("="*80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Experiment folder: {self.experiment_folder}")
        report_lines.append("")
        
        if self.config:
            report_lines.append("Experiment Configuration:")
            report_lines.append(f"  Datasets: {', '.join(self.config['datasets'])}")
            report_lines.append(f"  Models: {len(self.config['llm_configs'])}")
            for i, llm_config in enumerate(self.config['llm_configs'], 1):
                report_lines.append(f"    {i}. {llm_config['model_name']}")
            report_lines.append("")
        
        # Analyze each dataset
        for parquet_file in self.parquet_files:
            dataset_name = self._extract_dataset_name(parquet_file)
            report_lines.append("-"*80)
            report_lines.append(f"DATASET: {dataset_name}")
            report_lines.append("-"*80)
            
            try:
                df = pd.read_parquet(parquet_file)
                
                # Check format: relational (new) vs wide (old)
                if 'counterfactual_reference_response_model_info_model' in df.columns:
                    # Relational format: one row per counterfactual per model
                    report_lines.append("\nFormat: Relational (one record per model per counterfactual)")
                    
                    # Count unique counterfactuals
                    df['cf_id'] = df['original_question_idx'].astype(str) + '_' + df['counterfactual_question_idx'].astype(str)
                    n_counterfactuals = df['cf_id'].nunique()
                    
                    report_lines.append(f"Total counterfactuals: {n_counterfactuals}")
                    
                    # Get ground truth distribution (one per unique counterfactual)
                    gt_df = df.drop_duplicates(subset='cf_id')
                    report_lines.append(f"\nGround Truth Balance:")
                    gt_counts = gt_df['counterfactual_ground_truth'].value_counts()
                    for label, count in gt_counts.items():
                        pct = (count / n_counterfactuals * 100) if n_counterfactuals > 0 else 0
                        report_lines.append(f"  {label}: {count}/{n_counterfactuals} ({pct:.1f}%)")
                    
                    # Analyze each model
                    models = df['counterfactual_reference_response_model_info_model'].unique()
                    report_lines.append(f"\nModel Performance ({len(models)} models):")
                    report_lines.append("")
                    
                    for model in sorted(models):
                        model_df = df[df['counterfactual_reference_response_model_info_model'] == model]
                        short_name = model.split('/')[-1] if '/' in model else model
                        
                        # Check if thinking model
                        thinking_status = model_df['counterfactual_reference_response_model_info_thinking'].iloc[0] if 'counterfactual_reference_response_model_info_thinking' in model_df.columns else None
                        thinking_str = " [THINKING]" if thinking_status else ""
                        
                        # Count predictions
                        answers = model_df['counterfactual_reference_response_answer']
                        total_count = len(answers)
                        null_count = answers.isna().sum()
                        success_count = total_count - null_count
                        
                        report_lines.append(f"  {short_name}{thinking_str}:")
                        report_lines.append(f"    Total: {total_count}")
                        report_lines.append(f"    Successful: {success_count}")
                        report_lines.append(f"    Failed: {null_count} ({null_count/total_count*100:.1f}%)")
                        
                        # Prediction distribution
                        if success_count > 0:
                            pred_dist = answers.value_counts()
                            report_lines.append(f"    Distribution:")
                            for pred, count in pred_dist.items():
                                pct = (count / success_count * 100)
                                report_lines.append(f"      {pred}: {count} ({pct:.1f}%)")
                        
                        report_lines.append("")
                
                else:
                    # Old wide format
                    report_lines.append("\nFormat: Wide (one column per model)")
                    
                    # Get ground truth distribution
                    report_lines.append(f"\nGround Truth Balance:")
                    gt_counts = df['counterfactual_ground_truth'].value_counts()
                    total = len(df)
                    for label, count in gt_counts.items():
                        pct = (count / total * 100) if total > 0 else 0
                        report_lines.append(f"  {label}: {count}/{total} ({pct:.1f}%)")
                    
                    # Find all model response columns
                    model_columns = self._find_model_columns(df)
                    
                    if model_columns:
                        report_lines.append(f"\nModel Performance ({len(model_columns)} models):")
                        report_lines.append("")
                        
                        for model_col in sorted(model_columns):
                            model_name = model_col.replace('counterfactual_', '').replace('_response_answer', '')
                            
                            # Count predictions
                            answers = df[model_col]
                            total_count = len(answers)
                            null_count = answers.isna().sum()
                            success_count = total_count - null_count
                            
                            report_lines.append(f"  {model_name}:")
                            report_lines.append(f"    Total: {total_count}")
                            report_lines.append(f"    Successful: {success_count}")
                            report_lines.append(f"    Failed: {null_count} ({null_count/total_count*100:.1f}%)")
                            
                            # Prediction distribution
                            if success_count > 0:
                                pred_dist = answers.value_counts()
                                report_lines.append(f"    Distribution:")
                                for pred, count in pred_dist.items():
                                    pct = (count / success_count * 100)
                                    report_lines.append(f"      {pred}: {count} ({pct:.1f}%)")
                            
                            report_lines.append("")
                
                report_lines.append("")
                
            except Exception as e:
                report_lines.append(f"  Error analyzing {dataset_name}: {e}")
                report_lines.append("")
        
        # Write report
        report_path = self.experiment_folder / "summary_report.txt"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        # Also print to console
        print('\n'.join(report_lines))
        print(f"\nSaved summary report to: {report_path}")
    
    def _analyze_dataset(self, parquet_file: Path, dataset_name: str):
        """Analyze a single dataset's multi-model results"""
        try:
            df = pd.read_parquet(parquet_file)
            
            # The data is in relational format: one row per counterfactual per model
            # We need to pivot it so each model becomes a column
            
            # Check if this is relational format (new schema)
            if 'counterfactual_reference_response_model_info_model' in df.columns:
                # Relational format - pivot the data
                print("  Detected relational format (one record per model per counterfactual)")
                
                # Filter out Qwen3-32B if no thinking config was tracked
                # (This was run without the thinking field properly set)
                df = df[~df['counterfactual_reference_response_model_info_model'].str.contains('Qwen/Qwen3-32B', na=False)]
                print(f"  Filtered out Qwen3-32B (no thinking config tracked)")
                
                # Create a unique identifier for each counterfactual
                # Use combination of original_question_idx and counterfactual_question_idx
                df['cf_id'] = df['original_question_idx'].astype(str) + '_' + df['counterfactual_question_idx'].astype(str)
                
                # Check if we have answer_first field to separate cases
                if 'original_answer_first' in df.columns:
                    # Analyze separately for answer_first and answer_last
                    for answer_placement in [True, False]:
                        placement_name = "answer_first" if answer_placement else "answer_last"
                        df_filtered = df[df['original_answer_first'] == answer_placement]
                        
                        if len(df_filtered) == 0:
                            print(f"\n  No data for {placement_name}, skipping...")
                            continue
                        
                        print(f"\n  Analyzing {placement_name} cases...")
                        
                        # Pivot: rows = counterfactuals, columns = models, values = answers
                        pivot_df = df_filtered.pivot_table(
                            index='cf_id',
                            columns='counterfactual_reference_response_model_info_model',
                            values='counterfactual_reference_response_answer',
                            aggfunc='first'  # Take first if duplicates
                        )
                        
                        print(f"    Pivoted to {len(pivot_df)} counterfactuals x {len(pivot_df.columns)} models")
                        
                        # Extract predictions for each model
                        # Also get thinking status for display
                        model_predictions = {}
                        for model_name in pivot_df.columns:
                            # Get thinking status for this model
                            model_rows = df_filtered[df_filtered['counterfactual_reference_response_model_info_model'] == model_name]
                            thinking_status = None
                            if 'counterfactual_reference_response_model_info_thinking' in model_rows.columns:
                                thinking_status = model_rows['counterfactual_reference_response_model_info_thinking'].iloc[0]
                            
                            # Simplify model name for display
                            short_name = model_name.split('/')[-1] if '/' in model_name else model_name
                            if thinking_status:
                                short_name += "_thinking"
                            
                            model_predictions[short_name] = pivot_df[model_name].tolist()
                        
                        if len(model_predictions) < 2:
                            print(f"    Not enough models to compute correlations (found {len(model_predictions)})")
                            continue
                        
                        # Compute correlation matrix
                        correlation_matrix = self._compute_correlation_matrix(model_predictions)
                        
                        # Print correlation matrix
                        print(f"\n  Prediction Agreement Matrix ({placement_name}):")
                        print(correlation_matrix.to_string(float_format=lambda x: f"{x:.3f}"))
                        
                        # Save correlation matrix with placement suffix
                        dataset_name_with_placement = f"{dataset_name}_{placement_name}"
                        self._save_correlation_matrix(dataset_name_with_placement, correlation_matrix)
                        
                        # Print statistics
                        self._print_correlation_stats(correlation_matrix)
                else:
                    # No answer_first field - analyze all together
                    # Pivot: rows = counterfactuals, columns = models, values = answers
                    pivot_df = df.pivot_table(
                        index='cf_id',
                        columns='counterfactual_reference_response_model_info_model',
                        values='counterfactual_reference_response_answer',
                        aggfunc='first'  # Take first if duplicates
                    )
                    
                    print(f"  Pivoted to {len(pivot_df)} counterfactuals x {len(pivot_df.columns)} models")
                    
                    # Extract predictions for each model
                    # Also get thinking status for display
                    model_predictions = {}
                    for model_name in pivot_df.columns:
                        # Get thinking status for this model
                        model_rows = df[df['counterfactual_reference_response_model_info_model'] == model_name]
                        thinking_status = None
                        if 'counterfactual_reference_response_model_info_thinking' in model_rows.columns:
                            thinking_status = model_rows['counterfactual_reference_response_model_info_thinking'].iloc[0]
                        
                        # Simplify model name for display
                        short_name = model_name.split('/')[-1] if '/' in model_name else model_name
                        if thinking_status:
                            short_name += "_thinking"
                        
                        model_predictions[short_name] = pivot_df[model_name].tolist()
                    
                    if len(model_predictions) < 2:
                        print(f"  Not enough models to compute correlations (found {len(model_predictions)})")
                        return
                    
                    # Compute correlation matrix
                    correlation_matrix = self._compute_correlation_matrix(model_predictions)
                    
                    # Print correlation matrix
                    print("\nPrediction Agreement Matrix:")
                    print(correlation_matrix.to_string(float_format=lambda x: f"{x:.3f}"))
                    
                    # Save correlation matrix
                    self._save_correlation_matrix(dataset_name, correlation_matrix)
                    
                    # Print statistics
                    self._print_correlation_stats(correlation_matrix)
                
            else:
                # Old wide format - find model columns
                model_columns = self._find_model_columns(df)
                
                if len(model_columns) < 2:
                    print(f"  Not enough models to compute correlations (found {len(model_columns)})")
                    return
                
                # Extract predictions for each model
                model_predictions = {}
                for col in model_columns:
                    model_name = col.replace('counterfactual_', '').replace('_response_answer', '')
                    model_predictions[model_name] = df[col].tolist()
            
                if len(model_predictions) < 2:
                    print(f"  Not enough models to compute correlations (found {len(model_predictions)})")
                    return
                
                # Compute correlation matrix
                correlation_matrix = self._compute_correlation_matrix(model_predictions)
                
                # Print correlation matrix
                print("\nPrediction Agreement Matrix:")
                print(correlation_matrix.to_string(float_format=lambda x: f"{x:.3f}"))
                
                # Save correlation matrix
                self._save_correlation_matrix(dataset_name, correlation_matrix)
                
                # Print statistics
                self._print_correlation_stats(correlation_matrix)
            
        except Exception as e:
            print(f"  Error analyzing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    def _find_model_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Find all columns containing model response answers.
        
        These follow the pattern: counterfactual_<model_name>_response_answer
        but we need to exclude the original reference_response_answer
        """
        model_cols = []
        for col in df.columns:
            if col.startswith('counterfactual_') and col.endswith('_response_answer'):
                # Exclude the original reference response
                if col != 'counterfactual_reference_response_answer':
                    model_cols.append(col)
        
        # Also include the original reference if no others found
        if not model_cols and 'counterfactual_reference_response_answer' in df.columns:
            model_cols.append('counterfactual_reference_response_answer')
        
        return model_cols
    
    def _compute_correlation_matrix(self, model_predictions: Dict[str, List]) -> pd.DataFrame:
        """
        Compute pairwise agreement between models.
        
        Agreement = % of cases where both models made the same prediction
        (excluding cases where either model failed to produce a prediction)
        """
        model_names = sorted(model_predictions.keys())
        n_models = len(model_names)
        
        correlation_matrix = np.zeros((n_models, n_models))
        
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i == j:
                    correlation_matrix[i, j] = 1.0
                else:
                    preds1 = model_predictions[model1]
                    preds2 = model_predictions[model2]
                    
                    # Ensure same length
                    min_len = min(len(preds1), len(preds2))
                    preds1 = preds1[:min_len]
                    preds2 = preds2[:min_len]
                    
                    # Calculate agreement where both predicted
                    agreements = 0
                    valid_comparisons = 0
                    
                    for p1, p2 in zip(preds1, preds2):
                        # Check if both are non-null (handle both None and NaN)
                        if p1 is not None and p2 is not None:
                            if not (pd.isna(p1) or pd.isna(p2)):
                                valid_comparisons += 1
                                if p1 == p2:
                                    agreements += 1
                    
                    if valid_comparisons > 0:
                        correlation_matrix[i, j] = agreements / valid_comparisons
                    else:
                        correlation_matrix[i, j] = np.nan
        
        df = pd.DataFrame(
            correlation_matrix,
            index=model_names,
            columns=model_names
        )
        return df
    
    def _save_correlation_matrix(self, dataset_name: str, correlation_matrix: pd.DataFrame):
        """Save correlation matrix to CSV and generate heatmap"""
        # Save CSV
        csv_file = self.experiment_folder / f"{dataset_name}_correlations.csv"
        correlation_matrix.to_csv(csv_file)
        print(f"\nSaved correlation matrix to: {csv_file}")
        
        # Generate heatmap
        self._plot_correlation_heatmap(dataset_name, correlation_matrix)
    
    def _plot_correlation_heatmap(self, dataset_name: str, correlation_matrix: pd.DataFrame):
        """Generate and save correlation heatmap"""
        plt.figure(figsize=(10, 8))
        
        # Get min/max for color scale (excluding diagonal and NaN)
        values = []
        for i in range(len(correlation_matrix)):
            for j in range(len(correlation_matrix)):
                if i != j:
                    val = correlation_matrix.iloc[i, j]
                    if not np.isnan(val):
                        values.append(val)
        
        if values:
            vmin = max(0.0, min(values) - 0.02)
            vmax = min(1.0, max(values) + 0.02)
        else:
            vmin, vmax = 0.0, 1.0
        
        # Create heatmap
        sns.heatmap(
            correlation_matrix,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            vmin=vmin,
            vmax=vmax,
            square=True,
            cbar_kws={'label': 'Agreement Rate'},
            linewidths=0.5
        )
        
        plt.title(f'Model Agreement Matrix - {dataset_name}', fontsize=14, fontweight='bold')
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Model', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save plot
        plot_file = self.experiment_folder / f"{dataset_name}_correlation_heatmap.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved correlation heatmap to: {plot_file}")
    
    def _print_correlation_stats(self, correlation_matrix: pd.DataFrame):
        """Print statistics about correlations"""
        # Get upper triangle (excluding diagonal)
        n = len(correlation_matrix)
        upper_triangle = []
        for i in range(n):
            for j in range(i+1, n):
                val = correlation_matrix.iloc[i, j]
                if not np.isnan(val):
                    upper_triangle.append(val)
        
        if upper_triangle:
            print("\nCorrelation Statistics:")
            print(f"  Mean agreement: {np.mean(upper_triangle):.3f}")
            print(f"  Std agreement: {np.std(upper_triangle):.3f}")
            print(f"  Min agreement: {np.min(upper_triangle):.3f}")
            print(f"  Max agreement: {np.max(upper_triangle):.3f}")
            
            # Find most/least similar pairs
            model_names = list(correlation_matrix.index)
            pairs = []
            for i in range(n):
                for j in range(i+1, n):
                    val = correlation_matrix.iloc[i, j]
                    if not np.isnan(val):
                        pairs.append({
                            'model1': model_names[i],
                            'model2': model_names[j],
                            'agreement': val
                        })
            
            if pairs:
                pairs.sort(key=lambda x: x['agreement'], reverse=True)
                
                print("\nMost similar models:")
                for item in pairs[:3]:
                    print(f"  {item['model1']} ↔ {item['model2']}: {item['agreement']:.3f}")
                
                print("\nMost different models:")
                for item in pairs[-3:]:
                    print(f"  {item['model1']} ↔ {item['model2']}: {item['agreement']:.3f}")
    
    def _extract_dataset_name(self, parquet_file: Path) -> str:
        """Extract dataset name from parquet filename"""
        name = parquet_file.stem
        # Remove the suffix
        if name.endswith('_multi_model_responses'):
            name = name[:-len('_multi_model_responses')]
        return name
