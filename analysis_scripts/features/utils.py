"""
Shared utilities for feature-level analysis.

Contains common code for loading datasets, extracting feature changes,
and display name mappings used across egregious and answer change analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from scipy import stats

# Import the dataset templates
from src.templates.heart_disease import HeartDisease
from src.templates.pima_diabetes import PimaDiabetes
from src.templates.breast_cancer_recurrence import BreastCancerRecurrence
from src.templates.income import IncomeDataset
from src.templates.attrition import AttritionDataset
from src.templates.bank_marketing import BankMarketing


# ============== DATASET MAPPINGS ==============

DATASET_CLASSES = {
    'heart_disease': HeartDisease,
    'pima_diabetes': PimaDiabetes,
    'breast_cancer_recurrence': BreastCancerRecurrence,
    'income': IncomeDataset,
    'attrition': AttritionDataset,
    'bank_marketing': BankMarketing,
}

DATASET_DISPLAY_NAMES = {
    'heart_disease': 'Heart Disease',
    'pima_diabetes': 'Pima Diabetes',
    'breast_cancer_recurrence': 'Breast Cancer Recurrence',
    'income': 'Income',
    'attrition': 'Employee Attrition',
    'bank_marketing': 'Bank Marketing',
    'moral_machines': 'Moral Machines',
}

# Human-readable feature names (sentence case)
FEATURE_DISPLAY_NAMES = {
    # Heart Disease
    'exang': 'Exercise-induced angina',
    'cp': 'Chest pain type',
    'fbs': 'Fasting blood sugar',
    'slope': 'ST slope',
    'restecg': 'Resting ECG',
    'sex': 'Sex',
    'age_group': 'Age group',
    'chol_level': 'Cholesterol level',
    'trestbps_level': 'Blood pressure',
    
    # Pima Diabetes
    'pregnancies_cat': 'Pregnancy history',
    'glucose_level': 'Glucose level',
    'bp_level': 'Blood pressure',
    'bmi_cat': 'BMI category',
    'insulin_level': 'Insulin level',
    'pedigree_risk': 'Diabetes pedigree risk',
    
    # Breast Cancer
    'irradiat': 'Radiation therapy',
    'deg_malig': 'Degree of malignancy',
    'tumor_size': 'Tumor size',
    'breast_quad': 'Breast quadrant',
    'breast': 'Affected breast',
    'menopause': 'Menopause status',
    'age': 'Age',
    'inv_nodes': 'Involved nodes',
    'node_caps': 'Node capsule penetration',
    
    # Income
    'education': 'Education level',
    'occupation': 'Occupation',
    'hours-per-week': 'Hours per week',
    'marital-status': 'Marital status',
    'race': 'Race',
    'relationship': 'Relationship status',
    'workclass': 'Work class',
    'capital-gain': 'Capital gain',
    'capital-loss': 'Capital loss',
    
    # Attrition
    'YearsAtCompany': 'Years at company',
    'MonthlyIncome': 'Monthly income',
    'OverTime': 'Overtime',
    'DistanceFromHome': 'Distance from home',
    'JobLevel': 'Job level',
    'Gender': 'Gender',
    'BusinessTravel': 'Business travel',
    'Age': 'Age',
    'Department': 'Department',
    'MaritalStatus': 'Marital status',
    'Education': 'Education',
    
    # Bank Marketing
    'Duration of the last contact': 'Call duration',
    'Has an existing personal loan': 'Has personal loan',
    'Number of contacts performed during this campaign': 'Contacts this campaign',
    'Has an existing housing loan': 'Has housing loan',
    'Education level': 'Education level',
    'Marital status': 'Marital status',
    'Job type': 'Job type',
    'Age group': 'Age group',
    
    # Moral Machines dimensions
    'age': 'Age',
    'gender': 'Gender',
    'fitness': 'Fitness',
    'species': 'Species',
    'social_value': 'Social value',
    'utilitarianism': 'Utilitarianism',
    'random': 'Random',
}


# ============== DATA LOADING ==============

def load_raw_datasets(verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """Load all raw tabular datasets."""
    raw_data = {}
    for name, cls in DATASET_CLASSES.items():
        if verbose:
            print(f"Loading {name}...")
        try:
            df = cls.load_dataset()
            raw_data[name] = df
        except Exception as e:
            print(f"  Failed to load {name}: {e}")
    return raw_data


def load_moral_machines_raw(
    csv_path: str = "data/raw/moral_machines_raw.csv",
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load the raw Moral Machines CSV containing scenario metadata.
    
    This CSV has one row per generated scenario, with columns including:
    - scenario_dimension: age, gender, fitness, species, social_value, utilitarianism, random
    - is_in_car, is_interventionism, is_law: binary flags
    - count_dict_1, count_dict_2: character counts per case
    - prompt: the generated scenario text
    
    Returns:
        DataFrame indexed by scenario ID (row number in original CSV)
    """
    df = pd.read_csv(csv_path, index_col=0)
    if verbose:
        print(f"Loaded Moral Machines raw data: {len(df)} scenarios")
        print(f"  Dimensions: {df['scenario_dimension'].value_counts().to_dict()}")
    return df


def extract_moral_machines_dimensions(
    predictions_df: pd.DataFrame,
    mm_raw_df: pd.DataFrame,
    additional_columns: List[str] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Extract scenario dimension for each Moral Machines prediction.
    
    Filters to pairs where original and counterfactual have the same dimension.
    
    Args:
        predictions_df: DataFrame with MM predictions (must have 'original_question_idx' 
                       and 'counterfactual_question_idx')
        mm_raw_df: Raw Moral Machines DataFrame from load_moral_machines_raw()
        additional_columns: Columns from predictions_df to carry through (e.g. ['egregious'])
        verbose: Print alignment checks
    
    Returns:
        DataFrame with columns: dataset, dimension, plus any additional_columns
    """
    # Filter to moral_machines only
    mm_preds = predictions_df[predictions_df['original_dataset'] == 'moral_machines'].copy()
    
    if len(mm_preds) == 0:
        if verbose:
            print("No Moral Machines predictions found.")
        return pd.DataFrame()
    
    if verbose:
        print(f"Processing {len(mm_preds)} Moral Machines predictions...")
    
    results = []
    n_same_dim = 0
    n_diff_dim = 0
    n_missing = 0
    
    for _, row in mm_preds.iterrows():
        orig_idx = row['original_question_idx']
        cf_idx = row['counterfactual_question_idx']
        
        # Validate indices exist in raw data
        if orig_idx not in mm_raw_df.index or cf_idx not in mm_raw_df.index:
            n_missing += 1
            continue
        
        orig_dim = mm_raw_df.loc[orig_idx, 'scenario_dimension']
        cf_dim = mm_raw_df.loc[cf_idx, 'scenario_dimension']
        
        # Only include pairs with matching dimensions
        if orig_dim != cf_dim:
            n_diff_dim += 1
            continue
        
        n_same_dim += 1
        
        result = {
            'dataset': 'moral_machines',
            'dimension': orig_dim,
        }
        
        # Add any additional columns
        if additional_columns:
            for col in additional_columns:
                if col in row:
                    result[col] = row[col]
        
        results.append(result)
    
    if verbose:
        total = n_same_dim + n_diff_dim + n_missing
        print(f"  Alignment checks:")
        print(f"    Same dimension (included): {n_same_dim} ({100*n_same_dim/total:.1f}%)")
        print(f"    Different dimension (excluded): {n_diff_dim} ({100*n_diff_dim/total:.1f}%)")
        if n_missing > 0:
            print(f"    Missing indices (excluded): {n_missing}")
    
    return pd.DataFrame(results)


def load_prediction_files(
    files: List[str] = None,
    filter_tabular: bool = True
) -> pd.DataFrame:
    """
    Load and optionally filter prediction parquet files.
    
    Args:
        files: List of parquet file paths. If None, uses default list.
        filter_tabular: If True, filter to only tabular datasets.
    """
    if files is None:
        files = [
            'gpt_5_predictions.parquet',
            'claude_4_5_predictions.parquet', 
            'gemini_3_predictions.parquet',
        ]
    
    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            dfs.append(df)
            print(f"  {f}: {len(df)} rows")
        except Exception as e:
            print(f"  {f}: Failed - {e}")
    
    predictions_df = pd.concat(dfs) if dfs else pd.DataFrame()
    
    if filter_tabular:
        tabular_datasets = set(DATASET_CLASSES.keys())
        predictions_df = predictions_df[predictions_df['original_dataset'].isin(tabular_datasets)]
        print(f"Tabular datasets only: {len(predictions_df)} predictions")
    
    return predictions_df


# ============== FEATURE EXTRACTION ==============

def extract_feature_changes(
    predictions_df: pd.DataFrame,
    raw_datasets: Dict[str, pd.DataFrame],
    additional_columns: List[str] = None
) -> pd.DataFrame:
    """
    Extract feature change information for each prediction.
    
    Creates a "long-form" DataFrame where each row represents a single
    (prediction, feature) pair. For each prediction, this function looks up
    the original and counterfactual rows in the raw tabular dataset using
    their indices, then compares each feature to determine if it changed.
    
    How it works:
    1. For each prediction row in predictions_df:
       - Gets original_question_idx and counterfactual_question_idx
       - Looks up these row indices in the corresponding raw dataset
    2. For each feature column in that dataset:
       - Compares the original vs counterfactual values
       - Records whether the feature changed (True/False)
    3. Outputs one row per (prediction, feature) combination
    
    Example:
        If a prediction compares heart disease patient 42 (original) to 
        patient 87 (counterfactual), and heart_disease has 9 features,
        this function generates 9 output rows—one for each feature—each
        with a 'changed' boolean indicating whether that feature differs.
    
    Note:
        This relies on load_dataset() being deterministic (same rows in 
        same order every time). The raw datasets are NOT stored in the
        parquet files—they're reconstructed by joining on row indices.
    
    Args:
        predictions_df: DataFrame with prediction data. Must contain columns:
            - 'original_dataset': name of the dataset (e.g., 'heart_disease')
            - 'original_question_idx': row index in raw dataset for original
            - 'counterfactual_question_idx': row index for counterfactual
        raw_datasets: Dict mapping dataset name to the raw DataFrame 
            (from load_dataset()). Must have same row ordering as when
            counterfactuals were generated.
        additional_columns: Optional list of columns from predictions_df to
            carry through to output (e.g., ['egregious', 'answer_changed'])
        
    Returns:
        DataFrame with columns:
            - 'dataset': name of the dataset
            - 'feature': name of the feature
            - 'changed': bool, True if feature differs between orig/cf
            - Plus any columns specified in additional_columns
    """
    results = []
    
    for dataset_name, dataset_df in raw_datasets.items():
        mask = predictions_df['original_dataset'] == dataset_name
        pred_subset = predictions_df[mask]
        
        if len(pred_subset) == 0:
            continue
        
        feature_cols = [c for c in dataset_df.columns if c != 'target']
        
        for _, row in pred_subset.iterrows():
            orig_idx = row['original_question_idx']
            cf_idx = row['counterfactual_question_idx']
            
            if orig_idx >= len(dataset_df) or cf_idx >= len(dataset_df):
                continue
            
            orig_row = dataset_df.iloc[orig_idx]
            cf_row = dataset_df.iloc[cf_idx]
            
            for feat in feature_cols:
                result = {
                    'dataset': dataset_name,
                    'feature': feat,
                    'changed': (orig_row[feat] != cf_row[feat]),
                }
                
                # Add any additional columns from the prediction row
                if additional_columns:
                    for col in additional_columns:
                        if col in row:
                            result[col] = row[col]
                
                results.append(result)
    
    return pd.DataFrame(results)


# ============== STATISTICS ==============

def compute_wilson_ci(p: float, n: int, confidence: float = 0.95):
    """Compute Wilson score interval for a proportion."""
    if n == 0 or pd.isna(p):
        return np.nan, np.nan
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return max(0, center - margin), min(1, center + margin)


def compute_rr_ci(p1: float, n1: int, p2: float, n2: int, confidence: float = 0.95):
    """
    Compute confidence interval for relative risk using log-normal approximation.
    RR = p1 / p2
    """
    if p1 == 0 or p2 == 0 or pd.isna(p1) or pd.isna(p2):
        return np.nan, np.nan
    
    rr = p1 / p2
    se_log_rr = np.sqrt((1 - p1) / (n1 * p1) + (1 - p2) / (n2 * p2))
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    
    log_rr = np.log(rr)
    ci_low = np.exp(log_rr - z * se_log_rr)
    ci_high = np.exp(log_rr + z * se_log_rr)
    
    return ci_low, ci_high


def bootstrap_rr_ci(
    group_a_outcomes: np.ndarray,
    group_b_outcomes: np.ndarray,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42
) -> tuple:
    """
    Compute bootstrap confidence interval for relative risk.
    
    RR = mean(group_a) / mean(group_b)
    
    Uses a LOCAL RandomState to avoid affecting global random state
    (important for deterministic dataset loading elsewhere).
    
    Args:
        group_a_outcomes: Binary outcomes (0/1) for group A (numerator)
        group_b_outcomes: Binary outcomes (0/1) for group B (denominator)
        n_bootstrap: Number of bootstrap resamples
        confidence: Confidence level (e.g., 0.95 for 95% CI)
        seed: Random seed for reproducibility
        
    Returns:
        (rr_point_estimate, ci_low, ci_high)
    """
    # Use local RandomState to not affect global state
    rng = np.random.RandomState(seed)
    
    group_a = np.asarray(group_a_outcomes)
    group_b = np.asarray(group_b_outcomes)
    
    n_a = len(group_a)
    n_b = len(group_b)
    
    if n_a == 0 or n_b == 0:
        return np.nan, np.nan, np.nan
    
    # Point estimate
    p_a = group_a.mean()
    p_b = group_b.mean()
    
    if p_b == 0:
        return np.nan, np.nan, np.nan
    
    rr = p_a / p_b
    
    # Bootstrap
    bootstrap_rrs = []
    for _ in range(n_bootstrap):
        # Resample each group independently with replacement
        sample_a = rng.choice(group_a, size=n_a, replace=True)
        sample_b = rng.choice(group_b, size=n_b, replace=True)
        
        p_a_boot = sample_a.mean()
        p_b_boot = sample_b.mean()
        
        if p_b_boot > 0:
            bootstrap_rrs.append(p_a_boot / p_b_boot)
    
    if len(bootstrap_rrs) == 0:
        return rr, np.nan, np.nan
    
    # Percentile CI
    alpha = 1 - confidence
    ci_low = np.percentile(bootstrap_rrs, 100 * alpha / 2)
    ci_high = np.percentile(bootstrap_rrs, 100 * (1 - alpha / 2))
    
    return rr, ci_low, ci_high


def bootstrap_rate_ci(
    outcomes: np.ndarray,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42
) -> tuple:
    """
    Compute bootstrap confidence interval for a rate/proportion.
    
    Uses a LOCAL RandomState to avoid affecting global random state.
    
    Args:
        outcomes: Binary outcomes (0/1)
        n_bootstrap: Number of bootstrap resamples  
        confidence: Confidence level
        seed: Random seed
        
    Returns:
        (rate_point_estimate, ci_low, ci_high)
    """
    rng = np.random.RandomState(seed)
    outcomes = np.asarray(outcomes)
    n = len(outcomes)
    
    if n == 0:
        return np.nan, np.nan, np.nan
    
    rate = outcomes.mean()
    
    bootstrap_rates = []
    for _ in range(n_bootstrap):
        sample = rng.choice(outcomes, size=n, replace=True)
        bootstrap_rates.append(sample.mean())
    
    alpha = 1 - confidence
    ci_low = np.percentile(bootstrap_rates, 100 * alpha / 2)
    ci_high = np.percentile(bootstrap_rates, 100 * (1 - alpha / 2))
    
    return rate, ci_low, ci_high


# ============== TEXT UTILITIES ==============

def wrap_text(text: str, max_chars: int = 10) -> str:
    """Wrap text to two lines if longer than max_chars."""
    if len(text) <= max_chars:
        return text
    words = text.split()
    if len(words) == 1:
        return text
    mid = len(text) // 2
    best_break = 0
    break_idx = 1
    for i, word in enumerate(words[:-1]):
        pos = len(' '.join(words[:i+1]))
        if abs(pos - mid) < abs(best_break - mid):
            best_break = pos
            break_idx = i + 1
    return ' '.join(words[:break_idx]) + '\n' + ' '.join(words[break_idx:])


def get_display_name(feature: str) -> str:
    """Get human-readable display name for a feature."""
    return FEATURE_DISPLAY_NAMES.get(feature, feature.replace('_', ' ').title())
