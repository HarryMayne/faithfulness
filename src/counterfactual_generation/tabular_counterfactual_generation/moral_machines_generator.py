"""
Pipeline to generate moral machines counterfactual dataset.

This pipeline:
0. Generates raw scenarios (10,000 by default)
1. Extracts unique features and feature counts per scenario
2. Groups scenarios by their feature sets to identify counterfactual pairs
3. Exports to parquet format matching the schema

You control the difference between the counterfactuals via the features parsed into unique_features() and unique_feature_counts(). This way you can assert that fewer features are included in the overlapping requirements. Currently all features must be present but the counts must be different. This seems fair.

Usage: # note -- run with PYTHONHASHSEED=0 for full reproducibility
    PYTHONHASHSEED=0 python -m src.counterfactual_generation.tabular_counterfactual_generation.moral_machines_generator

TODO:
    1. NOTE: this could do with a good check... works fine...
    2. Save more metadata somewhere. Where is the best place in the schema? New?
"""
import random
import numpy as np
from src.counterfactual_generation.tabular_counterfactual_generation.tabular_utils import run_moral_machines_generation
import pandas as pd
import ast
from collections import Counter
from src.schema import (
    CounterfactualDatabase,
    FaithfulnessRecord,
    OriginalQuestion,
    CounterfactualInfo,
)
from src.templates.moral_machines import MoralMachines

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Define dataset size. Increased to get >1000 pairs with unique original prompts
N_SCENARIOS = 15_000

def unique_features(row):
    """
    Extract the unique characters and binary features for a scenario.

    Args:
        row: DataFrame row containing scenario information

    Returns:
        List of unique feature names (character types + binary flags)
    """
    l1 = list(ast.literal_eval(row['count_dict_1']).keys())
    l2 = list(ast.literal_eval(row['count_dict_2']).keys())
    all_features = l1 + l2

    # Add binary features as named strings
    if row['is_law']:
        all_features.append('law_present')
    if row['is_in_car']:
        all_features.append('is_in_car')
    if row['is_interventionism']:
        all_features.append('is_interventionism')

    return list(set(all_features))


def unique_feature_counts(row):
    """
    Extract the unique feature counts per scenario.

    Binary features are treated as 0/1 counts:
    - law_present: 1 if is_law is True, 0 otherwise
    - is_in_car: 1 if is_in_car is True, 0 otherwise
    - is_interventionism: 1 if is_interventionism is True, 0 otherwise

    Args:
        row: DataFrame row containing scenario information

    Returns:
        Dictionary of feature counts
    """
    counts_1 = ast.literal_eval(row['count_dict_1'])
    counts_2 = ast.literal_eval(row['count_dict_2'])

    # Add binary features
    binary_features = {
        'law_present': int(row['is_law']),
        'is_in_car': int(row['is_in_car']),
        'is_interventionism': int(row['is_interventionism'])
    }

    # Combine all counts
    combined = Counter(counts_1) + Counter(counts_2) + Counter(binary_features)
    return dict(combined)


def create_counterfactual_database(df, seed=42):
    """
    Create a CounterfactualDatabase from grouped scenarios.

    For each feature group with 2+ members:
    - Randomly sample 2 unique scenarios with different feature counts
    - Use one as the "original" (reference) and one as the counterfactual

    Args:
        df: DataFrame with scenarios, features, and feature_counts
        seed: Random seed for reproducibility

    Returns:
        CounterfactualDatabase object, pair count, and dimension stats dict
    """
    # Set random seed for deterministic sampling
    random.seed(seed)

    db = CounterfactualDatabase()
    grouped = df.groupby('feature_set')

    pair_count = 0

    # Track scenario dimensions for both originals and counterfactuals
    dimension_counts = Counter()

    for feature_set, group in grouped:
        if len(group) < 2:
            continue

        # Create hashable version of feature_counts for comparison
        group = group.copy()
        group['counts_hash'] = group['feature_counts'].apply(lambda x: frozenset(x.items()))

        # Get rows with distinct feature counts
        unique_counts = group.drop_duplicates(subset='counts_hash')

        if len(unique_counts) < 2:
            continue

        # Sample 2 unique scenarios randomly (deterministic due to seed set above)
        sampled = unique_counts.sample(n=2, random_state=seed + pair_count)
        reference_row = sampled.iloc[0]
        cf_row = sampled.iloc[1]

        # Track scenario dimensions
        dimension_counts[reference_row['scenario_dimension']] += 1
        dimension_counts[cf_row['scenario_dimension']] += 1

        # Randomly choose answer position (50% answer first, 50% answer last)
        answer_first = random.choice([True, False])

        # Format the reference prompt using the template
        reference_scenario = reference_row['prompt']
        reference_formatted_prompt = MoralMachines.create_reference_prompt(
            question=reference_scenario,
            answer_last=(not answer_first)  # answer_last is opposite of answer_first
        )

        # Create OriginalQuestion for reference
        original_question = OriginalQuestion(
            dataset="moral_machines",
            question=reference_scenario,  # Raw scenario text
            question_prompt=reference_formatted_prompt,  # Formatted prompt with instructions
            question_idx=int(reference_row.name),  # Use DataFrame index
            ground_truth=None,  # No ground truth for moral machines
            answer_first=answer_first,  # Randomly assigned
            description=reference_scenario  # Same as question for moral machines
        )

        # Compute feature difference as "hamming distance"
        ref_features = set(reference_row['features'])
        cf_features = set(cf_row['features'])

        # Format the counterfactual prompt using the template
        cf_scenario = cf_row['prompt']
        cf_formatted_prompt = MoralMachines.create_reference_prompt(
            question=cf_scenario,
            answer_last=(not answer_first)  # Use same answer position as reference
        )

        # Create counterfactual
        counterfactual = CounterfactualInfo(
            generator_model="moral_machines_feature_matching",
            generator_method="moral_machines_feature_matching",
            question=cf_scenario,  # Raw scenario text
            question_prompt=cf_formatted_prompt,  # Formatted prompt with instructions
            question_idx=int(cf_row.name),  # Use DataFrame index
            ground_truth=None,
            description=cf_scenario,  # Same as question for moral machines
            hamming_distance=None
        )

        record = FaithfulnessRecord(
            original_question=original_question,
            counterfactual=counterfactual
        )

        db.add_record(record)
        pair_count += 1

    return db, pair_count, dimension_counts


def main(seed=SEED):
    """
    Main pipeline for generating moral machines counterfactual dataset.

    Args:
        seed: Random seed for reproducibility

    Note:
        For full determinism, set PYTHONHASHSEED=0 in your environment:
        PYTHONHASHSEED=0 python -m src.counterfactual_generation.tabular_counterfactual_generation.moral_machines_generator
    """
    # Re-set seeds to ensure deterministic behavior
    random.seed(seed)
    np.random.seed(seed)

    # Warn if PYTHONHASHSEED is not set
    import os
    import sys
    hashseed = os.environ.get('PYTHONHASHSEED')
    if hashseed != '0':
        print("WARNING: PYTHONHASHSEED is not set to 0. Results may not be fully deterministic.")
        print("For full reproducibility, run with: PYTHONHASHSEED=0 python -m ...")
        print()

    print("="*60)
    print("Moral Machines Counterfactual Dataset Generator")
    print(f"Random seed: {seed}")
    print("="*60)

    # Step 0: Generate raw scenarios
    print(f"Generating {N_SCENARIOS} scenarios...")
    run_moral_machines_generation(n_scenarios=N_SCENARIOS, seed=seed)

    # Step 1: Load scenarios
    print("Loading scenarios...")
    df = pd.read_csv("data/raw/moral_machines_raw.csv", index_col=0)
    print(f"  Loaded {len(df)} scenarios")

    # Step 2: Extract features and counts
    print("Extracting features and counts...")
    df['features'] = df.apply(unique_features, axis=1)
    df['feature_counts'] = df.apply(unique_feature_counts, axis=1)
    df['feature_set'] = df['features'].apply(frozenset)

    # Step 3: Analyze feature groups
    print("Analyzing feature groups...")
    unique_groups = df['feature_set'].nunique()
    print(f"  Unique feature groups: {unique_groups}")

    group_sizes = df.groupby('feature_set').size()
    groups_with_pairs = (group_sizes >= 2).sum()
    print(f"  Groups with >= 2 members: {groups_with_pairs}")

    # Step 4: Create counterfactual database
    print("Creating counterfactual pairs...")
    db, pair_count, dimension_counts = create_counterfactual_database(df, seed=seed)
    print(f"  Created {pair_count} counterfactual pairs")

    # Step 5: Save to parquet
    output_path = "data/natural_counterfactuals/moral_machines_counterfactual_dataset_balanced.parquet"
    print(f"Saving to {output_path}...")
    db.save_parquet(output_path)

    # Calculate answer_first distribution
    answer_first_count = sum(1 for r in db.records if r.original_question.answer_first)
    answer_last_count = len(db.records) - answer_first_count

    print("\n" + "="*60)
    print(f"Output: {output_path}")
    print(f"Total records: {len(db.records)}")
    print(f"\nPrompt format distribution:")
    print(f"  Answer first: {answer_first_count} ({answer_first_count/len(db.records)*100:.1f}%)")
    print(f"  Answer last: {answer_last_count} ({answer_last_count/len(db.records)*100:.1f}%)")

    print(f"\nScenario dimension distribution (originals + counterfactuals):")
    total_scenarios = sum(dimension_counts.values())
    for dimension in sorted(dimension_counts.keys()):
        count = dimension_counts[dimension]
        pct = (count / total_scenarios) * 100
        print(f"  {dimension}: {count} ({pct:.1f}%)")

    print("="*60)

    return df, db

#######################################################
if __name__ == "__main__":
    df, db = main()