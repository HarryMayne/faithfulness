"""
Generate the combined dataset by random sampling without replacement.

Usage:
    python -m data.natural_counterfactuals.generate_combined
"""
import pandas as pd

# settings
datasets = ['attrition', 'breast_cancer_recurrence', 'heart_disease', 'income', 'pima_diabetes', 'bank_marketing', 'moral_machines']
sample = 1000
seed = 42

# load datasets
loaded_datasets = {}
for dataset in datasets:
    loaded_datasets.update({dataset:pd.read_parquet(f"data/natural_counterfactuals/{dataset}_counterfactual_dataset_balanced.parquet")})
print("Loaded datasets")

# concat
combined = pd.concat([loaded_datasets[x].sample(sample).reset_index(drop=True) for x in loaded_datasets.keys()])
print("Sampled and concatenated datasets")

# shuffle and save
combined = combined.sample(len(combined), random_state=seed).reset_index(drop=True) # default is sampling without replacement
combined.to_parquet('data/natural_counterfactuals/combined_dataset.parquet')
print(len(combined))
print("Save combined dataset")