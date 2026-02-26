import pandas as pd
import os

# make a sample
df = pd.read_parquet("tests/test_outputs/heart_disease_counterfactual_dataset_balanced.parquet")
df = df.sample(20)
df_bank = pd.read_parquet("tests/test_outputs/bank_marketing_counterfactual_dataset_balanced.parquet")
df_bank = df_bank.sample(20)
os.makedirs("tests/test_outputs", exist_ok=True)
df.to_parquet("tests/test_outputs/sample_data.parquet")
df_bank.to_parquet("tests/test_outputs/sample_data_bank.parquet")
