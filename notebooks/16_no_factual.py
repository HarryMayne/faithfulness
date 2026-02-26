# %%
import pandas as pd
import matplotlib.pyplot as plt
import re

# %%
standard_parquet = pd.read_parquet("../experiments/cot/predictions_normal.parquet")
no_factual_parquet = pd.read_parquet("../experiments/cot/predictions_no_factual.parquet")


standard_parquet['counterfactual_predictor_response_without_explanation_predictor_answers'] = no_factual_parquet['counterfactual_predictor_response_with_explanation_predictor_answers']

standard_parquet['counterfactual_predictor_response_without_explanation_predictor_names'] = no_factual_parquet['counterfactual_predictor_response_with_explanation_predictor_names']

# %%
# save
standard_parquet.to_parquet("../experiments/cot/predictions_no_factual_baseline.parquet")
# %%
