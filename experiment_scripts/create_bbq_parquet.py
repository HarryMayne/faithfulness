"""
Convert bbq_cleaned.csv (with 'question' and 'counterfactual_question' columns)
into a formatted parquet file for the faithfulness pipeline.

Usage:
    python experiment_scripts/create_bbq_parquet.py
"""
import sys
sys.path.insert(0, ".")

import pandas as pd
from src.schema import (
    FaithfulnessRecord,
    OriginalQuestion,
    CounterfactualInfo,
    CounterfactualDatabase,
)
from src.templates.bbq_dataset import BBQDataset

INPUT_CSV = "data/raw/bbq_cleaned.csv"
OUTPUT_PARQUET = "data/generated_counterfactuals/bbq_cleaned_formatted.parquet"

def main():
    print(f"Reading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows with columns: {list(df.columns)}")

    dataset = BBQDataset()
    db = CounterfactualDatabase()

    for idx, row in df.iterrows():
        question = row["question"]
        counterfactual = row["counterfactual_question"]

        record = FaithfulnessRecord(
            original_question=OriginalQuestion(
                dataset=dataset.to_string(),
                question=question,
                question_prompt=dataset.create_reference_prompt(
                    question=question, answer_last=False
                ),
                question_idx=idx,
                description=question,
            ),
            counterfactual=CounterfactualInfo(
                generator_model="pre-provided",
                generator_method="pre-provided",
                question=counterfactual,
                question_prompt=dataset.create_reference_prompt(
                    question=counterfactual, answer_last=False
                ),
            ),
        )
        db.add_record(record)

    db.save_parquet(OUTPUT_PARQUET)
    print(f"Saved {len(db.records)} records to {OUTPUT_PARQUET}")


if __name__ == "__main__":
    main()
