from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any, Type
from tqdm import tqdm
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path
from src.templates.heart_disease import HeartDisease
from src.templates.pima_diabetes import PimaDiabetes
from src.templates.breast_cancer_recurrence import BreastCancerRecurrence
from src.templates.multiple_choice_dataset import MultipleChoiceDataset
from src.templates.trait import Trait
from src.templates.income import IncomeDataset
from src.templates.attrition import AttritionDataset
from src.templates.moral_machines import MoralMachines
from src.templates.bank_marketing import BankMarketing
from src.templates.bbq_dataset import BBQDataset
import random

@dataclass
class ModelInfo:
    """
    Information about a model used for generation.
    """
    model: Optional[str] = None                 # Name of the model
    temperature: Optional[float] = None         # Temperature used for generation
    max_tokens: Optional[int] = None            # Max tokens used for generation
    thinking: Optional[str] = None             # Whether this is a thinking/reasoning model variant
    seed: Optional[int] = None                  # Random seed
    additional_params: Optional[dict] = None    # Additional sampling params, e.g. top_k, min_p


# Backwards compatibility aliases
ReferenceModelInfo = ModelInfo
PredictorModelInfo = ModelInfo


@dataclass
class Response:
    """
    Encapsulates a model's response with metadata.
    """
    cot: Optional[str] = None                                   # chain-of-thought for reasoning models
    raw_response: Optional[str] = None                          # Raw LLM output (JSON string)
    parsed_response: Optional[Dict[str, Any]] = None            # Parsed JSON dict
    answer: Optional[str] = None                                # Extracted answer field from parsed response
    model_info: Optional[ModelInfo] = None                      # Model that generated this response
    predictor_answers: Optional[List[Optional[str]]] = None     # All predictor answers from all models/repeats
    predictor_names: Optional[List[str]] = None                 # Model name for each answer in predictor_answers
    input_tokens: Optional[int] = None                          # Number of input tokens
    reasoning_tokens: Optional[int] = None                      # Number of reasoning (CoT) tokens
    output_tokens: Optional[int] = None                         # Number of output tokens (including reasoning)


@dataclass
class OriginalQuestion:
    dataset: str                                        # Name of the dataset (e.g., 'SQuAD', 'TruthfulQA'). 
    question: str                                       # The original question text (unformatted - just the natural language question)          
    question_prompt: str                                # The original question text (**full prompt**) 
    question_idx: int                                   # Unique index of the question within the dataset.
    ground_truth: Optional[str] = None                  # Ground-truth answer, if available.
    answer_first: Optional[bool] = None                 # Whether the original question asks for the answer first or answer second (after explanation)
    description: Optional[str] = None                   # Natural language description (e.g., "This is a male patient, 50-60 years old...")
    question_options: Optional[dict] = None             # Multiple-choice options, if applicable. Keys are A,B,C, D, values are options.
    reference_response: Optional[Response] = None       # Reference model's response to this original question (with explanation)


@dataclass
class CounterfactualInfo:
    generator_model: str                                # Name of the model that generated the counterfactual.
    generator_method: str                               # Method used to generate the counterfactual (e.g., 'Matton').
    question: str                                       # The counterfactual question text (unformatted - just the natural language question)
    question_prompt: str                                # The counterfactual question text (**full prompt**)
    generator_model_info: Optional[ModelInfo] = None    # Metadata about the generator model.
    generator_model_cot: Optional[str] = None           # Save the CoT from the counterfactual generator model. Useful for debugging.
    generator_model_raw: Optional[str] = None           # Raw generator output, if available.
    question_idx: Optional[int] = None                  # ← optional, auto-assigned later
    ground_truth: Optional[str] = None
    description: Optional[str] = None                   # Natural language description of counterfactual
    coherence_scored_by_generator: Optional[bool] = None
    coherence_explanation_by_generator: Optional[str] = None
    coherence_external_scoring_model: Optional[str] = None
    coherence_scored_by_external_model: Optional[bool] = None
    coherence_explanation_by_external_model: Optional[str] = None
    hamming_distance: Optional[int] = None
    question_options: Optional[dict] = None             # Multiple-choice options, if applicable. Keys are A,B,C, D, values are options.
    
    # Reference model's response to this counterfactual question (without explanation context)
    reference_response: Optional[Response] = None
    
    # Phase 3: Predictor prompts (generated using original question's reference answer)
    prompt_with_explanation: Optional[str] = None       # Prompt including original's explanation
    prompt_without_explanation: Optional[str] = None    # Prompt with only original's answer
    
    # Predictor model's responses to the prompts
    predictor_response_with_explanation: Optional[Response] = None
    predictor_response_without_explanation: Optional[Response] = None

    # Testability assessment (optional, scored 0-10)
    predictor_counterfactual_testability_score: Optional[float] = None
    predictor_counterfactual_testability_cot: Optional[str] = None
    
    # Cross-model experiment fields (only populated by cross_model_experiment.py)
    is_cross_model_explanation: Optional[bool] = None       # True if prompt uses explanation from different model
    explanation_source_model_info: Optional[ModelInfo] = None  # ModelInfo of the model that provided the explanation

@dataclass
class MatchInfo:
    """
    Evaluation metrics comparing predictor outputs against reference answers.
    """
    match_with_explanation: Optional[int] = None    # 1 if prediction (with explanation) matches reference answer, else 0.
    match_without_explanation: Optional[int] = None # 1 if prediction (without explanation) matches reference answer, else 0.
    match_delta: Optional[int] = None               # Difference: match_with_explanation - match_without_explanation.




@dataclass
class FaithfulnessRecord:
    """
    Represents one row in the counterfactual faithfulness/simulatability dataset.
    """
    original_question: OriginalQuestion
    counterfactual: CounterfactualInfo
    match_info: Optional[MatchInfo] = None

    def to_flat_dict(self):
        """Flatten the record for DataFrame/Parquet storage."""
        flat = {}
        
        # Flatten OriginalQuestion to add original_ prefix.
        if self.original_question is not None:
            for k, v in asdict(self.original_question).items():
                if k == 'reference_response' and isinstance(v, dict):
                    # Flatten Response object
                    for response_key, response_val in v.items():
                        if response_key == 'model_info' and isinstance(response_val, dict):
                            # Flatten ModelInfo
                            for model_key, model_val in response_val.items():
                                flat[f"original_reference_response_model_info_{model_key}"] = model_val
                        else:
                            flat[f"original_reference_response_{response_key}"] = response_val
                else:
                    flat[f"original_{k}"] = v
        
        # Flatten CounterfactualInfo
        if self.counterfactual is not None:
            for k, v in asdict(self.counterfactual).items():
                if k == 'generator_model_info' and isinstance(v, dict):
                    for model_key, model_val in v.items():
                        flat[f"counterfactual_generator_model_info_{model_key}"] = model_val
                    continue
                if k == 'explanation_source_model_info' and isinstance(v, dict):
                    for model_key, model_val in v.items():
                        flat[f"counterfactual_explanation_source_model_info_{model_key}"] = model_val
                    continue
                if k in ['reference_response', 'predictor_response_with_explanation', 'predictor_response_without_explanation'] and isinstance(v, dict):
                    # Flatten Response object
                    for response_key, response_val in v.items():
                        if response_key == 'model_info' and isinstance(response_val, dict):
                            # Flatten ModelInfo
                            for model_key, model_val in response_val.items():
                                flat[f"counterfactual_{k}_model_info_{model_key}"] = model_val
                        else:
                            flat[f"counterfactual_{k}_{response_key}"] = response_val
                else:
                    flat[f"counterfactual_{k}"] = v
        
        # Flatten MatchInfo
        if self.match_info is not None:
            for k, v in asdict(self.match_info).items():
                flat[f"match_{k}"] = v
        
        return flat


# -------------------------------------------------------------------------
# CounterfactualDatabase
# -------------------------------------------------------------------------

class CounterfactualDatabase:
    """
    A manager for FaithfulnessRecord objects.

    Handles:
        - Automatic index assignment.
        - Adding new records.
        - Saving/loading to Parquet.
        - Conversion to/from DataFrames.
    """

    dataset_class_map = {
            'heart_disease': HeartDisease,
            'pima_diabetes': PimaDiabetes,
            'breast_cancer_recurrence': BreastCancerRecurrence,
            'trait': Trait,
            'multiple_choice_dataset': MultipleChoiceDataset,
            'income': IncomeDataset,
            'attrition': AttritionDataset,
            'moral_machines': MoralMachines,
            'bank_marketing': BankMarketing,
            'bbq': BBQDataset,
        }
    
    def __init__(self):
        self.records: List[FaithfulnessRecord] = []
    # --------------------------------------------------

    def add_record(self, record: FaithfulnessRecord) -> None:
        """Add a new record and automatically assign indices."""

        # Assign unique cf_question index
        cf_question_indices = set(
            r.counterfactual.question_idx for r in self.records
            if r.counterfactual.question_idx is not None
        )
        max_index = max(cf_question_indices) if cf_question_indices else 100000000
        if record.counterfactual.question_idx is None:
            record.counterfactual.question_idx = max_index + 1
        self.records.append(record)

    # --------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the full database to a pandas DataFrame."""
        return pd.DataFrame([r.to_flat_dict() for r in self.records])

    # --------------------------------------------------

    def save_parquet(self, path: str | Path) -> None:
        """Save the entire database to a Parquet file."""
        df = self.to_dataframe()
        pq.write_table(pa.Table.from_pandas(df), path)

    # --------------------------------------------------

    @classmethod
    def load_parquet(cls, path: str | Path) -> "CounterfactualDatabase":
        """Load a CounterfactualDatabase from a Parquet file."""
        from .schema import (
            FaithfulnessRecord,
            OriginalQuestion,
            CounterfactualInfo,
            MatchInfo,
            Response,
            ModelInfo,
        )

        df = pq.read_table(path).to_pandas()
        db = cls()
        
        # Convert to list of dicts - much faster than iterrows()
        rows = df.to_dict('records')
        db.records = [None] * len(rows)  # Pre-allocate
        
        for i, row in enumerate(tqdm(rows, desc="Loading records")):
            # Reconstruct Response objects
            def reconstruct_response(prefix):
                """Reconstruct a Response object from flattened columns."""
                response_fields = {}
                model_info_fields = {}
                
                for k, v in row.items():
                    if k.startswith(f"{prefix}_model_info_"):
                        model_key = k[len(f"{prefix}_model_info_"):]
                        model_info_fields[model_key] = v
                    elif k.startswith(f"{prefix}_"):
                        response_key = k[len(f"{prefix}_"):]
                        if response_key != 'model_info':
                            response_fields[response_key] = v
                
                if response_fields:
                    response_fields['model_info'] = ModelInfo(**model_info_fields) if model_info_fields else None
                    return Response(**response_fields)
                return None
            
            # Reconstruct OriginalQuestion
            original_fields = {}
            for k, v in row.items():
                if k.startswith("original_") and not k.startswith("original_reference_response_"):
                    original_fields[k[len("original_"):]] = v
            original_fields['reference_response'] = reconstruct_response("original_reference_response")
            original_question = OriginalQuestion(**original_fields)
            
            # Reconstruct CounterfactualInfo
            counterfactual_fields = {}
            for k, v in row.items():
                if k.startswith("counterfactual_generator_model_info_"):
                    continue  # handled separately
                if k.startswith("counterfactual_explanation_source_model_info_"):
                    continue  # handled separately
                if k.startswith("counterfactual_") and not any(
                    k.startswith(f"counterfactual_{resp}_") 
                    for resp in ['reference_response', 'predictor_response_with_explanation', 'predictor_response_without_explanation']
                ):
                    counterfactual_fields[k[len("counterfactual_"):]] = v
            # Rebuild generator_model_info
            generator_model_info_fields = {}
            for k, v in row.items():
                if k.startswith("counterfactual_generator_model_info_"):
                    model_key = k[len("counterfactual_generator_model_info_"):]
                    generator_model_info_fields[model_key] = v
            counterfactual_fields['generator_model_info'] = ModelInfo(**generator_model_info_fields) if generator_model_info_fields else None
            # Rebuild explanation_source_model_info
            explanation_source_model_info_fields = {}
            for k, v in row.items():
                if k.startswith("counterfactual_explanation_source_model_info_"):
                    model_key = k[len("counterfactual_explanation_source_model_info_"):]
                    explanation_source_model_info_fields[model_key] = v
            counterfactual_fields['explanation_source_model_info'] = ModelInfo(**explanation_source_model_info_fields) if explanation_source_model_info_fields else None
            counterfactual_fields['reference_response'] = reconstruct_response("counterfactual_reference_response")
            counterfactual_fields['predictor_response_with_explanation'] = reconstruct_response("counterfactual_predictor_response_with_explanation")
            counterfactual_fields['predictor_response_without_explanation'] = reconstruct_response("counterfactual_predictor_response_without_explanation")
            counterfactual = CounterfactualInfo(**counterfactual_fields)
            
            # Reconstruct MatchInfo
            match_fields = {}
            for k, v in row.items():
                if k.startswith("match_"):
                    match_fields[k[len("match_"):]] = v
            match_info = MatchInfo(**match_fields) if match_fields else None

            record = FaithfulnessRecord(
                original_question=original_question,
                counterfactual=counterfactual,
                match_info=match_info,
            )
            db.records[i] = record
        return db