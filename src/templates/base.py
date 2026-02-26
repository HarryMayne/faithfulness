"""
Base template class. 
"""
from pyparsing import abstractmethod
from typing import Dict, List, Any, List, Set
import pandas as pd

# ============================================================================
# Abstract base class for tabular datasets
# ============================================================================
class TabularDataset:
    
    # Set of valid answers for this dataset (e.g., {"YES", "NO"})
    # Must be overridden by subclasses
    VALID_ANSWERS: Set[str] = set()

    def __init__(self):
        pass

    @staticmethod
    @abstractmethod
    def to_string():
        # Should be implemented for file naming and identification
        pass

    #####################################################
    # Abstract methods for loading and processing data
    #####################################################

    @staticmethod
    @abstractmethod
    def load_dataset() -> pd.DataFrame:
       pass
    
    @staticmethod
    @abstractmethod
    def description_generator(row_idx: int, row_data: pd.Series, feature_cols: List[str]) -> str:
        pass

    ######################################################
    # Prompt creation methods
    ######################################################

    @staticmethod
    @abstractmethod
    def create_reference_prompt(
            question: str,
            answer_last: bool = False
        ) -> str:
        pass

    @staticmethod
    @abstractmethod
    def create_counterfactual_prompt(
            question: str,
            question_explanation: Dict[str, Any],
            counterfactual_question: str,
            answer_last: bool = False
        ) -> str:
        pass

    @staticmethod
    @abstractmethod
    def create_counterfactual_prompt_no_explanation(
            question: str,
            question_explanation: Dict[str, Any],
            counterfactual_question: str,
            answer_last: bool = False
        ) -> str:
        pass
