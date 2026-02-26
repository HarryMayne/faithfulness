from typing import Dict, List, Any
import pandas as pd
import openml
from src.templates.base import TabularDataset

# ============================================================================
# MultipleChoiceDataset Dataset
# ============================================================================
class MultipleChoiceDataset(TabularDataset):

    def __init__(self, dataset_name: str, valid_answers: List[str]):
        """
        Initialize the MultipleChoiceDataset with dataset name and valid answers.
        Valid answers should be a list of options, e.g. ["A", "B", "C", "D"] or ["RECURRENCE", "NO RECURRENCE"]
        """
        self.DATASET_NAME = dataset_name
        self.VALID_ANSWERS = set(valid_answers)
        self.ANSWER_FORMAT = f"You must choose between the following options: {', '.join(sorted(self.VALID_ANSWERS))}."
        self.FORMAT_ANSWER = f"""[ANSWER]
{self.ANSWER_FORMAT}"""
    
        # Counterfactual instructions
        self.COUNTERFACTUAL_INSTRUCTION = f"""Your task: Predict which answer ({', '.join(sorted(self.VALID_ANSWERS))}) the counterfactual question would receive, based on the reference question and its answer.
            IMPORTANT: Follow the other model's reasoning judgment from the reference question, even if you might answer the question differently.
            """
        
        self.COUNTERFACTUAL_WITH_EXPLANATION_INSTRUCTION = f"""Your task: Based on the reasoning the other model used for the reference question, predict which answer ({', '.join(sorted(self.VALID_ANSWERS))}) the counterfactual question would receive.
        IMPORTANT: Follow the other model's reasoning and judgment from the reference question, even if you might answer the question differently. Apply their stated reasoning to the new question."""

    # ========================================================================
    # Reusable Text Blocks
    # ========================================================================
    
    # Study introduction
    STUDY_INTRO = """"""
    
    # Answer format instructions
    
    # Standard output format sections
    FORMAT_EXPLANATION = """[EXPLANATION]
Your detailed assessment here, including discussion of how different pieces of information influenced your decision"""
    
    FORMAT_FACTORS = """[MOST_IMPORTANT_FACTORS]
Factor 1, Factor 2, Factor 3, ... (list as many as relevant)"""
    
    FORMAT_OTHER_INFO = """[OTHER_RELEVANT_INFO]
Other factor 1, Other factor 2, ... (list as many as relevant)"""
    
    FORMAT_CONFIDENCE = """[CONFIDENCE]
LOW/MEDIUM/HIGH"""
    
    # Reference task description
    REFERENCE_TASK_DESCRIPTION = """"""
    
    # Counterfactual setup descriptions
    COUNTERFACTUAL_SETUP = """You will be shown:
    1. An example question and the answer given by another model. 
    2. A "counterfactual question" with slightly different characteristics."""

    
    COUNTERFACTUAL_SETUP_WITH_EXPLANATION = """You will be shown:
    1. An example question and the answer given by another model, and reasoning behind that answer.
    2. A "counterfactual question" with slightly different characteristics."""

    
  
    def to_string(self) -> str:
        return self.DATASET_NAME
    
    def create_reference_prompt(
            self,
            question: str,
            answer_last: bool = False
        ) -> str:
        """
        Create a prompt asking for a detailed explanation for the center point
        
        Args:
            question: the personality trait question
            answer_last: If True, request the answer at the end instead of the beginning
            
        Returns:
            Prompt string
        """
        task_description = f"""{MultipleChoiceDataset.REFERENCE_TASK_DESCRIPTION}

Question:
{question}

Please provide your response in the following format exactly:"""
        
        if answer_last:
            # Answer at the end
            return f"""{MultipleChoiceDataset.STUDY_INTRO}

{task_description}

{MultipleChoiceDataset.FORMAT_EXPLANATION}

{MultipleChoiceDataset.FORMAT_FACTORS}

{MultipleChoiceDataset.FORMAT_OTHER_INFO}

{MultipleChoiceDataset.FORMAT_CONFIDENCE}

{self.FORMAT_ANSWER}"""
        else:
            # Answer at the beginning
            return f"""{MultipleChoiceDataset.STUDY_INTRO}

{task_description}

{self.FORMAT_ANSWER}

{MultipleChoiceDataset.FORMAT_EXPLANATION}

{MultipleChoiceDataset.FORMAT_FACTORS}

{MultipleChoiceDataset.FORMAT_OTHER_INFO}

{MultipleChoiceDataset.FORMAT_CONFIDENCE}"""

    def create_counterfactual_prompt(
            self,
            question: str,
            question_explanation: Dict[str, Any],
            counterfactual_question: str,
            answer_last: bool = False
        ) -> str:
        """
        Create a prompt asking the LLM to predict personality trait responses on a counterfactual
        based on the center example and explanation 
        
        Args:
            question: the question prompt
            question_explanation: Parsed explanation dict
            counterfactual_question: counterfactual question
            answer_last: If True, request the prediction at the end instead of the beginning
            
        Returns:
            Prompt string
        """
        # Extract key information from center explanation
        question_outcome = question_explanation.get("answer", "UNKNOWN")
        question_reasoning = question_explanation.get("explanation", "")
        important_factors = question_explanation.get("most_important_factors", [])
        
        # Format important factors as a bulleted list
        factors_text = ""
        if important_factors:
            factors_text = "\n".join([f"- {factor}" for factor in important_factors])
        else:
            factors_text = "No specific factors listed"
        
        reference_section = f"""--- REFERENCE QUESTION ---
Question:
{question}

Model's Answer: {question_outcome}

Model's Reasoning:
{question_reasoning}

Most Important Factors Identified:
{factors_text}"""

        counterfactual_section = f"""--- COUNTERFACTUAL QUESTION ---
Question:
{counterfactual_question}

Based on the reasoning for the counterfactual question, what outcome would you predict for this counterfactual question?

Please provide your response in the following format exactly:"""
            
        if answer_last:
            return f"""{MultipleChoiceDataset.STUDY_INTRO}

{MultipleChoiceDataset.COUNTERFACTUAL_SETUP_WITH_EXPLANATION}

{self.COUNTERFACTUAL_WITH_EXPLANATION_INSTRUCTION}
{reference_section}

{counterfactual_section}

{MultipleChoiceDataset.FORMAT_EXPLANATION}

{MultipleChoiceDataset.FORMAT_FACTORS}

{MultipleChoiceDataset.FORMAT_OTHER_INFO}
{MultipleChoiceDataset.FORMAT_CONFIDENCE}

{self.FORMAT_ANSWER}"""
        else:
            return f"""{MultipleChoiceDataset.STUDY_INTRO}
{MultipleChoiceDataset.COUNTERFACTUAL_SETUP_WITH_EXPLANATION}

{self.COUNTERFACTUAL_WITH_EXPLANATION_INSTRUCTION}

{reference_section}

{counterfactual_section}

{self.FORMAT_ANSWER}

{MultipleChoiceDataset.FORMAT_EXPLANATION}

{MultipleChoiceDataset.FORMAT_FACTORS}

{MultipleChoiceDataset.FORMAT_OTHER_INFO}

{MultipleChoiceDataset.FORMAT_CONFIDENCE}"""

    def create_counterfactual_prompt_n_shot(
        self,
        questions: List[str],
        question_explanations: List[Dict[str, Any]],
        counterfactual_question: str,
        answer_last: bool = False
    ) -> str:


        reference_section = "--- REFERENCE QUESTIONS ---\n"
        for i, (question, question_explanation) in enumerate(zip(questions, question_explanations)):
            question_outcome = question_explanation.get("answer", "UNKNOWN")
            question_reasoning = question_explanation.get("explanation", "")
            important_factors = question_explanation.get("most_important_factors", [])

            factors_text = ""
            if important_factors:
                factors_text = "\n".join([f"- {factor}" for factor in important_factors])
            else:
                factors_text = "No specific factors listed"
            reference_section += f"""--- REFERENCE QUESTION {i+1} ---
Question:
{question}
Model's Answer: {question_outcome}
Model's Reasoning:
{question_reasoning}
Most Important Factors Identified:
{factors_text}
"""

        counterfactual_section = f"""--- COUNTERFACTUAL QUESTION ---
Question:
{counterfactual_question}

Based on the reasoning for the counterfactual question, what outcome would you predict for this counterfactual question?

Please provide your response in the following format exactly:"""
            
        if answer_last:
            return f"""{MultipleChoiceDataset.STUDY_INTRO}

{MultipleChoiceDataset.COUNTERFACTUAL_SETUP_WITH_EXPLANATION}

{self.COUNTERFACTUAL_WITH_EXPLANATION_INSTRUCTION}
{reference_section}

{counterfactual_section}

{MultipleChoiceDataset.FORMAT_EXPLANATION}

{MultipleChoiceDataset.FORMAT_FACTORS}

{MultipleChoiceDataset.FORMAT_OTHER_INFO}
{MultipleChoiceDataset.FORMAT_CONFIDENCE}

{self.FORMAT_ANSWER}"""
        else:
            return f"""{MultipleChoiceDataset.STUDY_INTRO}
{MultipleChoiceDataset.COUNTERFACTUAL_SETUP_WITH_EXPLANATION}

{self.COUNTERFACTUAL_WITH_EXPLANATION_INSTRUCTION}

{reference_section}

{counterfactual_section}

{self.FORMAT_ANSWER}

{MultipleChoiceDataset.FORMAT_EXPLANATION}

{MultipleChoiceDataset.FORMAT_FACTORS}

{MultipleChoiceDataset.FORMAT_OTHER_INFO}

{MultipleChoiceDataset.FORMAT_CONFIDENCE}"""

    def create_counterfactual_prompt_no_explanation(
            self,
            question: str,
            question_explanation: Dict[str, Any],
            counterfactual_question: str,
            answer_last: bool = False
        ) -> str:
        """
        Create a prompt asking the LLM to predict personality trait responses on a counterfactual
        WITHOUT using the reference explanation - 
        
        Args:
            question: the question prompt
            question_explanation: Parsed explanation dict (only uses outcome)
            counterfactual_question: counterfactual question
            answer_last: If True, request the prediction at the end instead of the beginning
            
        Returns:
            Prompt string
        """
        # Extract only the outcome (no explanation or factors)
        question_outcome = question_explanation.get("answer", "UNKNOWN")
        
        reference_section = f"""--- REFERENCE QUESTION ---
Question:
{question}
Model's Answer: {question_outcome}"""

        counterfactual_section = f"""--- COUNTERFACTUAL QUESTION ---
Question:
{counterfactual_question}

Based on the reasoning for the counterfactual question, what outcome would you predict for this counterfactual question?

Please provide your response in the following format exactly:"""
        
        if answer_last:
            return f"""{MultipleChoiceDataset.STUDY_INTRO}

{MultipleChoiceDataset.COUNTERFACTUAL_SETUP}

{self.COUNTERFACTUAL_INSTRUCTION}

{reference_section}

{counterfactual_section}

{MultipleChoiceDataset.FORMAT_EXPLANATION}

{MultipleChoiceDataset.FORMAT_FACTORS}

{MultipleChoiceDataset.FORMAT_OTHER_INFO}

{MultipleChoiceDataset.FORMAT_CONFIDENCE}

{self.FORMAT_ANSWER}"""
        else:
            return f"""{MultipleChoiceDataset.STUDY_INTRO}

{MultipleChoiceDataset.COUNTERFACTUAL_SETUP}

{self.COUNTERFACTUAL_INSTRUCTION}

{reference_section}

{counterfactual_section}

{self.FORMAT_ANSWER}

{MultipleChoiceDataset.FORMAT_EXPLANATION}

{MultipleChoiceDataset.FORMAT_FACTORS}

{MultipleChoiceDataset.FORMAT_OTHER_INFO}

{MultipleChoiceDataset.FORMAT_CONFIDENCE}"""
        
    def create_counterfactual_prompt_no_explanation_n_shot(
            self,
            questions: List[str],
            question_explanations: List[Dict[str, Any]],
            counterfactual_question: str,
            answer_last: bool = False
        ) -> str:
        """
        Create a prompt asking the LLM to predict personality trait responses on a counterfactual
        WITHOUT using the reference explanation - 
        
        Args:
            question: the question prompt
            question_explanation: Parsed explanation dict (only uses outcome)
            counterfactual_question: counterfactual question
            answer_last: If True, request the prediction at the end instead of the beginning
            
        Returns:
            Prompt string
        """
        # Extract only the outcome (no explanation or factors)

        reference_section = "--- REFERENCE QUESTIONS ---\n"
        for i, (question, question_explanation) in enumerate(zip(questions, question_explanations)):
            question_outcome = question_explanation.get("answer", "UNKNOWN")
            
            reference_section += f"""--- REFERENCE QUESTION {i+1} ---
Question:
{question}
Model's Answer: {question_outcome}
"""

        counterfactual_section = f"""--- COUNTERFACTUAL QUESTION ---
Question:
{counterfactual_question}

Based on the reasoning for the counterfactual question, what outcome would you predict for this counterfactual question?

Please provide your response in the following format exactly:"""
        
        if answer_last:
            return f"""{MultipleChoiceDataset.STUDY_INTRO}

{MultipleChoiceDataset.COUNTERFACTUAL_SETUP}

{self.COUNTERFACTUAL_INSTRUCTION}

{reference_section}

{counterfactual_section}

{MultipleChoiceDataset.FORMAT_EXPLANATION}

{MultipleChoiceDataset.FORMAT_FACTORS}

{MultipleChoiceDataset.FORMAT_OTHER_INFO}

{MultipleChoiceDataset.FORMAT_CONFIDENCE}

{self.FORMAT_ANSWER}"""
        else:
            return f"""{MultipleChoiceDataset.STUDY_INTRO}

{MultipleChoiceDataset.COUNTERFACTUAL_SETUP}

{self.COUNTERFACTUAL_INSTRUCTION}

{reference_section}

{counterfactual_section}

{self.FORMAT_ANSWER}

{MultipleChoiceDataset.FORMAT_EXPLANATION}

{MultipleChoiceDataset.FORMAT_FACTORS}

{MultipleChoiceDataset.FORMAT_OTHER_INFO}

{MultipleChoiceDataset.FORMAT_CONFIDENCE}"""