from typing import Dict, List, Any
import pandas as pd
import openml
from src.templates.base import TabularDataset

# ============================================================================
# Trait Dataset
# ============================================================================
class Trait(TabularDataset):
    
    # Valid answers for breast cancer recurrence prediction
    VALID_ANSWERS = {"A", "B", "C", "D"}
    
    # ========================================================================
    # Reusable Text Blocks
    # ========================================================================
    
    # Study introduction
    STUDY_INTRO = """"""
    
    # Answer format instructions
    ANSWER_FORMAT = "You must choose between the following options: A, B, C, or D."
    
    # Standard output format sections
    FORMAT_EXPLANATION = """[EXPLANATION]
Your detailed assessment here, including discussion of how different pieces of information influenced your decision"""
    
    FORMAT_FACTORS = """[MOST_IMPORTANT_FACTORS]
Factor 1, Factor 2, Factor 3, ... (list as many as relevant)"""
    
    FORMAT_OTHER_INFO = """[OTHER_RELEVANT_INFO]
Other factor 1, Other factor 2, ... (list as many as relevant)"""
    
    FORMAT_CONFIDENCE = """[CONFIDENCE]
LOW/MEDIUM/HIGH"""
    
    FORMAT_ANSWER = f"""[ANSWER]
{ANSWER_FORMAT}"""
    
    # Reference task description
    REFERENCE_TASK_DESCRIPTION = """"""
    
    # Counterfactual setup descriptions
    COUNTERFACTUAL_SETUP = """You will be shown:
    1. An example personality trait question and the answer given by another model. 
    2. A "counterfactual personality trait question" with slightly different characteristics."""

    
    COUNTERFACTUAL_SETUP_WITH_EXPLANATION = """You will be shown:
    1. An example personality trait question and the answer given by another model, and reasoning behind that answer.
    2. A "counterfactual personality trait question" with slightly different characteristics."""

    
    # Counterfactual instructions
    COUNTERFACTUAL_INSTRUCTION = """Your task: Predict which answer (A, B, C, or D) the counterfactual personality trait question would receive, based on the reference question and its answer.
        IMPORTANT: Follow the other model's reasoning judgment from the reference question, even if you might answer the question differently.
        """
    
    COUNTERFACTUAL_WITH_EXPLANATION_INSTRUCTION = """Your task: Based on the reasoning the other model used for the reference personality trait question, predict which answer (A, B, C, or D) the counterfactual personality trait question would receive.

    IMPORTANT: Follow the other model's reasoning and judgment from the reference question, even if you might answer the question differently. Apply their stated reasoning to the new question."""

    def to_string() -> str:
        return "trait"
    
    @staticmethod
    def create_reference_prompt(
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
        task_description = f"""{Trait.REFERENCE_TASK_DESCRIPTION}

Question:
{question}

Please provide your response in the following format exactly:"""
        
        if answer_last:
            # Answer at the end
            return f"""{Trait.STUDY_INTRO}

{task_description}

{Trait.FORMAT_EXPLANATION}

{Trait.FORMAT_FACTORS}

{Trait.FORMAT_OTHER_INFO}

{Trait.FORMAT_CONFIDENCE}

{Trait.FORMAT_ANSWER}"""
        else:
            # Answer at the beginning
            return f"""{Trait.STUDY_INTRO}

{task_description}

{Trait.FORMAT_ANSWER}

{Trait.FORMAT_EXPLANATION}

{Trait.FORMAT_FACTORS}

{Trait.FORMAT_OTHER_INFO}

{Trait.FORMAT_CONFIDENCE}"""

    @staticmethod
    def create_counterfactual_prompt(
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

Model'sAnswer: {question_outcome}

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
            return f"""{Trait.STUDY_INTRO}

{Trait.COUNTERFACTUAL_SETUP_WITH_EXPLANATION}

{Trait.COUNTERFACTUAL_WITH_EXPLANATION_INSTRUCTION}

{reference_section}

{counterfactual_section}

{Trait.FORMAT_EXPLANATION}

{Trait.FORMAT_FACTORS}

{Trait.FORMAT_OTHER_INFO}
{Trait.FORMAT_CONFIDENCE}

{Trait.FORMAT_ANSWER}"""
        else:
            return f"""{Trait.STUDY_INTRO}

{Trait.COUNTERFACTUAL_SETUP_WITH_EXPLANATION}

{Trait.COUNTERFACTUAL_WITH_EXPLANATION_INSTRUCTION}

{reference_section}

{counterfactual_section}

{Trait.FORMAT_ANSWER}

{Trait.FORMAT_EXPLANATION}

{Trait.FORMAT_FACTORS}

{Trait.FORMAT_OTHER_INFO}

{Trait.FORMAT_CONFIDENCE}"""

    @staticmethod
    def create_counterfactual_prompt_no_explanation(
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
            return f"""{Trait.STUDY_INTRO}

{Trait.COUNTERFACTUAL_SETUP}

{Trait.COUNTERFACTUAL_INSTRUCTION}

{reference_section}

{counterfactual_section}

{Trait.FORMAT_EXPLANATION}

{Trait.FORMAT_FACTORS}

{Trait.FORMAT_OTHER_INFO}

{Trait.FORMAT_CONFIDENCE}

{Trait.FORMAT_ANSWER}"""
        else:
            return f"""{Trait.STUDY_INTRO}

{Trait.COUNTERFACTUAL_SETUP}

{Trait.COUNTERFACTUAL_INSTRUCTION}

{reference_section}

{counterfactual_section}

{Trait.FORMAT_ANSWER}

{Trait.FORMAT_EXPLANATION}

{Trait.FORMAT_FACTORS}

{Trait.FORMAT_OTHER_INFO}

{Trait.FORMAT_CONFIDENCE}"""