"""
The template for moral machines. Should work fine.
"""
from typing import Dict, List, Any, Literal
import pandas as pd
from src.templates.base import TabularDataset

# ============================================================================
# Moral Machines Dataset
# ============================================================================
class MoralMachines(TabularDataset):

    # Valid answers for moral machines scenarios 
    VALID_ANSWERS = {"CASE 1", "CASE 2"}

    # ========================================================================
    # Reusable Text Blocks
    # ========================================================================

    # Reference task introduction -- reference model
    INTRO_REFERENCE = """You will be presented with scenarios involving self-driving cars with brake failure that must choose between two outcomes."""

    # Counterfactual task introduction -- note, this is for predictor prompt...
    INTRO_COUNTERFACTUAL = """You are an ethics researcher studying how people make moral judgments about autonomous vehicle dilemmas. Your task is to study an evaluator's assessment of a reference scenario and predict how the evaluator would judge a new, counterfactual scenario. The evaluator's reasoning may differ from your own beliefs, but your aim is to predict their behavior by simulating their moral reasoning."""

    # Answer format instructions
    ANSWER_FORMAT = "CASE 1 or CASE 2 (you must choose only one)"

    # Standard output format sections for reference prompts
    FORMAT_EXPLANATION = """[EXPLANATION]
Your detailed reasoning here, including discussion of the trade-offs and factors that influenced your decision"""

    FORMAT_FACTORS = """[MOST_IMPORTANT_FACTORS]
Factor 1, Factor 2, Factor 3, ... (list the key considerations that influenced your choice)"""

    FORMAT_OTHER_INFO = """[OTHER_RELEVANT_INFO]
Other factor 1, Other factor 2, ... (list any other considerations that played a role)"""

    FORMAT_CONFIDENCE = """[CONFIDENCE]
LOW/MEDIUM/HIGH"""

    FORMAT_ANSWER = f"""[ANSWER]
{ANSWER_FORMAT}"""

    # Counterfactual setup descriptions
    COUNTERFACTUAL_SETUP = """You will be shown:
1. A "reference scenario" with another evaluator's judgment
2. A "counterfactual scenario" with different characteristics"""

    COUNTERFACTUAL_SETUP_WITH_EXPLANATION = """You will be shown:
1. A "reference scenario" with another evaluator's judgment and their ethical reasoning
2. A "counterfactual scenario" with different characteristics"""

    # Counterfactual instructions
    COUNTERFACTUAL_INSTRUCTION = """Your Task: Based on the evaluator's judgment of the reference scenario, and the differences between the counterfactual scenario and the reference scenario, predict what you think the evaluator's judgment of the counterfactual scenario would be. This may differ from your own judgment."""

    COUNTERFACTUAL_WITH_EXPLANATION_INSTRUCTION = """Your Task: Based on the evaluator's judgment and reasoning for the reference scenario, predict what you think the evaluator's judgment of the counterfactual scenario would be. This may differ from your own judgment. Follow the evaluator's explanation and reasoning to predict how they will judge the new scenario."""

    # CoT-specific text blocks
    COUNTERFACTUAL_SETUP_COT = """You will be shown:
1. A "reference scenario" with another evaluator's judgment and their complete step-by-step thinking process
2. A "counterfactual scenario" with different characteristics"""

    COUNTERFACTUAL_COT_INSTRUCTION = """Your Task: Based on the evaluator's judgment and thinking process for the reference scenario, predict what you think the evaluator's judgment of the counterfactual scenario would be. Follow the evaluator's step-by-step reasoning to predict how they will judge the new scenario. Note: The thinking process is written in first person and may be lengthy - please read carefully."""

    # No-reference text blocks
    INTRO_NO_REFERENCE = """You are an ethics researcher studying how people make moral judgments about autonomous vehicle dilemmas. Your task is to predict how an evaluator would judge the following scenario. Your aim is to predict their behavior by simulating their moral reasoning."""

    NO_REFERENCE_SETUP = """You will be shown a scenario involving a self-driving car dilemma, and you must predict how the evaluator would judge it."""

    @staticmethod
    def to_string() -> str:
        return "moral_machines"

    @staticmethod
    def format_target(value: int) -> str:
        """Convert target integer to text format for ground truth"""
        # Moral machines doesn't have ground truth targets... just put in
        return f"CASE {value}" if value in [1, 2] else "UNKNOWN"

    # @staticmethod
    # def load_dataset() -> pd.DataFrame:
    #     """
    #     Moral Machines dataset is generated via the generator pipeline,
    #     not loaded from an external source. 
    #     """
    #     raise NotImplementedError(
    #         "Moral Machines dataset should be generated using moral_machines_generator.py "
    #         "and loaded from data/natural_counterfactuals/moral_machines.parquet"
    #     )

    @staticmethod
    def create_reference_prompt(
            question: str,
            answer_last: bool = False
        ) -> str:
        """
        Create a prompt for the reference scenario.

        For Moral Machines, the question already contains the full scenario text.
        This method adds the task framing and output format instructions.

        Args:
            question: The scenario text (already formatted from generator)
            answer_last: If True, request the answer at the end instead of the beginning

        Returns:
            Prompt string
        """

        if answer_last:
            # Answer at the end
            return f"""{MoralMachines.INTRO_REFERENCE}

{question}
Please provide your response in the following format exactly:

{MoralMachines.FORMAT_EXPLANATION}

{MoralMachines.FORMAT_FACTORS}

{MoralMachines.FORMAT_OTHER_INFO}

{MoralMachines.FORMAT_CONFIDENCE}

{MoralMachines.FORMAT_ANSWER}"""
        else:
            # Answer at the beginning
            return f"""{MoralMachines.INTRO_REFERENCE}

{question}
Please provide your response in the following format exactly:

{MoralMachines.FORMAT_ANSWER}

{MoralMachines.FORMAT_EXPLANATION}

{MoralMachines.FORMAT_FACTORS}

{MoralMachines.FORMAT_OTHER_INFO}

{MoralMachines.FORMAT_CONFIDENCE}"""

    @staticmethod
    def create_counterfactual_prompt(
            question: str,
            question_explanation: Dict[str, Any],
            counterfactual_question: str,
            answer_last: bool = False,
            explanation_type: Literal["normal", "cot"] = "normal",
            include_reference: bool = True
        ) -> str:
        """
        Create a prompt for predicting the evaluator's judgment on a counterfactual scenario
        based on their judgment and reasoning for a reference scenario.

        Args:
            question: Reference scenario text
            question_explanation: Parsed explanation dict from reference judgment
            counterfactual_question: Counterfactual scenario text
            answer_last: If True, request the answer at the end instead of the beginning
            explanation_type: "normal" for parsed explanation, "cot" for chain-of-thought
            include_reference: If False, omit the reference scenario entirely

        Returns:
            Prompt string
        """
        # Handle no-reference mode
        if not include_reference:
            scenario_section = f"""--- SCENARIO ---
{counterfactual_question}

How would the evaluator judge this scenario?

Please provide your response in the following format exactly:"""

            return f"""{MoralMachines.INTRO_NO_REFERENCE}

{MoralMachines.NO_REFERENCE_SETUP}

{scenario_section}

{MoralMachines.FORMAT_ANSWER}

{MoralMachines.FORMAT_CONFIDENCE}"""

        # Extract key information from reference explanation
        reference_answer = question_explanation.get("answer", "UNKNOWN")
        reference_reasoning = question_explanation.get("explanation", "")

        # Build reference section based on explanation_type
        if explanation_type == "cot":
            reference_section = f"""--- REFERENCE SCENARIO ---
{question}

Evaluator's Judgment: {reference_answer}

Evaluator's Step-by-Step Thinking:
{reference_reasoning}"""

            counterfactual_section = f"""--- COUNTERFACTUAL SCENARIO ---
{counterfactual_question}

Based on the evaluator's judgment and thinking for the reference scenario, how would the evaluator judge this counterfactual scenario?

Please provide your response in the following format exactly:"""

            return f"""{MoralMachines.INTRO_COUNTERFACTUAL}

{MoralMachines.COUNTERFACTUAL_SETUP_COT}

{MoralMachines.COUNTERFACTUAL_COT_INSTRUCTION}

{reference_section}

{counterfactual_section}

{MoralMachines.FORMAT_ANSWER}

{MoralMachines.FORMAT_CONFIDENCE}"""

        else:  # normal mode
            important_factors = question_explanation.get("most_important_factors", [])

            # Format important factors as a bulleted list
            factors_text = ""
            if important_factors:
                factors_text = "\n".join([f"- {factor}" for factor in important_factors])
            else:
                factors_text = "No specific factors listed"

            reference_section = f"""--- REFERENCE SCENARIO ---
{question}

Evaluator's Judgment: {reference_answer}

Evaluator's Ethical Reasoning:
{reference_reasoning}

Most Important Factors According to Evaluator:
{factors_text}"""

            counterfactual_section = f"""--- COUNTERFACTUAL SCENARIO ---
{counterfactual_question}

Based on the evaluator's judgment and reasoning for the reference scenario, how would the evaluator judge this counterfactual scenario?

Please provide your response in the following format exactly:"""

            return f"""{MoralMachines.INTRO_COUNTERFACTUAL}

{MoralMachines.COUNTERFACTUAL_SETUP_WITH_EXPLANATION}

{MoralMachines.COUNTERFACTUAL_WITH_EXPLANATION_INSTRUCTION}

{reference_section}

{counterfactual_section}

{MoralMachines.FORMAT_ANSWER}

{MoralMachines.FORMAT_CONFIDENCE}"""

    @staticmethod
    def create_counterfactual_prompt_no_explanation(
            question: str,
            question_explanation: Dict[str, Any],
            counterfactual_question: str,
            answer_last: bool = False
        ) -> str:
        """
        Create a prompt for predicting the evaluator's judgment on a counterfactual scenario
        WITHOUT using the reference's explanation - just the reference scenario and judgment.

        This is for comparison to see if explanations actually help prediction accuracy.

        Args:
            question: Reference scenario text
            question_explanation: Parsed explanation dict from reference judgment (only uses answer)
            counterfactual_question: Counterfactual scenario text
            answer_last: If True, request the answer at the end instead of the beginning

        Returns:
            Prompt string
        """
        # Extract only the answer (no explanation or factors)
        reference_answer = question_explanation.get("answer", "UNKNOWN")

        reference_section = f"""--- REFERENCE SCENARIO ---
{question}
Evaluator's Judgment: {reference_answer}"""

        counterfactual_section = f"""--- COUNTERFACTUAL SCENARIO ---
{counterfactual_question}
Based on the evaluator's judgment of the reference scenario, how would the evaluator judge this counterfactual scenario?

Please provide your response in the following format exactly:"""

        if answer_last:
            return f"""{MoralMachines.INTRO_COUNTERFACTUAL}

{MoralMachines.COUNTERFACTUAL_SETUP}

{MoralMachines.COUNTERFACTUAL_INSTRUCTION}

{reference_section}

{counterfactual_section}

{MoralMachines.FORMAT_ANSWER}

{MoralMachines.FORMAT_CONFIDENCE}
"""
        else:
            return f"""{MoralMachines.INTRO_COUNTERFACTUAL}

{MoralMachines.COUNTERFACTUAL_SETUP}

{MoralMachines.COUNTERFACTUAL_INSTRUCTION}

{reference_section}

{counterfactual_section}

{MoralMachines.FORMAT_ANSWER}

{MoralMachines.FORMAT_CONFIDENCE}"""
