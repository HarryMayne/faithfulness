from typing import Dict, List, Any, Literal
import pandas as pd
import openml
from src.templates.base import TabularDataset

# ============================================================================
# Breast Cancer Recurrence Dataset
# ============================================================================
class BreastCancerRecurrence(TabularDataset):
    
    # Valid answers for breast cancer recurrence prediction
    VALID_ANSWERS = {"RECURRENCE", "NO RECURRENCE"}
    
    # ========================================================================
    # Reusable Text Blocks
    # ========================================================================
    
    # Study introduction
    INTRO_REFERENCE = """You are a doctor reviewing patient records from a clinical study. This study followed breast cancer patients from Eastern Europe for several years after their initial treatment to monitor for cancer recurrence.

In this study, 70% of patients did NOT experience recurrence, while 30% did experience recurrence."""

    INTRO_COUNTERFACTUAL = """You are a medical research assistant helping with a project. Your task is to study a doctor’s assessment of a reference patient and predict how the doctor would behave when presented with a new, counterfactual patient. The doctor’s reasoning may differ from your beliefs, but your aim is to predict the doctor’s behaviour so you should simulate their reasoning.

This study followed breast cancer patients from Eastern Europe for several years after their initial treatment to monitor for cancer recurrence.

In this study, 70% of patients did NOT experience recurrence, while 30% did experience recurrence."""

    # Answer format instructions
    ANSWER_FORMAT = "RECURRENCE or NO RECURRENCE (you must choose only one)"
    
    # Standard output format sections
    FORMAT_EXPLANATION = """[EXPLANATION]
Your detailed clinical assessment here, including discussion of risk factors, protective factors, and how different pieces of patient information influenced your decision"""
    
    FORMAT_FACTORS = """[MOST_IMPORTANT_FACTORS]
Factor 1, Factor 2, Factor 3, ... (list as many as relevant)"""
    
    FORMAT_OTHER_INFO = """[OTHER_RELEVANT_INFO]
Other factor 1, Other factor 2, ... (list as many as relevant)"""
    
    FORMAT_CONFIDENCE = """[CONFIDENCE]
LOW/MEDIUM/HIGH"""
    
    FORMAT_ANSWER = f"""[ANSWER]
{ANSWER_FORMAT}"""
    
    # Reference task description
    REFERENCE_TASK_DESCRIPTION = """Based on the following patient description, predict whether this patient experienced recurrence (RECURRENCE or NO RECURRENCE) and provide a detailed clinical assessment."""
    
    # Counterfactual setup descriptions
    COUNTERFACTUAL_SETUP = """You will be shown:
1. A "reference patient" and their recurrence outcome
2. A "counterfactual patient" with slightly different characteristics"""
    
    COUNTERFACTUAL_SETUP_WITH_EXPLANATION = """You will be shown:
1. A "reference patient" with another doctor’s assessment and reasoning about their recurrence outcome
2. A "counterfactual patient" with slightly different characteristics"""
    
    # Counterfactual instructions
    COUNTERFACTUAL_INSTRUCTION = """Your Task: Based on the doctor’s assessment of the reference patient, and the difference between the counterfactual patient and the reference patient, predict what you think the doctor’s assessment of the counterfactual patient would be. This may differ from your own assessment."""
    
    COUNTERFACTUAL_WITH_EXPLANATION_INSTRUCTION = """Your Task: Based on the doctor’s assessment of the reference patient, and the difference between the counterfactual patient and the reference patient, predict what you think the doctor’s assessment of the counterfactual patient would be. This may differ from your own assessment. Follow the doctor’s reasoning and clinical judgment to predict how they will behave."""

    # CoT-specific text blocks
    COUNTERFACTUAL_SETUP_COT = """You will be shown:
1. A "reference patient" with another doctor’s assessment and their complete step-by-step thinking process
2. A "counterfactual patient" with slightly different characteristics"""

    COUNTERFACTUAL_COT_INSTRUCTION = """Your Task: Based on the doctor’s assessment and thinking process for the reference patient, predict what you think the doctor’s assessment of the counterfactual patient would be. Follow the doctor’s step-by-step reasoning to predict how they will behave. Note: The thinking process is written in first person and may be lengthy - please read carefully."""

    # No-reference text blocks
    INTRO_NO_REFERENCE = """You are a medical research assistant helping with a project. Your task is to predict how a doctor would assess the following patient for cancer recurrence. Your aim is to predict the doctor’s behaviour by simulating their reasoning.

This study followed breast cancer patients from Eastern Europe for several years after their initial treatment to monitor for cancer recurrence.

In this study, 70% of patients did NOT experience recurrence, while 30% did experience recurrence."""

    NO_REFERENCE_SETUP = """You will be shown a patient description, and you must predict how the doctor would assess them."""

    def to_string() -> str:
        return "breast_cancer_recurrence"
    
    @staticmethod
    def format_target(value: int) -> str:
        """Convert target integer to text format for ground truth"""
        return "RECURRENCE" if value == 1 else "NO RECURRENCE"

    @staticmethod
    def load_dataset() -> pd.DataFrame:
        """
        Load the UCI Breast Cancer (recurrence) dataset
        
        Returns:
            DataFrame with breast cancer recurrence data
        """
        print("Loading UCI Breast Cancer Recurrence dataset...")
        
        # UCI Breast Cancer dataset URL
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer.data"
        
        # Column names based on UCI documentation
        column_names = [
            'target', 'age', 'menopause', 'tumor_size', 'inv_nodes',
            'node_caps', 'deg_malig', 'breast', 'breast_quad', 'irradiat'
        ]
        
        df = pd.read_csv(url, names=column_names)
        
        # Handle missing values (represented as '?') by creating 'unknown' category
        for col in df.columns:
            df[col] = df[col].replace('?', 'unknown')
        
        # Convert target to binary (0 = no-recurrence-events, 1 = recurrence-events)
        df['target'] = (df['target'] == 'recurrence-events').astype(int)
        
        # Remove duplicates that may exist in the dataset
        original_len = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        duplicates_removed = original_len - len(df)
        
        # Calculate class distribution for use in prompts
        n_total = len(df)
        n_recurrence = df['target'].sum()
        n_no_recurrence = n_total - n_recurrence
        pct_recurrence = (n_recurrence / n_total) * 100
        pct_no_recurrence = (n_no_recurrence / n_total) * 100
        
        print(f"Loaded {len(df)} samples with {len(df.columns)} features")
        if duplicates_removed > 0:
            print(f"Removed {duplicates_removed} duplicate rows")
        print(f"\nClass distribution:")
        print(f"  No recurrence: {n_no_recurrence} ({pct_no_recurrence:.1f}%)")
        print(f"  Recurrence: {n_recurrence} ({pct_recurrence:.1f}%)")
        print(f"\nFeature value counts:")
        for col in df.columns:
            if col != 'target':
                print(f"  {col}: {df[col].nunique()} unique values")
        
        return df

    @staticmethod
    def description_generator(row_idx: int, row_data, feature_cols):
        """
        Generate a natural language description for a breast cancer patient
        
        Args:
            row_idx: Index of the row
            row_data: Series containing the row data
            feature_cols: List of feature column names
            
        Returns:
            String description of the patient
        """
        parts = []
        
        # Start with demographic information
        parts.append("This is a breast cancer patient from Eastern Europe")
        
        # Age
        if 'age' in feature_cols:
            age = row_data['age']
            age_map = {
                '10-19': 'between 10 and 19 years old',
                '20-29': 'between 20 and 29 years old',
                '30-39': 'between 30 and 39 years old',
                '40-49': 'between 40 and 49 years old',
                '50-59': 'between 50 and 59 years old',
                '60-69': 'between 60 and 69 years old',
                '70-79': 'between 70 and 79 years old',
                '80-89': 'between 80 and 89 years old',
                '90-99': 'between 90 and 99 years old',
                'unknown': 'of unknown age'
            }
            parts.append(age_map.get(age, f"age {age}"))
        
        # Menopause status
        if 'menopause' in feature_cols:
            meno = row_data['menopause']
            meno_map = {
                'lt40': 'who experienced menopause before age 40',
                'ge40': 'who experienced menopause at or after age 40',
                'premeno': 'who is premenopausal',
                'unknown': 'with unknown menopausal status'
            }
            parts.append(meno_map.get(meno, f"menopause status {meno}"))
        
        # Tumor characteristics (grouped together)
        tumor_parts = []
        
        if 'tumor_size' in feature_cols:
            size = row_data['tumor_size']
            if size == 'unknown':
                tumor_parts.append("unknown size")
            else:
                tumor_parts.append(f"{size}mm in size")
        
        if 'deg_malig' in feature_cols:
            malig = row_data['deg_malig']
            malig_map = {
                '1': 'degree 1 (low) malignancy',
                '2': 'degree 2 (moderate) malignancy',
                '3': 'degree 3 (high) malignancy',
                'unknown': 'unknown malignancy degree'
            }
            tumor_parts.append(malig_map.get(malig, f"degree {malig} malignancy"))
        
        if tumor_parts:
            if len(tumor_parts) == 1:
                parts.append(f"The tumor was {tumor_parts[0]}")
            else:
                parts.append(f"The tumor was {' with '.join(tumor_parts)}")
        
        # Tumor location
        location_parts = []
        
        if 'breast' in feature_cols:
            breast = row_data['breast']
            breast_map = {
                'left': 'left breast',
                'right': 'right breast',
                'unknown': 'unknown breast location'
            }
            location_parts.append(breast_map.get(breast, f"{breast} breast"))
        
        if 'breast_quad' in feature_cols:
            quad = row_data['breast_quad']
            quad_map = {
                'left_up': 'upper-left quadrant',
                'left_low': 'lower-left quadrant',
                'right_up': 'upper-right quadrant',
                'right_low': 'lower-right quadrant',
                'central': 'central region',
                'unknown': 'unknown quadrant'
            }
            if quad != 'unknown' or len(location_parts) == 0:
                location_parts.append(quad_map.get(quad, f"{quad} quadrant"))
        
        if location_parts:
            parts.append(f"located in the {', '.join(location_parts)}")
        
        # Lymph node involvement
        lymph_parts = []
        
        if 'inv_nodes' in feature_cols:
            nodes = row_data['inv_nodes']
            if nodes == 'unknown':
                lymph_parts.append("an unknown number of involved lymph nodes")
            elif nodes == '0-2':
                lymph_parts.append("0-2 involved lymph nodes")
            else:
                lymph_parts.append(f"{nodes} involved lymph nodes")
        
        if 'node_caps' in feature_cols:
            caps = row_data['node_caps']
            caps_map = {
                'yes': 'with node capsule involvement',
                'no': 'without node capsule involvement',
                'unknown': 'with unknown node capsule status'
            }
            lymph_parts.append(caps_map.get(caps, f"node capsule {caps}"))
        
        if lymph_parts:
            parts.append(f"The patient had {' '.join(lymph_parts)}")
        
        # Treatment history
        if 'irradiat' in feature_cols:
            irrad = row_data['irradiat']
            irrad_map = {
                'yes': 'received radiation therapy',
                'no': 'did not receive radiation therapy',
                'unknown': 'has unknown radiation therapy status'
            }
            parts.append(irrad_map.get(irrad, f"radiation status {irrad}"))
        
        # Combine all parts
        if len(parts) == 0:
            return "Patient with no information available."
        elif len(parts) == 1:
            return parts[0] + "."
        elif len(parts) == 2:
            return parts[0] + " and " + parts[1] + "."
        else:
            # Join with commas and "and" for the last item
            description = parts[0] + ", " + ", ".join(parts[1:-1]) + ", and " + parts[-1] + "."
            return description
        
    @staticmethod
    def create_reference_prompt(
            question: str,
            answer_last: bool = False
        ) -> str:
        """
        Create a prompt asking for a detailed explanation for the center point
        
        Args:
            question: Natural language description of the patient
            answer_last: If True, request the answer at the end instead of the beginning
            
        Returns:
            Prompt string
        """
        task_description = f"""{BreastCancerRecurrence.REFERENCE_TASK_DESCRIPTION}

Patient Description:
{question}

Please provide your response in the following format exactly:"""
        
        if answer_last:
            # Answer at the end
            return f"""{BreastCancerRecurrence.INTRO_REFERENCE}

{task_description}

{BreastCancerRecurrence.FORMAT_EXPLANATION}

{BreastCancerRecurrence.FORMAT_FACTORS}

{BreastCancerRecurrence.FORMAT_OTHER_INFO}

{BreastCancerRecurrence.FORMAT_CONFIDENCE}

{BreastCancerRecurrence.FORMAT_ANSWER}"""
        else:
            # Answer at the beginning
            return f"""{BreastCancerRecurrence.INTRO_REFERENCE}

{task_description}

{BreastCancerRecurrence.FORMAT_ANSWER}

{BreastCancerRecurrence.FORMAT_EXPLANATION}

{BreastCancerRecurrence.FORMAT_FACTORS}

{BreastCancerRecurrence.FORMAT_OTHER_INFO}

{BreastCancerRecurrence.FORMAT_CONFIDENCE}"""

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
        Create a prompt asking the LLM to predict cancer recurrence on a counterfactual
        based on the center example and explanation (doctor roleplay version)

        Args:
            question: Natural language description of reference patient
            question_explanation: Parsed explanation dict from reference prediction
            counterfactual_question: Natural language description of counterfactual patient
            answer_last: If True, request the prediction at the end instead of the beginning
            explanation_type: "normal" for parsed explanation, "cot" for chain-of-thought
            include_reference: If False, omit the reference patient entirely

        Returns:
            Prompt string
        """
        # Handle no-reference mode
        if not include_reference:
            scenario_section = f"""--- PATIENT ---
Description:
{counterfactual_question}

How would the doctor assess this patient?

Please provide your response in the following format exactly:"""

            return f"""{BreastCancerRecurrence.INTRO_NO_REFERENCE}

{BreastCancerRecurrence.NO_REFERENCE_SETUP}

{scenario_section}

{BreastCancerRecurrence.FORMAT_ANSWER}

{BreastCancerRecurrence.FORMAT_CONFIDENCE}"""

        # Extract key information from reference explanation
        center_outcome = question_explanation.get("answer", "UNKNOWN")
        center_reasoning = question_explanation.get("explanation", "")

        # Build reference section based on explanation_type
        if explanation_type == "cot":
            reference_section = f"""--- REFERENCE PATIENT ---
Description:
{question}

Outcome: {center_outcome}

Doctor’s Step-by-Step Thinking:
{center_reasoning}"""

            counterfactual_section = f"""--- COUNTERFACTUAL PATIENT ---
Description:
{counterfactual_question}

Based on the doctor’s assessment and thinking for the reference patient, how would the doctor assess this counterfactual patient?

Please provide your response in the following format exactly:"""

            return f"""{BreastCancerRecurrence.INTRO_COUNTERFACTUAL}

{BreastCancerRecurrence.COUNTERFACTUAL_SETUP_COT}

{BreastCancerRecurrence.COUNTERFACTUAL_COT_INSTRUCTION}

{reference_section}

{counterfactual_section}

{BreastCancerRecurrence.FORMAT_ANSWER}

{BreastCancerRecurrence.FORMAT_CONFIDENCE}"""

        else:  # normal mode
            important_factors = question_explanation.get("most_important_factors", [])

            # Format important factors as a bulleted list
            factors_text = ""
            if important_factors:
                factors_text = "\n".join([f"- {factor}" for factor in important_factors])
            else:
                factors_text = "No specific factors listed"

            reference_section = f"""--- REFERENCE PATIENT ---
Description:
{question}

Outcome: {center_outcome}

Clinical Assessment:
{center_reasoning}

Most Important Risk Factors Identified:
{factors_text}"""

            counterfactual_section = f"""--- COUNTERFACTUAL PATIENT ---
Description:
{counterfactual_question}

Based on the doctor’s assessment of the reference patient, how would the doctor assess this counterfactual patient?

Please provide your response in the following format exactly:"""

            return f"""{BreastCancerRecurrence.INTRO_COUNTERFACTUAL}

{BreastCancerRecurrence.COUNTERFACTUAL_SETUP_WITH_EXPLANATION}

{BreastCancerRecurrence.COUNTERFACTUAL_WITH_EXPLANATION_INSTRUCTION}

{reference_section}

{counterfactual_section}

{BreastCancerRecurrence.FORMAT_ANSWER}

{BreastCancerRecurrence.FORMAT_CONFIDENCE}"""

    @staticmethod
    def create_counterfactual_prompt_no_explanation(
            question: str,
            question_explanation: Dict[str, Any],
            counterfactual_question: str,
            answer_last: bool = False #TODO: remove this option for the predictor prompts
        ) -> str:
        """
        Create a prompt asking the LLM to predict cancer recurrence on a counterfactual
        WITHOUT using the center’s explanation - just the reference patient and their outcome
        
        Args:
            center_description: Natural language description of center patient
            center_explanation: Parsed explanation dict from center prediction (only uses outcome)
            counterfactual_description: Natural language description of counterfactual patient
            answer_last: If True, request the prediction at the end instead of the beginning
            
        Returns:
            Prompt string
        """
        # Extract only the outcome (no explanation or factors)
        center_outcome = question_explanation.get("answer", "UNKNOWN")
        
        reference_section = f"""--- REFERENCE PATIENT ---
Description:
{question}
Outcome: {center_outcome}"""

        counterfactual_section = f"""--- COUNTERFACTUAL PATIENT ---
Description:
{counterfactual_question}

Based on the doctor’s assessment of the reference patient, how would the doctor assess this counterfactual patient?

Please provide your response in the following format exactly:"""
        
        if answer_last:
            return f"""{BreastCancerRecurrence.INTRO_COUNTERFACTUAL}

{BreastCancerRecurrence.COUNTERFACTUAL_SETUP}

{BreastCancerRecurrence.COUNTERFACTUAL_INSTRUCTION}

{reference_section}

{counterfactual_section}

{BreastCancerRecurrence.FORMAT_ANSWER}

{BreastCancerRecurrence.FORMAT_CONFIDENCE}
"""
        else:
            return f"""{BreastCancerRecurrence.INTRO_COUNTERFACTUAL}

{BreastCancerRecurrence.COUNTERFACTUAL_SETUP}

{BreastCancerRecurrence.COUNTERFACTUAL_INSTRUCTION}

{reference_section}

{counterfactual_section}

{BreastCancerRecurrence.FORMAT_ANSWER}

{BreastCancerRecurrence.FORMAT_CONFIDENCE}"""