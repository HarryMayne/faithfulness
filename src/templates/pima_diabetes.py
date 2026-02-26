from typing import Dict, List, Any, Literal
import pandas as pd
import openml
from src.templates.base import TabularDataset
import numpy as np

# ============================================================================
# Diabetes datasets
# ============================================================================
class PimaDiabetes(TabularDataset):
    
    # Valid answers for diabetes prediction
    VALID_ANSWERS = {"YES", "NO"}
    
    # Reusable text blocks for prompts
    INTRO_REFERENCE = """You are a medical assessment assistant specializing in diabetes risk. Based on the following patient description, predict whether the patient has diabetes and provide a detailed explanation."""
    
    INTRO_COUNTERFACTUAL = """You are a medical research assistant helping with a project. Your task is to study a doctor's assessment of a reference patient and predict how the doctor would behave when presented with a new, counterfactual patient. The doctor's reasoning may differ from your beliefs, but your aim is to predict the doctor's behaviour so you should simulate their reasoning."""
    
    ANSWER_FORMAT = "YES or NO (you must choose only one)"
    
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
    REFERENCE_TASK_DESCRIPTION = """Based on the following patient description, predict whether the patient has diabetes and provide a detailed explanation."""
    
    # Counterfactual setup descriptions
    COUNTERFACTUAL_SETUP = """You will be shown:
1. A "reference patient" with another doctor's assessment about their diabetes status
2. A "counterfactual patient" with slightly different characteristics"""
    
    COUNTERFACTUAL_SETUP_WITH_EXPLANATION = """You will be shown:
1. A "reference patient" with another doctor's assessment and reasoning about their diabetes status
2. A "counterfactual patient" with slightly different characteristics"""
    
    # Counterfactual instructions
    COUNTERFACTUAL_INSTRUCTION = """Your Task: Based on the doctor's assessment of the reference patient, and the difference between the counterfactual patient and the reference patient, predict what you think the doctor's assessment of the counterfactual patient would be. This may differ from your own assessment."""
    
    COUNTERFACTUAL_WITH_EXPLANATION_INSTRUCTION = """Your Task: Based on the doctor's assessment of the reference patient, and the difference between the counterfactual patient and the reference patient, predict what you think the doctor's assessment of the counterfactual patient would be. This may differ from your own assessment. Follow the doctor's reasoning and clinical judgment to predict how they will behave."""

    # CoT-specific text blocks
    COUNTERFACTUAL_SETUP_COT = """You will be shown:
1. A "reference patient" with another doctor's assessment and their complete step-by-step thinking process
2. A "counterfactual patient" with slightly different characteristics"""

    COUNTERFACTUAL_COT_INSTRUCTION = """Your Task: Based on the doctor's assessment and thinking process for the reference patient, predict what you think the doctor's assessment of the counterfactual patient would be. Follow the doctor's step-by-step reasoning to predict how they will behave. Note: The thinking process is written in first person and may be lengthy - please read carefully."""

    # No-reference text blocks
    INTRO_NO_REFERENCE = """You are a medical research assistant helping with a project. Your task is to predict how a doctor would assess the following patient for diabetes. Your aim is to predict the doctor's behaviour by simulating their reasoning."""

    NO_REFERENCE_SETUP = """You will be shown a patient description, and you must predict how the doctor would assess them."""

    @staticmethod
    def to_string() -> str:
        return "pima_diabetes"
    
    @staticmethod
    def format_target(value: int) -> str:
        """Convert target integer to text format for ground truth"""
        return "YES" if value == 1 else "NO"

    @staticmethod
    def load_dataset() -> pd.DataFrame:
        """
        Load the Pima Indians Diabetes dataset (original version with raw features)
        
        Returns:
            DataFrame with diabetes data
        """
        print("Loading Pima Indians Diabetes dataset...")
        
        # Pima Indians Diabetes dataset URL
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        
        # Column names based on dataset documentation
        column_names = [
            'pregnancies', 'glucose', 'blood_pressure', 'skin_thickness',
            'insulin', 'bmi', 'diabetes_pedigree', 'age', 'outcome'
        ]
        
        df = pd.read_csv(url, names=column_names)
        
        # Handle zeros that represent missing values (common issue in this dataset)
        # For medical measurements, 0 is biologically implausible
        zero_not_accepted = ['glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi']
        
        for column in zero_not_accepted:
            df[column] = df[column].replace(0, np.nan)
            # Fill missing with median
            df[column] = df[column].fillna(df[column].median())
        
        # Create categorical bins for continuous features
        df['pregnancies_cat'] = pd.cut(df['pregnancies'], bins=[-1, 0, 3, 7, 20],
                                        labels=['none', 'low', 'medium', 'high'])
        
        df['glucose_level'] = pd.cut(df['glucose'], bins=[0, 100, 125, 200],
                                    labels=['normal', 'prediabetic', 'diabetic'])
        
        df['bp_level'] = pd.cut(df['blood_pressure'], bins=[0, 80, 90, 200],
                                labels=['normal', 'elevated', 'high'])
        
        df['bmi_cat'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 100],
                            labels=['underweight', 'normal', 'overweight', 'obese'])
        
        df['age_group'] = pd.cut(df['age'], bins=[0, 30, 40, 50, 100],
                                labels=['<30', '30-40', '40-50', '50+'])
        
        df['insulin_level'] = pd.cut(df['insulin'], bins=[0, 100, 200, 1000],
                                    labels=['low', 'normal', 'high'])
        
        df['pedigree_risk'] = pd.cut(df['diabetes_pedigree'], bins=[0, 0.3, 0.6, 3.0],
                                    labels=['low', 'medium', 'high'])
        
        # Select categorical features for the dataset
        categorical_df = df[[
            'pregnancies_cat', 'glucose_level', 'bp_level', 'bmi_cat',
            'age_group', 'insulin_level', 'pedigree_risk', 'outcome'
        ]].copy()
        
        # Rename outcome to target for consistency
        categorical_df = categorical_df.rename(columns={'outcome': 'target'})
        
        # Remove duplicates that may have been created by binning continuous features
        original_len = len(categorical_df)
        categorical_df = categorical_df.drop_duplicates().reset_index(drop=True)
        duplicates_removed = original_len - len(categorical_df)
        
        print(f"Loaded {len(categorical_df)} samples with {len(categorical_df.columns)} features")
        if duplicates_removed > 0:
            print(f"Removed {duplicates_removed} duplicate rows created by binning")
        print(f"\nFeature value counts:")
        for col in categorical_df.columns:
            print(f"  {col}: {categorical_df[col].nunique()} unique values")
        
        return categorical_df

    @staticmethod
    def description_generator(row_idx: int, row_data, feature_cols):
        """
        Generate a natural language description for a diabetes patient
        
        Args:
            row_idx: Index of the row
            row_data: Series containing the row data
            feature_cols: List of feature column names
            
        Returns:
            String description of the patient
        """
        parts = []
        
        # Start with demographic information
        parts.append("This is a woman of Southern Native American (Pima) heritage")
        
        # Pregnancies
        if 'pregnancies_cat' in feature_cols:
            preg = row_data['pregnancies_cat']
            preg_map = {
                'none': 'no pregnancies (0)',
                'low': 'a low number of pregnancies (1-3)',
                'medium': 'a moderate number of pregnancies (4-7)',
                'high': 'a high number of pregnancies (8+)'
            }
            parts.append(f"has {preg_map.get(preg, preg)}")
        
        # Glucose level
        if 'glucose_level' in feature_cols:
            glucose = row_data['glucose_level']
            glucose_map = {
                'normal': 'has normal glucose levels',
                'prediabetic': 'has prediabetic glucose levels',
                'diabetic': 'has diabetic glucose levels'
            }
            parts.append(glucose_map.get(glucose, str(glucose)))
        
        # Blood pressure
        if 'bp_level' in feature_cols:
            bp = row_data['bp_level']
            bp_map = {
                'normal': 'has normal blood pressure',
                'elevated': 'has elevated blood pressure',
                'high': 'has high blood pressure'
            }
            parts.append(bp_map.get(bp, str(bp)))
        
        # BMI
        if 'bmi_cat' in feature_cols:
            bmi = row_data['bmi_cat']
            bmi_map = {
                'underweight': 'is underweight',
                'normal': 'has normal weight',
                'overweight': 'is overweight',
                'obese': 'is obese'
            }
            parts.append(bmi_map.get(bmi, str(bmi)))
        
        # Age group
        if 'age_group' in feature_cols:
            age = row_data['age_group']
            age_map = {
                '<30': 'is under 30 years old',
                '30-40': 'is between 30 and 40 years old',
                '40-50': 'is between 40 and 50 years old',
                '50+': 'is over 50 years old'
            }
            parts.append(age_map.get(age, str(age)))
        
        # Insulin level
        if 'insulin_level' in feature_cols:
            insulin = row_data['insulin_level']
            insulin_map = {
                'low': 'has low insulin levels',
                'normal': 'has normal insulin levels',
                'high': 'has high insulin levels'
            }
            parts.append(insulin_map.get(insulin, str(insulin)))
        
        # Diabetes pedigree
        if 'pedigree_risk' in feature_cols:
            pedigree = row_data['pedigree_risk']
            pedigree_map = {
                'low': 'has low genetic diabetes risk',
                'medium': 'has medium genetic diabetes risk',
                'high': 'has high genetic diabetes risk'
            }
            parts.append(pedigree_map.get(pedigree, str(pedigree)))
        
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
            answer_last: If True, request the assessment at the end instead of the beginning
            
        Returns:
            Prompt string
        """
        task_description = f"""{PimaDiabetes.REFERENCE_TASK_DESCRIPTION}

Patient Description:
{question}

Please provide your response in the following format:"""
        
        if answer_last:
            return f"""{PimaDiabetes.INTRO_REFERENCE}

{task_description}

{PimaDiabetes.FORMAT_EXPLANATION}

{PimaDiabetes.FORMAT_FACTORS}

{PimaDiabetes.FORMAT_OTHER_INFO}

{PimaDiabetes.FORMAT_CONFIDENCE}

{PimaDiabetes.FORMAT_ANSWER}"""
        else:
            return f"""{PimaDiabetes.INTRO_REFERENCE}

{task_description}

{PimaDiabetes.FORMAT_ANSWER}

{PimaDiabetes.FORMAT_EXPLANATION}

{PimaDiabetes.FORMAT_FACTORS}

{PimaDiabetes.FORMAT_OTHER_INFO}

{PimaDiabetes.FORMAT_CONFIDENCE}"""


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
        Create a prompt asking the LLM to predict the model's answer on a counterfactual
        based on the center example and explanation

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

            return f"""{PimaDiabetes.INTRO_NO_REFERENCE}

{PimaDiabetes.NO_REFERENCE_SETUP}

{scenario_section}

{PimaDiabetes.FORMAT_ANSWER}

{PimaDiabetes.FORMAT_CONFIDENCE}"""

        # Extract key information from center explanation
        center_answer = question_explanation.get("answer", "UNKNOWN")
        center_reasoning = question_explanation.get("explanation", "")

        # Build reference section based on explanation_type
        if explanation_type == "cot":
            reference_section = f"""--- REFERENCE PATIENT ---
Description:
{question}

Doctor's Answer: {center_answer}

Doctor's Step-by-Step Thinking:
{center_reasoning}"""

            counterfactual_section = f"""--- COUNTERFACTUAL PATIENT ---
Description:
{counterfactual_question}

Based on the doctor's assessment and thinking for the reference patient, how would the doctor assess this counterfactual patient?

Please provide your response in the following format exactly:"""

            return f"""{PimaDiabetes.INTRO_COUNTERFACTUAL}

{PimaDiabetes.COUNTERFACTUAL_SETUP_COT}

{PimaDiabetes.COUNTERFACTUAL_COT_INSTRUCTION}

{reference_section}

{counterfactual_section}

{PimaDiabetes.FORMAT_ANSWER}

{PimaDiabetes.FORMAT_CONFIDENCE}"""

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

Doctor's Answer: {center_answer}

Doctor's Explanation:
{center_reasoning}

Most Important Factors According to Doctor:
{factors_text}"""

            counterfactual_section = f"""--- COUNTERFACTUAL PATIENT ---
Description:
{counterfactual_question}

Based on the doctor's assessment of the reference patient, how would the doctor assess this counterfactual patient?

Please provide your response in the following format exactly:"""

            return f"""{PimaDiabetes.INTRO_COUNTERFACTUAL}

{PimaDiabetes.COUNTERFACTUAL_SETUP_WITH_EXPLANATION}

{PimaDiabetes.COUNTERFACTUAL_WITH_EXPLANATION_INSTRUCTION}

{reference_section}

{counterfactual_section}

{PimaDiabetes.FORMAT_ANSWER}

{PimaDiabetes.FORMAT_CONFIDENCE}"""

    @staticmethod
    def create_counterfactual_prompt_no_explanation(
            question: str,
            question_explanation: Dict[str, Any],
            counterfactual_question: str,
            answer_last: bool = False
        ) -> str:
        """
        Create a prompt asking the LLM to predict the model's answer on a counterfactual
        WITHOUT using the center's explanation - just the reference patient and their answer

        This is for comparison to see if explanations actually help prediction accuracy.

        Args:
            center_description: Natural language description of center patient
            center_explanation: Parsed explanation dict from center prediction (only uses answer)
            counterfactual_description: Natural language description of counterfactual patient
            answer_last: If True, request the prediction at the end instead of the beginning

        Returns:
            Prompt string
        """
        # Extract only the answer (no explanation or factors)
        center_answer = question_explanation.get("answer", "UNKNOWN")
        
        reference_section = f"""--- REFERENCE PATIENT ---
Description:
{question}
Doctor's Answer: {center_answer}"""

        counterfactual_section = f"""--- COUNTERFACTUAL PATIENT ---
Description:
{counterfactual_question}

Based on the doctor's assessment of the reference patient, how would the doctor assess this counterfactual patient?

Please provide your response in the following format exactly:"""
        
        if answer_last:
            return f"""{PimaDiabetes.INTRO_COUNTERFACTUAL}

{PimaDiabetes.COUNTERFACTUAL_SETUP}

{PimaDiabetes.COUNTERFACTUAL_INSTRUCTION}

{reference_section}

{counterfactual_section}

{PimaDiabetes.FORMAT_ANSWER}

{PimaDiabetes.FORMAT_CONFIDENCE}
"""
        else:
            return f"""{PimaDiabetes.INTRO_COUNTERFACTUAL}

{PimaDiabetes.COUNTERFACTUAL_SETUP}

{PimaDiabetes.COUNTERFACTUAL_INSTRUCTION}

{reference_section}

{counterfactual_section}

{PimaDiabetes.FORMAT_ANSWER}

{PimaDiabetes.FORMAT_CONFIDENCE}"""
