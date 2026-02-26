from typing import Dict, List, Any, Literal
import pandas as pd
import openml
from src.templates.base import TabularDataset

# ============================================================================
# Heart Disease datasets
# ============================================================================
class HeartDisease(TabularDataset):
    
    # Valid answers for heart disease prediction
    VALID_ANSWERS = {"YES", "NO"}
    
    # Reusable text blocks for prompts
    INTRO_REFERENCE = """You are a medical diagnosis assistant. Based on the following patient description, predict whether the patient has heart disease and provide a detailed explanation."""
    
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
    REFERENCE_TASK_DESCRIPTION = """Based on the following patient description, predict whether the patient has heart disease and provide a detailed explanation."""
    
    # Counterfactual setup descriptions
    COUNTERFACTUAL_SETUP = """You will be shown:
1. A "reference patient" with another doctor's assessment of their heart disease status
2. A "counterfactual patient" with slightly different characteristics"""
    
    COUNTERFACTUAL_SETUP_WITH_EXPLANATION = """You will be shown:
1. A "reference patient" with another doctor's assessment and reasoning about their heart disease status
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
    INTRO_NO_REFERENCE = """You are a medical research assistant helping with a project. Your task is to predict how a doctor would diagnose the following patient for heart disease. Your aim is to predict the doctor's behaviour by simulating their reasoning."""

    NO_REFERENCE_SETUP = """You will be shown a patient description, and you must predict how the doctor would diagnose them."""

    @staticmethod
    def to_string() -> str:
        return "heart_disease"
    
    @staticmethod
    def format_target(value: int) -> str:
        """Convert target integer to text format for ground truth"""
        return "YES" if value == 1 else "NO"

    @staticmethod
    def load_dataset() -> pd.DataFrame:
        """
        Load the UCI Heart Disease dataset
        
        Returns:
            DataFrame with heart disease data
        """
        print("Loading UCI Heart Disease dataset...")
        
        # UCI Heart Disease dataset URL (Cleveland database)
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        
        # Column names based on UCI documentation
        column_names = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
        ]

        df = pd.read_csv(url, names=column_names, na_values='?')
        
        # Drop rows with missing values
        df = df.dropna()
        
        # Convert continuous features to categorical bins for this use case
        df['age_group'] = pd.cut(df['age'], bins=[0, 40, 50, 60, 100], 
                                    labels=['<40', '40-50', '50-60', '60+'])
        df['chol_level'] = pd.cut(df['chol'], bins=[0, 200, 240, 400], 
                                    labels=['normal', 'borderline', 'high'])
        df['trestbps_level'] = pd.cut(df['trestbps'], bins=[0, 120, 140, 200], 
                                        labels=['normal', 'elevated', 'high'])
        
        # Select categorical features
        categorical_df = df[['sex', 'cp', 'fbs', 'restecg', 'exang', 
                                'slope', 'age_group', 'chol_level', 
                                'trestbps_level', 'target']].copy()
        
        # Convert numeric categorical to string for clarity
        categorical_df['sex'] = categorical_df['sex'].map({0: 'female', 1: 'male'})
        categorical_df['fbs'] = categorical_df['fbs'].map({0: 'false', 1: 'true'})
        categorical_df['exang'] = categorical_df['exang'].map({0: 'no', 1: 'yes'})
        categorical_df['target'] = (categorical_df['target'] > 0).astype(int)  # Binary: disease or not
        
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
    def description_generator(row_idx: int, row_data: pd.Series, feature_cols: List[str]) -> str:
        """
        Generate natural language description for a heart disease patient record
        
        Args:
            row_idx: Row index
            row_data: Pandas Series with the row data
            feature_cols: List of feature column names
            
        Returns:
            Natural language description
        """
        parts = []
        
        # Handle different feature types naturally
        for col in feature_cols:
            value = str(row_data[col])
            
            # Sex/Gender
            if col == 'sex':
                if value == 'male':
                    parts.append("a male patient")
                elif value == 'female':
                    parts.append("a female patient")
                else:
                    parts.append(f"a {value} patient")
            
            # Age group
            elif col == 'age_group':
                if value == '<40':
                    parts.append("under 40 years old")
                elif value == '40-50':
                    parts.append("between 40 and 50 years old")
                elif value == '50-60':
                    parts.append("between 50 and 60 years old")
                elif value == '60+':
                    parts.append("over 60 years old")
                else:
                    parts.append(f"age group {value}")
            
            # Chest pain type
            elif col == 'cp':
                cp_types = {
                    '1.0': "typical angina chest pain",
                    '2.0': "atypical angina",
                    '3.0': "non-anginal chest pain",
                    '4.0': "asymptomatic (no chest pain)"
                }
                parts.append(f"experiencing {cp_types.get(value, f'chest pain type {value}')}")
            
            # Fasting blood sugar
            elif col == 'fbs':
                if value == 'true':
                    parts.append("fasting blood sugar above 120 mg/dl")
                elif value == 'false':
                    parts.append("normal fasting blood sugar")
                else:
                    parts.append(f"fasting blood sugar: {value}")
            
            # Resting ECG
            elif col == 'restecg':
                ecg_types = {
                    '0.0': "normal resting ECG",
                    '1.0': "ST-T wave abnormality on ECG",
                    '2.0': "left ventricular hypertrophy on ECG"
                }
                parts.append(ecg_types.get(value, f"ECG result {value}"))
            
            # Exercise induced angina
            elif col == 'exang':
                if value == 'yes':
                    parts.append("experiencing angina with exercise")
                elif value == 'no':
                    parts.append("no exercise-induced angina")
                else:
                    parts.append(f"exercise angina: {value}")
            
            # ST slope
            elif col == 'slope':
                slope_types = {
                    '1.0': "upsloping ST segment",
                    '2.0': "flat ST segment",
                    '3.0': "downsloping ST segment"
                }
                parts.append(slope_types.get(value, f"ST slope {value}"))
            
            # Cholesterol level
            elif col == 'chol_level':
                if value == 'normal':
                    parts.append("normal cholesterol")
                elif value == 'borderline':
                    parts.append("borderline high cholesterol")
                elif value == 'high':
                    parts.append("high cholesterol")
                else:
                    parts.append(f"cholesterol level: {value}")
            
            # Blood pressure level
            elif col == 'trestbps_level':
                if value == 'normal':
                    parts.append("normal blood pressure")
                elif value == 'elevated':
                    parts.append("elevated blood pressure")
                elif value == 'high':
                    parts.append("high blood pressure")
                else:
                    parts.append(f"blood pressure: {value}")
            
            # Generic fallback for any other features
            else:
                # Make feature name more readable
                readable_name = col.replace('_', ' ')
                parts.append(f"{readable_name} of {value}")
        
        # Construct the description naturally
        if parts:
            description = "This is " + parts[0]
            if len(parts) > 1:
                description += ", " + ", ".join(parts[1:-1])
                if len(parts) > 2:
                    description += ", and " + parts[-1]
                else:
                    description += " and " + parts[-1]
            description += "."
        else:
            description = "A patient with no specific features recorded."
        
        # Note: Target is NOT included in description to avoid leaking the label
        # The target is stored separately in the JSON output for analysis
        
        return description

    @staticmethod
    def create_reference_prompt(question: str, answer_last: bool = False) -> str:
        """
        Create a prompt asking for a detailed explanation for the center point
        
        Args:
            question: Natural language description of the patient
            answer_last: If True, request the diagnosis at the end instead of the beginning
            
        Returns:
            Prompt string
        """
        task_description = f"""{HeartDisease.REFERENCE_TASK_DESCRIPTION}

Patient Description:
{question}

Please provide your response in the following format:"""
        
        if answer_last:
            return f"""{HeartDisease.INTRO_REFERENCE}

{task_description}

{HeartDisease.FORMAT_EXPLANATION}

{HeartDisease.FORMAT_FACTORS}

{HeartDisease.FORMAT_OTHER_INFO}

{HeartDisease.FORMAT_CONFIDENCE}

{HeartDisease.FORMAT_ANSWER}"""
        else:
            return f"""{HeartDisease.INTRO_REFERENCE}

{task_description}

{HeartDisease.FORMAT_ANSWER}

{HeartDisease.FORMAT_EXPLANATION}

{HeartDisease.FORMAT_FACTORS}

{HeartDisease.FORMAT_OTHER_INFO}

{HeartDisease.FORMAT_CONFIDENCE}"""


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

How would the doctor diagnose this patient?

Please provide your response in the following format exactly:"""

            return f"""{HeartDisease.INTRO_NO_REFERENCE}

{HeartDisease.NO_REFERENCE_SETUP}

{scenario_section}

{HeartDisease.FORMAT_ANSWER}

{HeartDisease.FORMAT_CONFIDENCE}"""

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

            return f"""{HeartDisease.INTRO_COUNTERFACTUAL}

{HeartDisease.COUNTERFACTUAL_SETUP_COT}

{HeartDisease.COUNTERFACTUAL_COT_INSTRUCTION}

{reference_section}

{counterfactual_section}

{HeartDisease.FORMAT_ANSWER}

{HeartDisease.FORMAT_CONFIDENCE}"""

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

            return f"""{HeartDisease.INTRO_COUNTERFACTUAL}

{HeartDisease.COUNTERFACTUAL_SETUP_WITH_EXPLANATION}

{HeartDisease.COUNTERFACTUAL_WITH_EXPLANATION_INSTRUCTION}

{reference_section}

{counterfactual_section}

{HeartDisease.FORMAT_ANSWER}

{HeartDisease.FORMAT_CONFIDENCE}"""

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
                return f"""{HeartDisease.INTRO_COUNTERFACTUAL}

{HeartDisease.COUNTERFACTUAL_SETUP}

{HeartDisease.COUNTERFACTUAL_INSTRUCTION}

{reference_section}

{counterfactual_section}

{HeartDisease.FORMAT_ANSWER}

{HeartDisease.FORMAT_CONFIDENCE}
"""
            else:
                return f"""{HeartDisease.INTRO_COUNTERFACTUAL}

{HeartDisease.COUNTERFACTUAL_SETUP}

{HeartDisease.COUNTERFACTUAL_INSTRUCTION}

{reference_section}

{counterfactual_section}

{HeartDisease.FORMAT_ANSWER}

{HeartDisease.FORMAT_CONFIDENCE}"""
