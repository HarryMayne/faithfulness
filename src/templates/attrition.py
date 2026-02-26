from typing import Dict, List, Any, Literal
import pandas as pd
from src.templates.base import TabularDataset

# ============================================================================
# Employee Attrition Prediction Dataset (IBM HR Analytics)
# ============================================================================
class AttritionDataset(TabularDataset):
    
    # Valid answers for attrition prediction
    VALID_ANSWERS = {"YES", "NO"}
    
    # ========================================================================
    # Reusable Text Blocks
    # ========================================================================
    
    # Study introduction
    INTRO_REFERENCE = """You are analyzing employee data from a company's HR department. Your task is to predict employee attrition risk. The goal is to determine if an employee is likely to leave the company soon."""
    
    INTRO_COUNTERFACTUAL = """You are a research assistant helping with a project. Your task is to study an HR analyst's assessment of a reference employee and predict how the analyst would behave when presented with a new, counterfactual employee. The analyst's reasoning may differ from your beliefs, but your aim is to predict the analyst's behaviour so you should simulate their reasoning.

This analysis uses employee data from a company's HR department. The goal is to determine if an employee is likely to leave the company soon."""
    
    # Answer format instructions
    ANSWER_FORMAT = "YES or NO (you must choose only one)"
    
    # Standard output format sections
    FORMAT_EXPLANATION = """[EXPLANATION]
Your detailed assessment here, including discussion of risk factors and how different pieces of information influenced your decision"""
    
    FORMAT_FACTORS = """[MOST_IMPORTANT_FACTORS]
Factor 1, Factor 2, Factor 3, ... (list as many as relevant)"""
    
    FORMAT_OTHER_INFO = """[OTHER_RELEVANT_INFO]
Other factor 1, Other factor 2, ... (list as many as relevant)"""
    
    FORMAT_CONFIDENCE = """[CONFIDENCE]
LOW/MEDIUM/HIGH"""
    
    FORMAT_ANSWER = f"""[ANSWER]
{ANSWER_FORMAT}"""
    
    # Reference task description
    REFERENCE_TASK_DESCRIPTION = """Based on the following employee's profile, predict whether they are likely to leave the company soon (YES or NO) and provide a detailed assessment."""
    
    # Counterfactual setup descriptions
    COUNTERFACTUAL_SETUP = """You will be shown:
1. A "reference employee" and their attrition prediction (YES for likely to leave, NO for likely to stay)
2. A "counterfactual employee" with slightly different characteristics"""
    
    COUNTERFACTUAL_SETUP_WITH_EXPLANATION = """You will be shown:
1. A "reference employee" with an assessment and reasoning about their attrition risk
2. A "counterfactual employee" with slightly different characteristics"""
    
    # Counterfactual instructions
    COUNTERFACTUAL_INSTRUCTION = """Your Task: Based on the analyst's assessment of the reference employee, and the difference between the counterfactual employee and the reference employee, predict what you think the analyst's assessment of the counterfactual employee would be. This may differ from your own assessment."""
    
    COUNTERFACTUAL_WITH_EXPLANATION_INSTRUCTION = """Your Task: Based on the analyst's assessment of the reference employee, and the difference between the counterfactual employee and the reference employee, predict what you think the analyst's assessment of the counterfactual employee would be. This may differ from your own assessment. Follow the analyst's reasoning and judgment to predict how they will behave."""

    # CoT-specific text blocks
    COUNTERFACTUAL_SETUP_COT = """You will be shown:
1. A "reference employee" with an analyst's assessment and their complete step-by-step thinking process
2. A "counterfactual employee" with slightly different characteristics"""

    COUNTERFACTUAL_COT_INSTRUCTION = """Your Task: Based on the analyst's assessment and thinking process for the reference employee, predict what you think the analyst's assessment of the counterfactual employee would be. Follow the analyst's step-by-step reasoning to predict how they will behave. Note: The thinking process is written in first person and may be lengthy - please read carefully."""

    # No-reference text blocks
    INTRO_NO_REFERENCE = """You are a research assistant helping with a project. Your task is to predict how an HR analyst would assess the following employee's attrition risk. Your aim is to predict the analyst's behaviour by simulating their reasoning.

This analysis uses employee data from a company's HR department. The goal is to determine if an employee is likely to leave the company soon."""

    NO_REFERENCE_SETUP = """You will be shown an employee's profile, and you must predict how the analyst would assess them."""

    @staticmethod
    def to_string() -> str:
        return "attrition"
    
    @staticmethod
    def format_target(value) -> str:
        """Convert target integer to text format for ground truth"""
        return "YES" if value == 1 else "NO"

    @staticmethod
    def load_dataset() -> pd.DataFrame:
        """
        Load the IBM HR Analytics Employee Attrition dataset
        
        Returns:
            DataFrame with employee attrition data
        """
        print("Loading IBM HR Attrition dataset...")
        
        # GitHub raw URL for the dataset
        url = "https://raw.githubusercontent.com/nibeditans/Employee-Attrition-Analysis-On-IBM-HR-Data/main/IBM%20HR%20Employee%20Attrition%20Data.csv"
        
        df = pd.read_csv(url)
        
        # Convert Attrition to binary target (1 = Yes/Left, 0 = No/Stayed)
        df['target'] = (df['Attrition'] == 'Yes').astype(int)
        
        # --------------------------------------------------------
        # Feature Engineering & Binning (Applied BEFORE deduplication)
        # --------------------------------------------------------
        
        # 1. Bin 'Age' into 4 categories
        def bin_age(age):
            if age <= 30: return '18-30'
            if age <= 40: return '31-40'
            if age <= 50: return '41-50'
            return '51+'
        df['Age'] = df['Age'].apply(bin_age)
        
        # 2. Bin 'Education' into 3 categories
        # Original: 1=Below College, 2=College, 3=Bachelor, 4=Master, 5=Doctor
        def bin_education(edu):
            if edu <= 2: return 'College or below'
            if edu == 3: return 'Bachelor'
            return 'Post-graduate'
        df['Education'] = df['Education'].apply(bin_education)
        
        # 3. Map JobLevel to descriptive labels
        job_level_map = {1: 'Entry', 2: 'Junior', 3: 'Mid-level', 4: 'Senior', 5: 'Executive'}
        df['JobLevel'] = df['JobLevel'].map(job_level_map)
        
        # 6. Bin MonthlyIncome into 4 categories
        def bin_income(income):
            if income < 3000: return 'Low (<$3k)'
            if income < 6000: return 'Medium ($3k-$6k)'
            if income < 10000: return 'High ($6k-$10k)'
            return 'Very High (>$10k)'
        df['MonthlyIncome'] = df['MonthlyIncome'].apply(bin_income)
        
        # 7. Bin DistanceFromHome into 3 categories
        def bin_distance(dist):
            if dist <= 9: return 'Near (1-9 miles)'
            if dist <= 20: return 'Moderate (10-20 miles)'
            return 'Far (21+ miles)'
        df['DistanceFromHome'] = df['DistanceFromHome'].apply(bin_distance)
        
        # 5. Bin YearsAtCompany into 4 categories
        def bin_company_years(years):
            if years <= 2: return '0-2 years (New)'
            if years <= 5: return '3-5 years (Established)'
            if years <= 10: return '6-10 years (Veteran)'
            return '11+ years (Tenured)'
        df['YearsAtCompany'] = df['YearsAtCompany'].apply(bin_company_years)
        
        # 6. Clean up BusinessTravel values for readability
        travel_map = {
            'Non-Travel': 'No travel',
            'Travel_Rarely': 'Travels rarely',
            'Travel_Frequently': 'Travels frequently'
        }
        df['BusinessTravel'] = df['BusinessTravel'].map(travel_map)
        
        # --------------------------------------------------------
        # Select only the features we want to keep
        # --------------------------------------------------------
        keep_columns = [
            'Age', 'Education', 'Gender', 'MaritalStatus', 'Department',
            'JobLevel', 'MonthlyIncome', 'OverTime', 'DistanceFromHome',
            'YearsAtCompany', 'BusinessTravel',
            'target'
        ]
        df = df[keep_columns]
        
        # --------------------------------------------------------
        # Remove duplicates AFTER feature engineering
        # --------------------------------------------------------
        original_len = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        duplicates_removed = original_len - len(df)
        
        # Calculate class distribution
        n_total = len(df)
        n_left = df['target'].sum()
        n_stayed = n_total - n_left
        pct_left = (n_left / n_total) * 100
        pct_stayed = (n_stayed / n_total) * 100
        
        print(f"Loaded {len(df)} samples with {len(df.columns)} features")
        if duplicates_removed > 0:
            print(f"Removed {duplicates_removed} duplicate rows (post-binning)")
            
        print(f"\nClass distribution:")
        print(f"  NO (Stayed): {n_stayed} ({pct_stayed:.1f}%)")
        print(f"  YES (Left): {n_left} ({pct_left:.1f}%)")
        
        return df

    @staticmethod
    def description_generator(row_idx: int, row_data, feature_cols):
        """
        Generate a natural language description for an employee
        
        Args:
            row_idx: Index of the row
            row_data: Series containing the row data
            feature_cols: List of feature column names
            
        Returns:
            String description of the employee
        """
        # Get basic info (11 features)
        age_bin = row_data.get('Age', 'unknown')
        gender = row_data.get('Gender', 'person')
        marital = row_data.get('MaritalStatus', 'unknown')
        education = row_data.get('Education', 'unknown')
        department = row_data.get('Department', 'unknown')
        job_level = row_data.get('JobLevel', 'unknown')
        income = row_data.get('MonthlyIncome', 'unknown')
        company_years = row_data.get('YearsAtCompany', 'unknown')
        overtime = row_data.get('OverTime', 'unknown')
        travel = row_data.get('BusinessTravel', 'unknown')
        distance = row_data.get('DistanceFromHome', 'unknown')
        
        # Set pronouns
        if gender == 'Male':
            pronoun = "He"
        elif gender == 'Female':
            pronoun = "She"
        else:
            pronoun = "They"
        
        # Build sentence 1: Demographics and marital status
        marital_desc = ""
        if marital == 'Single':
            marital_desc = ", single"
        elif marital == 'Married':
            marital_desc = ", married"
        elif marital == 'Divorced':
            marital_desc = ", divorced"
        
        sentence1 = f"This is a {gender.lower()} employee aged {age_bin}{marital_desc}"
        
        # Add education
        if education != 'unknown':
            sentence1 += f", with a {education.lower()} level of education"
        
        sentence1 += "."
        
        # Build sentence 2: Work position and compensation
        work_parts = []
        if department != 'unknown':
            work_parts.append(f"works in the {department} department")
        if job_level != 'unknown':
            article = "an" if job_level.lower() == "entry" else "a"
            work_parts.append(f"holds {article} {job_level.lower()} position")
        if income != 'unknown':
            work_parts.append(f"earns a {income.lower()} monthly salary")
        
        if work_parts:
            sentence2 = pronoun + " " + ", ".join(work_parts) + "."
        else:
            sentence2 = ""
        
        # Build sentence 3: Tenure at company
        if company_years != 'unknown':
            company_clean = company_years.lower().replace(' (', ' - ').replace(')', '')
            sentence3 = f"{pronoun} has been at this company for {company_clean}."
        else:
            sentence3 = ""
        
        # Build sentence 4: Work style (overtime, travel, commute)
        lifestyle_parts = []
        if overtime == 'Yes':
            lifestyle_parts.append("regularly works overtime")
        elif overtime == 'No':
            lifestyle_parts.append("does not work overtime")
        
        if travel != 'unknown':
            lifestyle_parts.append(travel.lower())
        
        if distance != 'unknown':
            dist_clean = distance.lower()
            lifestyle_parts.append(f"commutes a {dist_clean} distance")
        
        if lifestyle_parts:
            sentence4 = pronoun + " " + ", ".join(lifestyle_parts) + "."
        else:
            sentence4 = ""
        
        # Combine all non-empty sentences
        sentences = [s for s in [sentence1, sentence2, sentence3, sentence4] if s]
        description = " ".join(sentences)
        
        return description

    @staticmethod
    def create_reference_prompt(
            question: str,
            answer_last: bool = False
        ) -> str:
        """
        Create a prompt asking for a detailed explanation for the employee
        """
        task_description = f"""{AttritionDataset.REFERENCE_TASK_DESCRIPTION}

Employee Profile:
{question}

Please provide your response in the following format exactly:"""
        
        if answer_last:
            return f"""{AttritionDataset.INTRO_REFERENCE}

{task_description}

{AttritionDataset.FORMAT_EXPLANATION}

{AttritionDataset.FORMAT_FACTORS}

{AttritionDataset.FORMAT_OTHER_INFO}

{AttritionDataset.FORMAT_CONFIDENCE}

{AttritionDataset.FORMAT_ANSWER}"""
        else:
            return f"""{AttritionDataset.INTRO_REFERENCE}

{task_description}

{AttritionDataset.FORMAT_ANSWER}

{AttritionDataset.FORMAT_EXPLANATION}

{AttritionDataset.FORMAT_FACTORS}

{AttritionDataset.FORMAT_OTHER_INFO}

{AttritionDataset.FORMAT_CONFIDENCE}"""

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
        Create a prompt asking the LLM to predict attrition on a counterfactual
        based on the reference example and explanation

        Args:
            question: Natural language description of reference employee
            question_explanation: Parsed explanation dict from reference prediction
            counterfactual_question: Natural language description of counterfactual employee
            answer_last: If True, request the prediction at the end instead of the beginning
            explanation_type: "normal" for parsed explanation, "cot" for chain-of-thought
            include_reference: If False, omit the reference employee entirely

        Returns:
            Prompt string
        """
        # Handle no-reference mode
        if not include_reference:
            scenario_section = f"""--- EMPLOYEE ---
Profile:
{counterfactual_question}

How would the analyst assess this employee?

Please provide your response in the following format exactly:"""

            return f"""{AttritionDataset.INTRO_NO_REFERENCE}

{AttritionDataset.NO_REFERENCE_SETUP}

{scenario_section}

{AttritionDataset.FORMAT_ANSWER}

{AttritionDataset.FORMAT_CONFIDENCE}"""

        center_outcome = question_explanation.get("answer", "UNKNOWN")
        center_reasoning = question_explanation.get("explanation", "")

        # Build reference section based on explanation_type
        if explanation_type == "cot":
            reference_section = f"""--- REFERENCE EMPLOYEE ---
Profile:
{question}

Likely to Leave: {center_outcome}

Analyst's Step-by-Step Thinking:
{center_reasoning}"""

            counterfactual_section = f"""--- COUNTERFACTUAL EMPLOYEE ---
Profile:
{counterfactual_question}

Based on the analyst's assessment and thinking for the reference employee, how would the analyst assess this counterfactual employee?

Please provide your response in the following format exactly:"""

            return f"""{AttritionDataset.INTRO_COUNTERFACTUAL}

{AttritionDataset.COUNTERFACTUAL_SETUP_COT}

{AttritionDataset.COUNTERFACTUAL_COT_INSTRUCTION}

{reference_section}

{counterfactual_section}

{AttritionDataset.FORMAT_ANSWER}

{AttritionDataset.FORMAT_CONFIDENCE}"""

        else:  # normal mode
            important_factors = question_explanation.get("most_important_factors", [])

            # Format important factors as a bulleted list
            factors_text = ""
            if important_factors:
                factors_text = "\n".join([f"- {factor}" for factor in important_factors])
            else:
                factors_text = "No specific factors listed"

            reference_section = f"""--- REFERENCE EMPLOYEE ---
Profile:
{question}

Likely to Leave: {center_outcome}

Assessment:
{center_reasoning}

Most Important Factors Identified:
{factors_text}"""

            counterfactual_section = f"""--- COUNTERFACTUAL EMPLOYEE ---
Profile:
{counterfactual_question}

Based on the analyst's assessment of the reference employee, how would the analyst assess this counterfactual employee?

Please provide your response in the following format exactly:"""

            return f"""{AttritionDataset.INTRO_COUNTERFACTUAL}

{AttritionDataset.COUNTERFACTUAL_SETUP_WITH_EXPLANATION}

{AttritionDataset.COUNTERFACTUAL_WITH_EXPLANATION_INSTRUCTION}

{reference_section}

{counterfactual_section}

{AttritionDataset.FORMAT_ANSWER}

{AttritionDataset.FORMAT_CONFIDENCE}"""

    @staticmethod
    def create_counterfactual_prompt_no_explanation(
            question: str,
            question_explanation: Dict[str, Any],
            counterfactual_question: str,
            answer_last: bool = False
        ) -> str:
        """
        Create a prompt asking the LLM to predict attrition on a counterfactual
        WITHOUT using the reference's explanation
        """
        center_outcome = question_explanation.get("answer", "UNKNOWN")
        
        reference_section = f"""--- REFERENCE EMPLOYEE ---
Profile:
{question}
Likely to Leave: {center_outcome}"""

        counterfactual_section = f"""--- COUNTERFACTUAL EMPLOYEE ---
Profile:
{counterfactual_question}

Based on the analyst's assessment of the reference employee, how would the analyst assess this counterfactual employee?

Please provide your response in the following format exactly:"""
        
        if answer_last:
            return f"""{AttritionDataset.INTRO_COUNTERFACTUAL}

{AttritionDataset.COUNTERFACTUAL_SETUP}

{AttritionDataset.COUNTERFACTUAL_INSTRUCTION}

{reference_section}

{counterfactual_section}

{AttritionDataset.FORMAT_ANSWER}

{AttritionDataset.FORMAT_CONFIDENCE}
"""
        else:
            return f"""{AttritionDataset.INTRO_COUNTERFACTUAL}

{AttritionDataset.COUNTERFACTUAL_SETUP}

{AttritionDataset.COUNTERFACTUAL_INSTRUCTION}

{reference_section}

{counterfactual_section}

{AttritionDataset.FORMAT_ANSWER}

{AttritionDataset.FORMAT_CONFIDENCE}"""
