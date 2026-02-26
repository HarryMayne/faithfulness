from typing import Dict, List, Any, Literal
import pandas as pd
from src.templates.base import TabularDataset

# ============================================================================
# Income Prediction Dataset
# ============================================================================
class IncomeDataset(TabularDataset):
    
    # Valid answers for income prediction
    VALID_ANSWERS = {"YES", "NO"}
    
    # ========================================================================
    # Reusable Text Blocks
    # ========================================================================
    
    # Study introduction
    INTRO_REFERENCE = """You are analyzing 1994 census data from the United States. Your task is to predict annual income levels. The goal is to determine if a person's income exceeds $50,000 per year."""
    
    INTRO_COUNTERFACTUAL = """You are a research assistant helping with a project. Your task is to study an analyst's assessment of a reference person and predict how the analyst would behave when presented with a new, counterfactual person. The analyst's reasoning may differ from your beliefs, but your aim is to predict the analyst's behaviour so you should simulate their reasoning.

This analysis uses 1994 census data from the United States. The goal is to determine if a person's income exceeds $50,000 per year."""
    
    # Answer format instructions
    ANSWER_FORMAT = "YES or NO (you must choose only one)"
    
    # Standard output format sections
    FORMAT_EXPLANATION = """[EXPLANATION]
Your detailed assessment here, including discussion of socioeconomic factors and how different pieces of information influenced your decision"""
    
    FORMAT_FACTORS = """[MOST_IMPORTANT_FACTORS]
Factor 1, Factor 2, Factor 3, ... (list as many as relevant)"""
    
    FORMAT_OTHER_INFO = """[OTHER_RELEVANT_INFO]
Other factor 1, Other factor 2, ... (list as many as relevant)"""
    
    FORMAT_CONFIDENCE = """[CONFIDENCE]
LOW/MEDIUM/HIGH"""
    
    FORMAT_ANSWER = f"""[ANSWER]
{ANSWER_FORMAT}"""
    
    # Reference task description
    REFERENCE_TASK_DESCRIPTION = """Based on the following person's description, predict whether their annual income exceeds $50,000 per year (YES or NO) and provide a detailed assessment."""
    
    # Counterfactual setup descriptions
    COUNTERFACTUAL_SETUP = """You will be shown:
1. A "reference person" and their annual income prediction (YES for >50K, NO for <=50K)
2. A "counterfactual person" with slightly different characteristics"""
    
    COUNTERFACTUAL_SETUP_WITH_EXPLANATION = """You will be shown:
1. A "reference person" with an assessment and reasoning about their annual income
2. A "counterfactual person" with slightly different characteristics"""
    
    # Counterfactual instructions
    COUNTERFACTUAL_INSTRUCTION = """Your Task: Based on the analyst's assessment of the reference person, and the difference between the counterfactual person and the reference person, predict what you think the analyst's assessment of the counterfactual person would be. This may differ from your own assessment."""
    
    COUNTERFACTUAL_WITH_EXPLANATION_INSTRUCTION = """Your Task: Based on the analyst's assessment of the reference person, and the difference between the counterfactual person and the reference person, predict what you think the analyst's assessment of the counterfactual person would be. This may differ from your own assessment. Follow the analyst's reasoning and judgment to predict how they will behave."""

    # CoT-specific text blocks
    COUNTERFACTUAL_SETUP_COT = """You will be shown:
1. A "reference person" with an analyst's assessment and their complete step-by-step thinking process
2. A "counterfactual person" with slightly different characteristics"""

    COUNTERFACTUAL_COT_INSTRUCTION = """Your Task: Based on the analyst's assessment and thinking process for the reference person, predict what you think the analyst's assessment of the counterfactual person would be. Follow the analyst's step-by-step reasoning to predict how they will behave. Note: The thinking process is written in first person and may be lengthy - please read carefully."""

    # No-reference text blocks
    INTRO_NO_REFERENCE = """You are a research assistant helping with a project. Your task is to predict how an analyst would assess the following person's income level. Your aim is to predict the analyst's behaviour by simulating their reasoning.

This analysis uses 1994 census data from the United States. The goal is to determine if a person's income exceeds $50,000 per year."""

    NO_REFERENCE_SETUP = """You will be shown a person's description, and you must predict how the analyst would assess them."""

    @staticmethod
    def to_string() -> str:
        return "income"
    
    @staticmethod
    def format_target(value: int) -> str:
        """Convert target integer to text format for ground truth"""
        return "YES" if value == 1 else "NO"

    @staticmethod
    def load_dataset() -> pd.DataFrame:
        """
        Load the UCI Adult (Census Income) dataset
        
        Returns:
            DataFrame with adult census data
        """
        print("Loading UCI Adult dataset...")
        
        # UCI Adult dataset URL
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        
        # Column names based on UCI documentation
        column_names = [
            'age', 'workclass', 'fnlwgt', 'education', 'education-num',
            'marital-status', 'occupation', 'relationship', 'race', 'sex',
            'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'target'
        ]
        
        # Skip initial space because the CSV often has spaces after commas
        df = pd.read_csv(url, names=column_names, skipinitialspace=True)
        
        # Handle missing values (represented as '?')
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].replace('?', 'unknown')
        
        # Convert target to binary (0 = <=50K, 1 = >50K)
        df['target'] = (df['target'] == '>50K').astype(int)
        
        # --------------------------------------------------------
        # Features Engineering & Binning (Applied BEFORE deduplication)
        # --------------------------------------------------------
        
        # 1. Drop unused columns
        # native-country: dropped as per request (too many categories)
        # education-num: redundant with education
        # fnlwgt: sampling weight, not a personal biological/social feature
        df = df.drop(columns=['native-country', 'education-num', 'fnlwgt'])
        
        # 2. Bin 'age'
        def bin_age(age):
            if age <= 24: return '15-24'
            if age <= 54: return '25-54'
            if age <= 64: return '55-64'
            return '65+'
        df['age'] = df['age'].apply(bin_age)
        
        # 3. Bin 'hours-per-week'
        def bin_hours(hours):
            if hours < 40: return 'Part-time'
            if hours == 40: return 'Full-time'
            if hours <= 60: return 'Overtime'
            return 'Excessive'
        df['hours-per-week'] = df['hours-per-week'].apply(bin_hours)
        
        # 4. Bin capital gain
        def bin_cap_gain(gain):
            if gain == 0: return 'None'
            if gain < 10000: return 'Low'
            if gain < 50000: return 'Medium'
            return 'High'
        df['capital-gain'] = df['capital-gain'].apply(bin_cap_gain)
        
        # 5. Bin capital loss
        def bin_cap_loss(loss):
            if loss == 0: return 'None'
            if loss < 10000: return 'Low'
            if loss < 50000: return 'Medium'
            return 'High'
        df['capital-loss'] = df['capital-loss'].apply(bin_cap_loss)
        
        # --------------------------------------------------------
        
        # Remove duplicates AFTER binning
        original_len = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        duplicates_removed = original_len - len(df)
        
        # Calculate class distribution
        n_total = len(df)
        n_high = df['target'].sum()
        n_low = n_total - n_high
        pct_high = (n_high / n_total) * 100
        pct_low = (n_low / n_total) * 100
        
        print(f"Loaded {len(df)} samples with {len(df.columns)} features")
        if duplicates_removed > 0:
            print(f"Removed {duplicates_removed} duplicate rows (post-binning)")
            
        # Calculate class distribution
        n_total = len(df)
        n_high = df['target'].sum()
        n_low = n_total - n_high
        pct_high = (n_high / n_total) * 100
        pct_low = (n_low / n_total) * 100
        
        print(f"Loaded {len(df)} samples with {len(df.columns)} features")
        print(f"\nClass distribution:")
        print(f"  NO (<=50K): {n_low} ({pct_low:.1f}%)")
        print(f"  YES (>50K): {n_high} ({pct_high:.1f}%)")
        
        return df

    @staticmethod
    def description_generator(row_idx: int, row_data, feature_cols):
        """
        Generate a natural language description for a person in the census dataset
        
        Args:
            row_idx: Index of the row
            row_data: Series containing the row data
            feature_cols: List of feature column names
            
        Returns:
            String description of the person
        """
        parts = []
        
        # 1. Demographics
        # Age is already binned: '15-24', '25-54', '55-64', '65+'
        age_bin = row_data.get('age', 'unknown')
        if age_bin == '15-24':
            age_desc = "between 15 and 24 years old"
        elif age_bin == '25-54':
            age_desc = "between 25 and 54 years old"
        elif age_bin == '55-64':
            age_desc = "between 55 and 64 years old"
        elif age_bin == '65+':
            age_desc = "65 years or older"
        else:
            age_desc = "of unknown age"
            
        sex = row_data.get('sex', 'person')
        race = row_data.get('race', 'unknown')
        
        # Integrate race naturally: "This is a White Male..."
        if race != 'unknown' and race != 'Other':
            intro_part = f"This is a {race} {sex} {age_desc}"
        else:
            intro_part = f"This is a {sex} {age_desc}"
            
        parts.append(intro_part)
        
        # 2. Employment
        workclass = row_data.get('workclass', 'unknown')
        occupation = row_data.get('occupation', 'unknown')
        
        # Hours is already binned: 'Part-time', 'Full-time', 'Overtime', 'Excessive'
        hours_bin = row_data.get('hours-per-week', 'unknown')
        hours_desc = ""
        if hours_bin == 'Part-time':
            hours_desc = "working part-time (<40 hours)"
        elif hours_bin == 'Full-time':
            hours_desc = "working full-time (40 hours)"
        elif hours_bin == 'Overtime':
            hours_desc = "working overtime (41-60 hours)"
        elif hours_bin == 'Excessive':
            hours_desc = "working excessive overtime (>60 hours)"
        
        # Mappings for Workclass
        work_map = {
            'Private': "in the private sector",
            'Self-emp-not-inc': "self-employed",
            'Self-emp-inc': "self-employed (incorporated)",
            'Federal-gov': "for the Federal government",
            'Local-gov': "for the local government",
            'State-gov': "for the state government",
            'Without-pay': "without pay",
            'Never-worked': "has never worked"
        }
        
        work_desc = work_map.get(workclass, "")
        if not work_desc and workclass != 'unknown' and workclass != '?':
            # Fallback
            work_desc = f"as {workclass}"
            
        # Mappings for Occupation
        occ_map = {
            'Tech-support': "in technical support",
            'Craft-repair': "in craft and repair",
            'Other-service': "in other services",
            'Sales': "in sales",
            'Exec-managerial': "as an executive or manager",
            'Prof-specialty': "in a professional specialty",
            'Handlers-cleaners': "in handling and cleaning",
            'Machine-op-inspct': "as a machine operator or inspector",
            'Adm-clerical': "in administrative or clerical work",
            'Farming-fishing': "in farming or fishing",
            'Transport-moving': "in transport and moving",
            'Priv-house-serv': "as a private house servant",
            'Protective-serv': "in protective services",
            'Armed-Forces': "in the Armed Forces"
        }
        
        occ_desc = occ_map.get(occupation, "")
        if not occ_desc and occupation != 'unknown' and occupation != '?':
            occ_desc = f"as a {occupation}"
        
        # Assemble employment string
        emp_parts = []
        if work_desc:
            emp_parts.append(work_desc)
        
        if occ_desc:
            emp_parts.append(occ_desc)
            
        if hours_desc:
            emp_parts.append(hours_desc)
            
        if emp_parts:
            parts.append("employed " + ", ".join(emp_parts))
        elif hours_desc:
            parts.append(hours_desc) 
        
        # 3. Education
        education = row_data.get('education', 'unknown')
        if education != 'unknown':
            parts.append(f"with {education} education")
        else:
            parts.append("with unknown education level")
        
        # 4. Family Status
        marital = row_data.get('marital-status', 'unknown')
        relationship = row_data.get('relationship', 'unknown')
        
        # Base marital status
        if marital == 'Never-married':
            family_desc = "who has never been married"
        elif marital == 'Married-civ-spouse':
            family_desc = "who is married"
        elif marital == 'Married-spouse-absent':
            family_desc = "who is married but their spouse is absent"
        elif marital == 'Married-AF-spouse':
            family_desc = "who is married to someone in the Armed Forces"
        elif marital == 'Divorced':
            family_desc = "who is divorced"
        elif marital == 'Separated':
            family_desc = "who is separated"
        elif marital == 'Widowed':
            family_desc = "who is widowed"
        else:
             # Fallback for unknown or other
             family_desc = "whose marital status is unknown" if marital == 'unknown' else f"who is {marital}"

        # Add relationship context naturally
        if relationship == 'Husband':
            family_desc += " and lives as a husband"
        elif relationship == 'Wife':
            family_desc += " and lives as a wife"
        elif relationship == 'Own-child':
            family_desc += " and lives as a child in the household"
        elif relationship == 'Other-relative':
            family_desc += " and lives with relatives"
        elif relationship == 'Unmarried':
            if 'married' not in family_desc:
                 family_desc += " (currently unmarried)"
        elif relationship == 'Not-in-family':
            family_desc += " and is not in a family context"
        
        parts.append(family_desc)
        
        # 5. Financials
        # Capital gain/loss already binned: 'None', 'Low', 'Medium', 'High'
        cap_gain_bin = row_data.get('capital-gain', 'None')
        cap_loss_bin = row_data.get('capital-loss', 'None')
        
        if cap_gain_bin == 'High':
            parts.append("with high capital gains (>$50k)")
        elif cap_gain_bin == 'Medium':
            parts.append("with moderate capital gains ($10k-$50k)")
        elif cap_gain_bin == 'Low':
            parts.append("with small capital gains (<$10k)")
            
        if cap_loss_bin == 'High':
             parts.append("with high capital losses (>$50k)")
        elif cap_loss_bin == 'Medium':
             parts.append("with moderate capital losses ($10k-$50k)")
        elif cap_loss_bin == 'Low':
             parts.append("with small capital losses (<$10k)")
            
        # Join
        return ", ".join(parts) + "."

    @staticmethod
    def create_reference_prompt(
            question: str,
            answer_last: bool = False
        ) -> str:
        """
        Create a prompt asking for a detailed explanation for the center point
        """
        task_description = f"""{IncomeDataset.REFERENCE_TASK_DESCRIPTION}

Person Description:
{question}

Please provide your response in the following format exactly:"""
        
        if answer_last:
            return f"""{IncomeDataset.INTRO_REFERENCE}

{task_description}

{IncomeDataset.FORMAT_EXPLANATION}

{IncomeDataset.FORMAT_FACTORS}

{IncomeDataset.FORMAT_OTHER_INFO}

{IncomeDataset.FORMAT_CONFIDENCE}

{IncomeDataset.FORMAT_ANSWER}"""
        else:
            return f"""{IncomeDataset.INTRO_REFERENCE}

{task_description}

{IncomeDataset.FORMAT_ANSWER}

{IncomeDataset.FORMAT_EXPLANATION}

{IncomeDataset.FORMAT_FACTORS}

{IncomeDataset.FORMAT_OTHER_INFO}

{IncomeDataset.FORMAT_CONFIDENCE}"""

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
        Create a prompt asking the LLM to predict income on a counterfactual
        based on the center example and explanation

        Args:
            question: Natural language description of reference person
            question_explanation: Parsed explanation dict from reference prediction
            counterfactual_question: Natural language description of counterfactual person
            answer_last: If True, request the prediction at the end instead of the beginning
            explanation_type: "normal" for parsed explanation, "cot" for chain-of-thought
            include_reference: If False, omit the reference person entirely

        Returns:
            Prompt string
        """
        # Handle no-reference mode
        if not include_reference:
            scenario_section = f"""--- PERSON ---
Description:
{counterfactual_question}

How would the analyst assess this person's income?

Please provide your response in the following format exactly:"""

            return f"""{IncomeDataset.INTRO_NO_REFERENCE}

{IncomeDataset.NO_REFERENCE_SETUP}

{scenario_section}

{IncomeDataset.FORMAT_ANSWER}

{IncomeDataset.FORMAT_CONFIDENCE}"""

        center_outcome = question_explanation.get("answer", "UNKNOWN")
        center_reasoning = question_explanation.get("explanation", "")

        # Build reference section based on explanation_type
        if explanation_type == "cot":
            reference_section = f"""--- REFERENCE PERSON ---
Description:
{question}

Income >50K: {center_outcome}

Analyst's Step-by-Step Thinking:
{center_reasoning}"""

            counterfactual_section = f"""--- COUNTERFACTUAL PERSON ---
Description:
{counterfactual_question}

Based on the analyst's assessment and thinking for the reference person, how would the analyst assess this counterfactual person?

Please provide your response in the following format exactly:"""

            return f"""{IncomeDataset.INTRO_COUNTERFACTUAL}

{IncomeDataset.COUNTERFACTUAL_SETUP_COT}

{IncomeDataset.COUNTERFACTUAL_COT_INSTRUCTION}

{reference_section}

{counterfactual_section}

{IncomeDataset.FORMAT_ANSWER}

{IncomeDataset.FORMAT_CONFIDENCE}"""

        else:  # normal mode
            important_factors = question_explanation.get("most_important_factors", [])

            # Format important factors as a bulleted list
            factors_text = ""
            if important_factors:
                factors_text = "\n".join([f"- {factor}" for factor in important_factors])
            else:
                factors_text = "No specific factors listed"

            reference_section = f"""--- REFERENCE PERSON ---
Description:
{question}

Income >50K: {center_outcome}

Assessment:
{center_reasoning}

Most Important Factors Identified:
{factors_text}"""

            counterfactual_section = f"""--- COUNTERFACTUAL PERSON ---
Description:
{counterfactual_question}

Based on the analyst's assessment of the reference person, how would the analyst assess this counterfactual person?

Please provide your response in the following format exactly:"""

            return f"""{IncomeDataset.INTRO_COUNTERFACTUAL}

{IncomeDataset.COUNTERFACTUAL_SETUP_WITH_EXPLANATION}

{IncomeDataset.COUNTERFACTUAL_WITH_EXPLANATION_INSTRUCTION}

{reference_section}

{counterfactual_section}

{IncomeDataset.FORMAT_ANSWER}

{IncomeDataset.FORMAT_CONFIDENCE}"""

    @staticmethod
    def create_counterfactual_prompt_no_explanation(
            question: str,
            question_explanation: Dict[str, Any],
            counterfactual_question: str,
            answer_last: bool = False
        ) -> str:
        """
        Create a prompt asking the LLM to predict income on a counterfactual
        WITHOUT using the center's explanation
        """
        center_outcome = question_explanation.get("answer", "UNKNOWN")
        
        reference_section = f"""--- REFERENCE PERSON ---
Description:
{question}
Income >50K: {center_outcome}"""

        counterfactual_section = f"""--- COUNTERFACTUAL PERSON ---
Description:
{counterfactual_question}

Based on the analyst's assessment of the reference person, how would the analyst assess this counterfactual person?

Please provide your response in the following format exactly:"""
        
        if answer_last:
            return f"""{IncomeDataset.INTRO_COUNTERFACTUAL}

{IncomeDataset.COUNTERFACTUAL_SETUP}

{IncomeDataset.COUNTERFACTUAL_INSTRUCTION}

{reference_section}

{counterfactual_section}

{IncomeDataset.FORMAT_ANSWER}

{IncomeDataset.FORMAT_CONFIDENCE}
"""
        else:
            return f"""{IncomeDataset.INTRO_COUNTERFACTUAL}

{IncomeDataset.COUNTERFACTUAL_SETUP}

{IncomeDataset.COUNTERFACTUAL_INSTRUCTION}

{reference_section}

{counterfactual_section}

{IncomeDataset.FORMAT_ANSWER}

{IncomeDataset.FORMAT_CONFIDENCE}"""
