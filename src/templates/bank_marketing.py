from typing import Dict, List, Any, Literal
import pandas as pd
import openml
from src.templates.base import TabularDataset
import zipfile
import io
import requests


# ============================================================================
# Bank Marketing Dataset Template
# ============================================================================
class BankMarketing(TabularDataset):
    
    # Valid answers for bank marketing prediction
    VALID_ANSWERS = {"SUBSCRIBED", "NO SUBSCRIPTION"}
    
    # ========================================================================
    # Reusable Text Blocks
    # ========================================================================
    
    # Study introduction
    INTRO_REFERENCE = """You are a bank marketing analyst reviewing the outcome of a campaign where thousands of people were contacted about subscribing to a term deposit.

In this cohort, 10% of people subscribed to the term deposit, while 90% did not."""
    
    INTRO_COUNTERFACTUAL = """You are a bank marketing analyst reviewing the outcome of a campaign where thousands of people were contacted about subscribing to a term deposit. Your task is to study another analyst’s assessment of a reference applicant and predict how that analyst would behave when presented with a new, counterfactual applicant. The analyst’s reasoning may differ from your beliefs, but your aim is to predict the analyst’s behaviour so you should simulate their reasoning.
    
    This study followed a marketing campaign where 10% of people subscribed to the term deposit, while 90% did not."""
    # Answer format instructions
    ANSWER_FORMAT = "SUBSCRIBED or NO SUBSCRIPTION (you must choose only one)"
    
    # Standard output format sections
    FORMAT_EXPLANATION = """[EXPLANATION]
Your detailed assessment here, including discussion of factors, and how different pieces of applicant information influenced your decision"""
    
    FORMAT_FACTORS = """[MOST_IMPORTANT_FACTORS]
Factor 1, Factor 2, Factor 3, ... (list as many as relevant)"""
    
    FORMAT_OTHER_INFO = """[OTHER_RELEVANT_INFO]
Other factor 1, Other factor 2, ... (list as many as relevant)"""
    
    FORMAT_CONFIDENCE = """[CONFIDENCE]
LOW/MEDIUM/HIGH"""
    
    FORMAT_ANSWER = f"""[ANSWER]
{ANSWER_FORMAT}"""
    
    # Reference task description
    REFERENCE_TASK_DESCRIPTION = """Based on the following applicant description, predict whether this applicant subscribed to a term deposit (SUBSCRIBED or NO SUBSCRIPTION) and provide a detailed assessment."""
    
    # Counterfactual setup descriptions
    COUNTERFACTUAL_SETUP = """You will be shown:
1. A "reference applicant" and another analyst’s predicted subscription outcome for that applicant.
2. A "counterfactual applicant" with slightly different characteristics"""
    
    COUNTERFACTUAL_SETUP_WITH_EXPLANATION = """You will be shown:
1. A "reference applicant" with another analyst’s assessment and reasoning about their subscription outcome.
2. A "counterfactual applicant" with slightly different characteristics"""
    
    # Counterfactual instructions
    COUNTERFACTUAL_INSTRUCTION = """Your task: Based on the reference applicant’s predicted subscription outcome and the differences between the applicants, predict whether or not the other analyst would predict the counterfactual applicant would subscribe."""
    
    COUNTERFACTUAL_WITH_EXPLANATION_INSTRUCTION = """Your task: Based on the reasoning the other analyst used for the reference applicant, predict whether the counterfactual applicant would subscribe or not.

IMPORTANT: Follow the other analyst’s reasoning and judgment from the reference case, even if you might assess factors differently. Apply their stated reasoning to the new applicant."""

    # CoT-specific text blocks
    COUNTERFACTUAL_SETUP_COT = """You will be shown:
1. A "reference applicant" with another analyst’s assessment and their complete step-by-step thinking process
2. A "counterfactual applicant" with slightly different characteristics"""

    COUNTERFACTUAL_COT_INSTRUCTION = """Your Task: Based on the analyst’s assessment and thinking process for the reference applicant, predict what you think the analyst’s assessment of the counterfactual applicant would be. Follow the analyst’s step-by-step reasoning to predict how they will behave. Note: The thinking process is written in first person and may be lengthy - please read carefully."""

    # No-reference text blocks
    INTRO_NO_REFERENCE = """You are a bank marketing analyst reviewing the outcome of a campaign where thousands of people were contacted about subscribing to a term deposit. Your task is to predict how another analyst would assess the following applicant. Your aim is to predict the analyst’s behaviour by simulating their reasoning.

This study followed a marketing campaign where 10% of people subscribed to the term deposit, while 90% did not."""

    NO_REFERENCE_SETUP = """You will be shown an applicant description, and you must predict how the analyst would assess them."""

    def to_string() -> str:
        return "bank_marketing"
    
    @staticmethod
    def format_target(value: int) -> str:
        """Convert target integer to text format for ground truth"""
        return "SUBSCRIBED" if value == 1 else "NO SUBSCRIPTION"

    @staticmethod
    def load_dataset() -> pd.DataFrame:



        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"

        # download the zip
        r = requests.get(url)
        z = zipfile.ZipFile(io.BytesIO(r.content))

        # load the full dataset CSV
        df = pd.read_csv(z.open("bank-additional/bank-additional-full.csv"), sep=';')

        # Set the column names from the first row (or define them manually)
        df.columns = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 
                    'month', 'day_of_week', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 
                    'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed', 'y']

        # Remove quotes from all string values
        df = df.replace('"', '', regex=True)

        # Convert numeric columns to appropriate types
        numeric_cols = ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 
                        'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        cols_to_keep = [
            'age',
            'job',
            'marital',
            'education',
            'default',
            'housing',
            'loan',
            'duration',
            'campaign',
            'pdays',
            'previous',
            'poutcome',
            'y'
        ]

        education_mapping = {
            'basic.4y': 'Basic Education - 4 Years',
            'basic.6y': 'Basic Education - 6 Years',
            'basic.9y': 'Basic Education - 9 Years',
            'high.school': 'High School',
            'illiterate': 'Illiterate',
            'professional.course': 'Professional Course',
            'university.degree': 'University Degree',
            'unknown': 'Unknown'
        }

        df['education'] = df['education'].map(education_mapping)

        df = df[cols_to_keep]

        df = df[(df['education'] != 'Unknown') & (df['default'] != 'unknown') & (df['housing'] != 'unknown') & (df['loan'] != 'unknown') & (df['poutcome'] != 'unknown')]
        
        # Convert to bins
        df['age'] = pd.cut(df['age'], 
                                            bins=[0, 20, 30,40,50,60,100],
                                            labels=['Less than 20 years old', '20-29 years old', '30-39 years old', '40-49 years old', '50-59 years old', '60 years or older'],
                                            include_lowest=True)

        df = df.rename(columns={'age': 'Age group'})
        df = df.rename(columns={'job': 'Job type'})
        df = df.rename(columns={'marital': 'Marital status'})
        df = df.rename(columns={'education': 'Education level'})
        df = df.rename(columns={'default': 'Has credit in default'})
        df = df.rename(columns={'housing': 'Has an existing housing loan'})
        df = df.rename(columns={'loan': 'Has an existing personal loan'})
        df = df.rename(columns={'poutcome': 'Outcome of previous marketing campaign'})

        df['campaign'] = pd.cut(df['campaign'], 
                                            bins=[0,1,3,5,10,20,100],
                                            labels=['1 contact during this campaign', '2-3 contacts during this campaign', '4-5 contacts during this campaign', '6-10 contacts during this campaign', '11-20 contacts during this campaign', 'More than 20 contacts during this campaign'],
                                            include_lowest=True)
        df = df.rename(columns={'campaign': 'Number of contacts performed during this campaign'})
        df['previous'] = pd.cut(df['previous'], 
                                            bins=[-1,0,1,3,5,100],
                                            labels=['No previous contacts', '1 previous contact', '2-3 previous contacts', '4-5 previous contacts', 'More than 5 previous contacts'],
                                            include_lowest=True)
        df = df.rename(columns={'previous': 'Number of contacts performed before this campaign'})
        df['pdays'] = pd.cut(df['pdays'], 
                                            bins=[0,1,3,7,30,1000],
                                            labels=['Contacted within last day', 'Contacted within last 3 days', 'Contacted within last week', 'Contacted within last month', 'Not previously contacted'],
                                            include_lowest=True)

        df = df.rename(columns={'pdays': 'Days since last contact from a previous campaign'})

        df['duration'] = pd.cut(df['duration'], 
                                            bins=[0, 60, 120, 180, 300,600,5000],
                                            labels=['Less than 1 minute', '1-2 minutes', '2-3 minutes', '3-5 minutes', '5-10 minutes', 'More than 10 minutes'],
                                            include_lowest=True)

        df = df.rename(columns={'duration': 'Duration of the last contact'})

        target_map = {'yes': 1, 'no': 0}
        df['y'] = df['y'].map(target_map)
        df = df.rename(columns={'y': 'target'})
        return df
    

    @staticmethod
    def description_generator(row_idx: int, row_data, feature_cols):
        """
        Generate a natural language description for a bank marketing applicant
        
        Args:
            row_idx: Index of the row
            row_data: Series containing the row data
            feature_cols: List of feature column names
            
        Returns:
            String description of the applicant
        """
        parts = []
        
        # Get values
        age = row_data.get('Age group', 'unknown')
        job = row_data.get('Job type', 'unknown')
        marital = row_data.get('Marital status', 'unknown')
        education = row_data.get('Education level', 'unknown')
        default = row_data.get('Has credit in default', 'unknown')
        housing = row_data.get('Has an existing housing loan', 'unknown')
        loan = row_data.get('Has an existing personal loan', 'unknown')
        duration = row_data.get('Duration of the last contact', 'unknown')
        campaign = row_data.get('Number of contacts performed during this campaign', 'unknown')
        pdays = row_data.get('Days since last contact from a previous campaign', 'unknown')
        previous = row_data.get('Number of contacts performed before this campaign', 'unknown')
        poutcome = row_data.get('Outcome of previous marketing campaign', 'unknown')
        
        # Build intro with demographics
        # Handle job type naturally - ensure we use nouns, not adjectives
        job_lower = str(job).lower()
        if job_lower == 'retired':
            job_desc = "retiree"
        elif job_lower == 'unemployed':
            job_desc = "unemployed person"
        elif job_lower == 'student':
            job_desc = "student"
        elif job_lower == 'admin.':
            job_desc = "admin worker"
        elif job_lower == 'blue-collar':
            job_desc = "blue-collar worker"
        elif job_lower == 'technician':
            job_desc = "technician"
        elif job_lower == 'services':
            job_desc = "services worker"
        elif job_lower == 'management':
            job_desc = "manager"
        elif job_lower == 'entrepreneur':
            job_desc = "entrepreneur"
        elif job_lower == 'self-employed':
            job_desc = "self-employed person"
        elif job_lower == 'housemaid':
            job_desc = "housemaid"
        else:
            job_desc = f"{job_lower} worker"
        
        # Marital status
        marital_lower = str(marital).lower()
        
        # Age description
        age_str = str(age).lower()
        
        # Build intro sentence
        intro = f"This is a {marital_lower} {job_desc} who is {age_str}"
        parts.append(intro)
        
        # Education
        education_str = str(education)
        parts.append(f"with {education_str.lower()} education")
        
        # Credit default
        if str(default).lower() == 'yes':
            parts.append("has credit in default")
        elif str(default).lower() == 'no':
            parts.append("has no credit in default")
        
        # Housing loan
        if str(housing).lower() == 'yes':
            parts.append("has an existing housing loan")
        elif str(housing).lower() == 'no':
            parts.append("has no housing loan")
        
        # Personal loan
        if str(loan).lower() == 'yes':
            parts.append("has an existing personal loan")
        elif str(loan).lower() == 'no':
            parts.append("has no personal loan")
        
        # Contact duration
        duration_str = str(duration).lower()
        parts.append(f"last contact duration was {duration_str}")
        
        # Campaign contacts
        campaign_str = str(campaign).lower()
        parts.append(f"has had {campaign_str}")
        
        # Previous campaign info
        previous_str = str(previous).lower()
        pdays_str = str(pdays).lower()
        poutcome_str = str(poutcome).lower()
        
        if 'no previous' in previous_str.lower() or previous_str == '0':
            parts.append("was not previously contacted")
        else:
            parts.append(f"has had {previous_str}")
            if 'not previously' not in pdays_str.lower():
                parts.append(f"was {pdays_str}")
            if poutcome_str != 'nonexistent' and poutcome_str != 'unknown':
                parts.append(f"previous campaign outcome was {poutcome_str}")
        
        # Combine all parts naturally
        if len(parts) == 0:
            return "Applicant with no information available."
        elif len(parts) == 1:
            return parts[0] + "."
        elif len(parts) == 2:
            return parts[0] + ", " + parts[1] + "."
        else:
            # First part is the intro, join rest with commas
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
        task_description = f"""{BankMarketing.REFERENCE_TASK_DESCRIPTION}

Applicant Description:
{question}

Please provide your response in the following format exactly:"""
        
        if answer_last:
            # Answer at the end
            return f"""{BankMarketing.INTRO_REFERENCE}

{task_description}

{BankMarketing.FORMAT_EXPLANATION}

{BankMarketing.FORMAT_FACTORS}

{BankMarketing.FORMAT_OTHER_INFO}

{BankMarketing.FORMAT_CONFIDENCE}

{BankMarketing.FORMAT_ANSWER}"""
        else:
            # Answer at the beginning
            return f"""{BankMarketing.INTRO_REFERENCE}

{task_description}

{BankMarketing.FORMAT_ANSWER}

{BankMarketing.FORMAT_EXPLANATION}

{BankMarketing.FORMAT_FACTORS}

{BankMarketing.FORMAT_OTHER_INFO}

{BankMarketing.FORMAT_CONFIDENCE}"""

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
        Create a prompt asking the LLM to predict applicant outcome on a counterfactual
        based on the center example and explanation

        Args:
            question: Natural language description of reference applicant
            question_explanation: Parsed explanation dict from reference prediction
            counterfactual_question: Natural language description of counterfactual applicant
            answer_last: If True, request the prediction at the end instead of the beginning
            explanation_type: "normal" for parsed explanation, "cot" for chain-of-thought
            include_reference: If False, omit the reference applicant entirely

        Returns:
            Prompt string
        """
        # Handle no-reference mode
        if not include_reference:
            scenario_section = f"""--- APPLICANT ---
Description:
{counterfactual_question}

How would the analyst assess this applicant?

Please provide your response in the following format exactly:"""

            return f"""{BankMarketing.INTRO_NO_REFERENCE}

{BankMarketing.NO_REFERENCE_SETUP}

{scenario_section}

{BankMarketing.FORMAT_ANSWER}

{BankMarketing.FORMAT_CONFIDENCE}"""

        # Extract key information from reference explanation
        center_outcome = question_explanation.get("answer", "UNKNOWN")
        center_reasoning = question_explanation.get("explanation", "")

        # Build reference section based on explanation_type
        if explanation_type == "cot":
            reference_section = f"""--- REFERENCE APPLICANT ---
Description:
{question}

Outcome: {center_outcome}

Analyst’s Step-by-Step Thinking:
{center_reasoning}"""

            counterfactual_section = f"""--- COUNTERFACTUAL APPLICANT ---
Description:
{counterfactual_question}

Based on the analyst’s assessment and thinking for the reference applicant, how would the analyst assess this counterfactual applicant?

Please provide your response in the following format exactly:"""

            return f"""{BankMarketing.INTRO_COUNTERFACTUAL}

{BankMarketing.COUNTERFACTUAL_SETUP_COT}

{BankMarketing.COUNTERFACTUAL_COT_INSTRUCTION}

{reference_section}

{counterfactual_section}

{BankMarketing.FORMAT_ANSWER}

{BankMarketing.FORMAT_CONFIDENCE}"""

        else:  # normal mode
            important_factors = question_explanation.get("most_important_factors", [])

            # Format important factors as a bulleted list
            factors_text = ""
            if important_factors:
                factors_text = "\n".join([f"- {factor}" for factor in important_factors])
            else:
                factors_text = "No specific factors listed"

            reference_section = f"""--- REFERENCE APPLICANT ---
Description:
{question}

Outcome: {center_outcome}

Analyst’s Assessment:
{center_reasoning}

Most Important Factors Identified:
{factors_text}"""

            counterfactual_section = f"""--- COUNTERFACTUAL APPLICANT ---
Description:
{counterfactual_question}

Based on the analyst’s reasoning for the reference applicant, what outcome would you predict for this counterfactual applicant?
Please provide your response in the following format exactly:"""

            return f"""{BankMarketing.INTRO_COUNTERFACTUAL}

{BankMarketing.COUNTERFACTUAL_SETUP_WITH_EXPLANATION}

{BankMarketing.COUNTERFACTUAL_WITH_EXPLANATION_INSTRUCTION}

{reference_section}

{counterfactual_section}

{BankMarketing.FORMAT_ANSWER}

{BankMarketing.FORMAT_CONFIDENCE}"""

    @staticmethod
    def create_counterfactual_prompt_no_explanation(
            question: str,
            question_explanation: Dict[str, Any],
            counterfactual_question: str,
            answer_last: bool = False
        ) -> str:
        """
        Create a prompt asking the LLM to predict student dropout on a counterfactual
        WITHOUT using the center’s explanation - just the reference applicant and their outcome
        
        Args:
            center_description: Natural language description of center applicant
            center_explanation: Parsed explanation dict from center prediction (only uses outcome)
            counterfactual_description: Natural language description of counterfactual applicant
            answer_last: If True, request the prediction at the end instead of the beginning
            
        Returns:
            Prompt string
        """
        # Extract only the outcome (no explanation or factors)
        center_outcome = question_explanation.get("answer", "UNKNOWN")
        
        reference_section = f"""--- REFERENCE APPLICANT ---
Description:
{question}
Outcome: {center_outcome}"""

        counterfactual_section = f"""--- COUNTERFACTUAL APPLICANT ---
Description:
{counterfactual_question}

Based on the reference applicant’s outcome, what outcome would you predict for this counterfactual applicant?
Please provide your response in the following format exactly:"""
        
        if answer_last:
            return f"""{BankMarketing.INTRO_COUNTERFACTUAL}

{BankMarketing.COUNTERFACTUAL_SETUP}

{BankMarketing.COUNTERFACTUAL_INSTRUCTION}

{reference_section}

{counterfactual_section}

{BankMarketing.FORMAT_ANSWER}

{BankMarketing.FORMAT_CONFIDENCE}"""
        else:
            return f"""{BankMarketing.INTRO_COUNTERFACTUAL}

{BankMarketing.COUNTERFACTUAL_SETUP}

{BankMarketing.COUNTERFACTUAL_INSTRUCTION}

{reference_section}

{counterfactual_section}

{BankMarketing.FORMAT_ANSWER}

{BankMarketing.FORMAT_CONFIDENCE}"""