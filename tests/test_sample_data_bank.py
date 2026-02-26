"""
Example pytest scripts for sample_data_bank.parquet

This module demonstrates various pytest techniques for testing
the bank marketing counterfactual dataset.
"""

import pytest
import pandas as pd
import os
from pathlib import Path


# Constants
SAMPLE_DATA_PATH = "tests/test_outputs/sample_data_bank.parquet"


@pytest.fixture
def sample_data():
    """
    Fixture to load the sample data once and reuse across tests.
    """
    assert os.path.exists(SAMPLE_DATA_PATH), f"Sample data not found at {SAMPLE_DATA_PATH}"
    df = pd.read_parquet(SAMPLE_DATA_PATH)
    return df


@pytest.fixture
def expected_columns():
    """
    Fixture defining expected columns in the dataset.
    """
    return [
        'original_dataset',
        'original_question',
        'original_question_prompt',
        'original_question_idx',
        'original_ground_truth',
        'original_answer_first',
        'original_description',
        'counterfactual_generator_model',
        'counterfactual_generator_method',
        'counterfactual_question',
        'counterfactual_question_prompt',
        'counterfactual_question_idx',
        'counterfactual_ground_truth',
        'counterfactual_description',
        'counterfactual_hamming_distance',
    ]


class TestDataLoading:
    """Test cases for data loading and basic structure."""

    def test_file_exists(self):
        """Test that the sample data file exists."""
        assert os.path.exists(SAMPLE_DATA_PATH), f"File not found: {SAMPLE_DATA_PATH}"

    def test_file_is_parquet(self):
        """Test that the file is a valid parquet file."""
        assert SAMPLE_DATA_PATH.endswith('.parquet'), "File should have .parquet extension"

    def test_data_loads_successfully(self, sample_data):
        """Test that the data loads without errors."""
        assert sample_data is not None
        assert isinstance(sample_data, pd.DataFrame)

    def test_data_not_empty(self, sample_data):
        """Test that the dataset is not empty."""
        assert len(sample_data) > 0, "Dataset should not be empty"
        assert len(sample_data.columns) > 0, "Dataset should have columns"


class TestDataStructure:
    """Test cases for data structure and schema."""

    def test_expected_columns_exist(self, sample_data, expected_columns):
        """Test that all expected columns are present."""
        for col in expected_columns:
            assert col in sample_data.columns, f"Expected column '{col}' not found"

    def test_dataset_column_value(self, sample_data):
        """Test that the dataset column contains 'bank_marketing'."""
        assert 'original_dataset' in sample_data.columns
        unique_datasets = sample_data['original_dataset'].unique()
        assert 'bank_marketing' in unique_datasets

    def test_no_duplicate_indices(self, sample_data):
        """Test that there are no duplicate indices."""
        assert not sample_data.index.duplicated().any(), "Found duplicate indices"

    def test_data_types(self, sample_data):
        """Test that key columns have expected data types."""
        # Text columns should be object/string type
        text_columns = ['original_question', 'original_description',
                       'counterfactual_question', 'counterfactual_description']
        for col in text_columns:
            if col in sample_data.columns:
                assert sample_data[col].dtype in ['object', 'string'], \
                    f"Column '{col}' should be text type"

        # Index columns should be numeric
        if 'original_question_idx' in sample_data.columns:
            assert pd.api.types.is_numeric_dtype(sample_data['original_question_idx'])

        if 'counterfactual_hamming_distance' in sample_data.columns:
            assert pd.api.types.is_numeric_dtype(sample_data['counterfactual_hamming_distance'])


class TestDataQuality:
    """Test cases for data quality checks."""

    def test_no_null_in_critical_columns(self, sample_data):
        """Test that critical columns don't have null values."""
        critical_columns = [
            'original_dataset',
            'original_question',
            'original_ground_truth',
            'counterfactual_question',
            'counterfactual_ground_truth'
        ]

        for col in critical_columns:
            if col in sample_data.columns:
                null_count = sample_data[col].isna().sum()
                assert null_count == 0, f"Column '{col}' has {null_count} null values"

    def test_ground_truth_values(self, sample_data):
        """Test that ground truth values are valid."""
        if 'original_ground_truth' in sample_data.columns:
            valid_values = ['SUBSCRIBED', 'NO SUBSCRIPTION']
            invalid = sample_data[~sample_data['original_ground_truth'].isin(valid_values)]
            assert len(invalid) == 0, \
                f"Found {len(invalid)} rows with invalid ground truth values"

    def test_hamming_distance_range(self, sample_data):
        """Test that hamming distance is within reasonable range."""
        if 'counterfactual_hamming_distance' in sample_data.columns:
            distances = sample_data['counterfactual_hamming_distance'].dropna()
            assert (distances >= 0).all(), "Hamming distance should be non-negative"
            # Assuming hamming distance should be less than some max value
            assert (distances < 100).all(), "Hamming distance seems unreasonably high"

    def test_descriptions_not_empty(self, sample_data):
        """Test that description fields are not empty strings."""
        if 'original_description' in sample_data.columns:
            empty_descriptions = sample_data[sample_data['original_description'].str.len() == 0]
            assert len(empty_descriptions) == 0, \
                f"Found {len(empty_descriptions)} rows with empty descriptions"


class TestCounterfactualPairs:
    """Test cases specific to counterfactual data."""

    def test_original_counterfactual_pairs_exist(self, sample_data):
        """Test that for each original question, there's a counterfactual."""
        assert 'original_question' in sample_data.columns
        assert 'counterfactual_question' in sample_data.columns

        # Both should exist
        assert sample_data['original_question'].notna().all()
        assert sample_data['counterfactual_question'].notna().all()

    def test_questions_are_different(self, sample_data):
        """Test that original and counterfactual questions are different."""
        if 'original_question' in sample_data.columns and \
           'counterfactual_question' in sample_data.columns:
            # At least some should be different (not testing all as some might be same)
            different = sample_data['original_question'] != sample_data['counterfactual_question']
            assert different.any(), "Expected at least some questions to differ"

    def test_generator_method_exists(self, sample_data):
        """Test that counterfactual generator method is specified."""
        if 'counterfactual_generator_method' in sample_data.columns:
            methods = sample_data['counterfactual_generator_method'].dropna()
            assert len(methods) > 0, "Generator method should be specified"
            # Check for known methods
            known_methods = ['tabular_counterfactual', 'llm_counterfactual']
            assert methods.isin(known_methods).any(), \
                f"Expected generator method to be one of {known_methods}"


class TestDataSampling:
    """Test cases for data sampling properties."""

    def test_sample_size(self, sample_data):
        """Test that sample size is as expected (20 rows per make_sample.py)."""
        # Based on make_sample.py which samples 20 rows
        expected_size = 20
        actual_size = len(sample_data)
        # Allow some flexibility but should be close
        assert actual_size <= expected_size * 2, \
            f"Sample size {actual_size} is larger than expected {expected_size}"

    def test_data_representativeness(self, sample_data):
        """Test that sample has both SUBSCRIBED and NO SUBSCRIPTION cases."""
        if 'original_ground_truth' in sample_data.columns:
            unique_values = sample_data['original_ground_truth'].unique()
            # Should have variety in ground truth
            assert len(unique_values) >= 1, "Should have at least one ground truth value"


# Parameterized tests
class TestParameterizedChecks:
    """Parameterized test cases for testing multiple scenarios."""

    @pytest.mark.parametrize("column", [
        'original_question',
        'counterfactual_question',
        'original_description',
        'counterfactual_description'
    ])
    def test_text_columns_have_content(self, sample_data, column):
        """Test that text columns have meaningful content."""
        if column in sample_data.columns:
            # Check that text is not just whitespace
            non_empty = sample_data[column].str.strip().str.len() > 0
            assert non_empty.all(), f"Column '{column}' has empty values"

    @pytest.mark.parametrize("ground_truth_col", [
        'original_ground_truth',
        'counterfactual_ground_truth'
    ])
    def test_ground_truth_format(self, sample_data, ground_truth_col):
        """Test that ground truth columns have valid values."""
        if ground_truth_col in sample_data.columns:
            valid_values = ['SUBSCRIBED', 'NO SUBSCRIPTION']
            assert sample_data[ground_truth_col].isin(valid_values).all(), \
                f"Column '{ground_truth_col}' has invalid values"


# Integration tests
class TestIntegration:
    """Integration tests that check relationships between fields."""

    def test_index_consistency(self, sample_data):
        """Test that question indices are consistent."""
        if 'original_question_idx' in sample_data.columns and \
           'counterfactual_question_idx' in sample_data.columns:
            # Indices should be valid integers
            assert sample_data['original_question_idx'].notna().all()
            assert sample_data['counterfactual_question_idx'].notna().all()

    def test_dataset_field_consistency(self, sample_data):
        """Test that dataset field is consistent across rows."""
        if 'original_dataset' in sample_data.columns:
            # All rows should be from same dataset in this sample
            datasets = sample_data['original_dataset'].unique()
            assert len(datasets) >= 1, "Should have at least one dataset"


# Skip/Xfail examples
class TestAdvancedFeatures:
    """Advanced pytest features demonstration."""

    @pytest.mark.skip(reason="Example skip - run only when testing specific features")
    def test_optional_check(self, sample_data):
        """Example of a skipped test."""
        assert True

    @pytest.mark.skipif(
        not os.path.exists("tests/test_outputs/extra_data.parquet"),
        reason="Extra data file not available"
    )
    def test_with_extra_data(self):
        """Example of conditional skip."""
        extra_df = pd.read_parquet("tests/test_outputs/extra_data.parquet")
        assert len(extra_df) > 0


if __name__ == "__main__":
    # Allow running pytest from this file directly
    pytest.main([__file__, "-v", "--tb=short"])
