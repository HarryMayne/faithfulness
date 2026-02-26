#!/bin/bash
####################################################################
# Run the entire pipeline (over smaple size 20 at the moment)

# TODO:
#   - Convert this to pytest (new to me...?)
#   - Add tests for different parts of the predictor model pipeline
####################################################################

CUDA_VISIBLE_DEVICES="1,2"
export CUDA_VISIBLE_DEVICES="1,2"

export OPENROUTER_API_KEY="REDACTED_KEY"

# run the entire tabular data generator and store to test_outputs (will create parquets). Then make a small sample using the utils function.
#python -m src.counterfactual_generation.tabular_counterfactual_generation.tabular_to_text --output_dir tests/test_outputs
#python -m tests.utils.make_sample


####################################################################
# Basic pipeline
####################################################################
# run the reference model pipeline
# python -m src.reference_answer_generation.generate_reference_answers \
#       tests/test_outputs/sample_data.parquet \
#       --output-parquet tests/test_outputs/sample_responses.parquet \
#       --model Qwen/Qwen3-1.7B \
#       --batch-size 10000

# run the predictor model pipeline
python -m src.prediction_generation.generate_predictor_answers \
        tests/test_outputs/sample_responses.parquet \
        --output-parquet tests/test_outputs/sample_predictions.parquet \
        --model openrouter/google/gemma-3-27b-it openrouter/openai/gpt-oss-20b openrouter/qwen/qwen3-32b \
        --batch-size 10000

# run the analsysis script
python -m analysis_scripts.analyze_simulatability \
        tests/test_outputs/sample_predictions.parquet \
        --output tests/test_outputs/results_basic.csv
####################################################################




####################################################################
# Basic pipeline + predictor model repeats + assess testability
####################################################################
# run the reference model pipeline
# python -m src.reference_answer_generation.generate_reference_answers \
#       tests/test_outputs/sample_data.parquet \
#       --output-parquet tests/test_outputs/sample_responses.parquet \
#       --model Qwen/Qwen3-1.7B \
#       --batch-size 10000

# # run the predictor model pipeline
# python -m src.prediction_generation.generate_predictor_answers \
#         tests/test_outputs/sample_responses.parquet \
#         --output-parquet tests/test_outputs/sample_predictions.parquet \
#         --model Qwen/Qwen3-1.7B \
#         --batch-size 10000 \
#         --predictor-repeats 2 \
#         --assess-testability

# # run the analsysis script
# python -m analysis_scripts.analyze_simulatability \
#         tests/test_outputs/sample_predictions.parquet \
#         --output tests/test_outputs/results_basic_plus.csv
# ####################################################################





# ####################################################################
# # Try to add a new predictor model to the existing predictions
# ####################################################################
# echo "Try to add a new predictor model to the existing predictions"
# python -m src.prediction_generation.generate_predictor_answers \
#         tests/test_outputs/sample_predictions.parquet \
#         --output-parquet tests/test_outputs/sample_predictions.parquet \
#         --model Qwen/Qwen3-1.7B Qwen/Qwen3-1.7B \
#         --batch-size 10000 \
#         --predictor-repeats 2 \
#         --assess-testability



####################################################################
# API models pipeline [Dewi]
####################################################################


####################################################################


####################################################################
# Counterfactual generation pipeline [Dewi]
####################################################################


####################################################################