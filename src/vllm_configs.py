"""
Config files for vllm. Edit as necessary.
"""
from src.utils import LLMConfig

max_tokens = 2000
max_tokens_reasoning = 20000
max_model_len_reasoning = 16384

## HOW YOU WANT TO CONFIGURE THESE MAY DIFFER DEPENDING ON YOUR SETUP ##

VLLM_CONFIGS = {
    # Example of the different types of parameters
    ###########################################################################
    # Qwen 3 series
    ###########################################################################
    "Qwen/Qwen3-0.6B": LLMConfig(
        model_name="Qwen/Qwen3-0.6B",                                   # init param
        tensor_parallel_size=1,                                         # init param
        gpu_memory_utilization=0.90,                                    # init param
        trust_remote_code=True,                                         # init param
        dtype="bfloat16",                                               # init param
        max_model_len=10000,                                            # init param
        max_tokens=10000,                                               # sampling param
        additional_params={                                             # sampling param
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
        },
        enable_reasoning=True,                                          # chat_template_kwargs param
        cot_flags=["<think>", "</think>"]                               # the cot flags. Note the final flag as to be the separator
    ),

    "Qwen/Qwen3-1.7B": LLMConfig(
        model_name="Qwen/Qwen3-1.7B",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.25,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=10000,
        max_tokens=10000,
        additional_params={
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
        },
        enable_reasoning=True,
        cot_flags=["<think>", "</think>"]
    ),

    "Qwen/Qwen3-4B": LLMConfig(
        model_name="Qwen/Qwen3-4B",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=10000,
        max_tokens=10000,
        additional_params={
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
        },
        enable_reasoning=True,
        cot_flags=["<think>", "</think>"]
    ),

    "Qwen/Qwen3-8B": LLMConfig(
        model_name="Qwen/Qwen3-8B",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=15000,
        max_tokens=15000,
        additional_params={
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
        },
        enable_reasoning=True,
        cot_flags=["<think>", "</think>"]
    ),

    "Qwen/Qwen3-14B": LLMConfig(
        model_name="Qwen/Qwen3-14B",
        tensor_parallel_size=2,
        gpu_memory_utilization=0.80,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=12000,
        max_tokens=10000,
        additional_params={
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
        },
        enable_reasoning=True,
        cot_flags=["<think>", "</think>"]
    ),

    "Qwen/Qwen3-32B": LLMConfig(
        model_name="Qwen/Qwen3-32B",
        tensor_parallel_size=2,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=16384,
        max_tokens=12000,
        additional_params={
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
        },
        enable_reasoning=True,
        cot_flags=["<think>", "</think>"]
    ),

    "nvidia/Llama-3_3-Nemotron-Super-49B-v1_5": LLMConfig(
        model_name="nvidia/Llama-3_3-Nemotron-Super-49B-v1_5",
        max_tokens=max_tokens_reasoning,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.70,
        max_model_len=max_model_len_reasoning,
        trust_remote_code=True,
        dtype="auto"
    ),
    
    "microsoft/MediPhi-Instruct": LLMConfig(
        model_name="microsoft/MediPhi-Instruct",
        max_tokens=max_tokens,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.70,
        max_model_len=4096,
        trust_remote_code=True
    ),
    
    "zai-org/GLM-4-32B-0414": LLMConfig(
        model_name="zai-org/GLM-4-32B-0414",
        max_tokens=max_tokens,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.70,
        max_model_len=4096,
        trust_remote_code=True
    ),
    
    # Google Models
    "google/medgemma-27b-text-it": LLMConfig(
        model_name="google/medgemma-27b-text-it",
        max_tokens=max_tokens,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.70,
        max_model_len=8192,
        dtype="bfloat16",
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 0}
    ),

    ###########################################################################
    # Gemma 3 series
    ###########################################################################
    "google/gemma-3-1b-it": LLMConfig(
        model_name="google/gemma-3-1b-it",
        max_tokens=max_tokens,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        max_model_len=10000,
        dtype="bfloat16",
        trust_remote_code=True,
        additional_params={
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 64,
        },
    ),

    "google/gemma-3-4b-it": LLMConfig(
        model_name="google/gemma-3-4b-it",
        max_tokens=max_tokens,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        max_model_len=10000,
        dtype="bfloat16",
        trust_remote_code=True,
        additional_params={
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 64,
        },
    ),

    "google/gemma-3-12b-it": LLMConfig(
        model_name="google/gemma-3-12b-it",
        max_tokens=max_tokens,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        max_model_len=10000,
        dtype="bfloat16",
        trust_remote_code=True,
        additional_params={
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 64,
        },
    ),

    "google/gemma-3-27b-it": LLMConfig(
        model_name="google/gemma-3-27b-it",
        max_tokens=max_tokens,
        tensor_parallel_size=2, # can be run on a single GPU. Way faster to do on 2
        gpu_memory_utilization=0.90,
        max_model_len=5000, # unsure how long our prompts are. Probably fine.
        dtype="bfloat16",
        trust_remote_code=True,
        additional_params={
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 64,
        },
    ),
    
    "google/gemma-3-27b-it-1gpu":LLMConfig(
        model_name="google/gemma-3-27b-it",
        tensor_parallel_size=1,
        max_tokens=2000,
        max_model_len=4098,
        trust_remote_code=True,
        dtype="bfloat16",
        limit_mm_per_prompt={"image": 0} # this leads to odd behaviour. Get a lot of corrupted outputs
    ),

    ###########################################################################
    # Llama 2 series
    ###########################################################################

    "meta-llama/Llama-2-7b-chat-hf": LLMConfig(
        model_name="meta-llama/Llama-2-7b-chat-hf",
        max_tokens=max_tokens,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        dtype="bfloat16",
        trust_remote_code=True,
        additional_params={
            "temperature": 0.9,
            "top_p": 0.6,
        },
    ),
    "meta-llama/Llama-2-13b-chat-hf": LLMConfig(
        model_name="meta-llama/Llama-2-13b-chat-hf",
        max_tokens=max_tokens,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        dtype="bfloat16",
        trust_remote_code=True,
        additional_params={
            "temperature": 0.9,
            "top_p": 0.6,
        },
    ),
    "meta-llama/Llama-2-70b-chat-hf": LLMConfig(
        model_name="meta-llama/Llama-2-70b-chat-hf",
        max_tokens=max_tokens,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        dtype="bfloat16",
        trust_remote_code=True,
        additional_params={
            "temperature": 0.9,
            "top_p": 0.6,
        },
    ),

    ###########################################################################
    # Llama 3 series
    ###########################################################################

    "meta-llama/Meta-Llama-3-8B-Instruct": LLMConfig(
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        max_tokens=max_tokens,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        max_model_len=8192,
        dtype="bfloat16",
        trust_remote_code=True,
        additional_params={
            "temperature": 0.6,
            "top_p": 0.9,
        },
    ),
    "meta-llama/Meta-Llama-3-70B-Instruct": LLMConfig(
        model_name="meta-llama/Meta-Llama-3-70B-Instruct",
        max_tokens=max_tokens,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.90,
        max_model_len=8192,
        dtype="bfloat16",
        trust_remote_code=True,
        additional_params={
            "temperature": 0.6,
            "top_p": 0.9,
        },
    ),
    
    # Qwen Models
    "Qwen/Qwen3-32B-reasoning": LLMConfig(
        model_name="Qwen/Qwen3-32B",
        max_tokens=max_tokens_reasoning,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.90,
        max_model_len=max_model_len_reasoning,
        trust_remote_code=True,
        enable_reasoning=True
    ),
    
    "Qwen/Qwen3-32B-direct": LLMConfig(
        model_name="Qwen/Qwen3-32B",
        max_tokens=max_tokens,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.70,
        max_model_len=4096,
        trust_remote_code=True,
        enable_reasoning=False
    ),
    
    "Qwen/Qwen3-30B-A3B-Instruct-2507": LLMConfig(
        model_name="Qwen/Qwen3-30B-A3B-Instruct-2507",
        max_tokens=max_tokens,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.70,
        max_model_len=3072,
        trust_remote_code=True
    ),

    "openrouter/google/gemini-2.5-flash-lite": LLMConfig(
        model_name="google/gemini-2.5-flash-lite",
        api_model=True,temperature=0,
        max_tokens=max_tokens,
        seed=666
    ),

    "openrouter/google/gemini-2.0-flash-lite-001": LLMConfig(
        model_name="google/gemini-2.0-flash-lite-001",
        api_model=True,
        temperature=0,
        max_tokens=max_tokens,
        seed=666
    ),

    "openrouter/qwen/qwen3-8b": LLMConfig(
        model_name="qwen/qwen3-8b",
        api_model=True,
        temperature=0,
        max_tokens=max_tokens,
        seed=666
    ),

   "openrouter/qwen/qwen3-14b": LLMConfig(
        model_name="qwen/qwen3-14b",
        api_model=True,
        temperature=0,
        max_tokens=max_tokens,
        seed=666
    ), 
    ############################################################
    # PREDICTOR MODELS FOR COT EXPERIMENTS
    ############################################################
    "openrouter/qwen/qwen3-32b": LLMConfig( # QWEN 3 32B PREDICTOR
        model_name="qwen/qwen3-32b",
        api_model=True,
        temperature=0.6,
        #top_p=0.95,
        max_tokens=max_tokens,
        seed=666,
        #enable_reasoning="medium",
    ),
    "openrouter/google/gemma-3-27b-it": LLMConfig( # QWEN 3 32B PREDICTOR
        model_name="google/gemma-3-27b-it",
        api_model=True,
        temperature=1,
        #top_p=0.95,
        max_tokens=max_tokens,
        seed=666
    ),
    "openrouter/openai/gpt-oss-20b": LLMConfig( # QWEN 3 32B PREDICTOR
        model_name="openai/gpt-oss-20b",
        api_model=True,
        temperature=0.6,
        max_tokens=max_tokens,
        seed=666,
        enable_reasoning='medium',
    ),

    ############################################################

    "openrouter/anthropic/claude-3.5-haiku": LLMConfig(
        model_name="anthropic/claude-3.5-haiku",
        api_model=True,
        temperature=0,
        max_tokens=max_tokens,
        seed=666
    ),

    "openrouter/qwen/qwen3-4b:free": LLMConfig(
        model_name="qwen/qwen3-4b:free",
        api_model=True,
        temperature=0,
        max_tokens=max_tokens,
        seed=666
    ),

    "openrouter/deepseek/deepseek-r1-distill-qwen-32b": LLMConfig(
        model_name="deepseek/deepseek-r1-distill-qwen-32b",
        api_model=True,
        temperature=0,
        max_tokens=max_tokens,
        seed=666
    ),

    "openrouter/openai/gpt-4.1-nano": LLMConfig(
        model_name="openai/gpt-4.1-nano",
        api_model=True,
        temperature=0,
        max_tokens=5000,
        seed=666
    ),

    "openrouter/openai/gpt-5-low": LLMConfig(
        model_name="openai/gpt-5",
        api_model=True,
        temperature=0,
        max_tokens=2000,
        enable_reasoning="low"),

    "openrouter/openai/gpt-5-medium": LLMConfig(
        model_name="openai/gpt-5",
        api_model=True,
        temperature=0,
        max_tokens=10000,
        enable_reasoning="medium"),

    "openrouter/openai/gpt-5-high": LLMConfig(
        model_name="openai/gpt-5",
        api_model=True,
        temperature=0,
        max_tokens=2000,
        enable_reasoning="high"),   

    "openrouter/openai/gpt-5-mini": LLMConfig(
        model_name="openai/gpt-5-mini",
        api_model=True,
        max_tokens=10000,
    ),


    "openrouter/openai/gpt-5-nano": LLMConfig(
        model_name="openai/gpt-5-nano",
        api_model=True,
        max_tokens=10000,
    ),

    "openrouter/anthropic/claude-haiku-4.5": LLMConfig(
        model_name="anthropic/claude-haiku-4.5",
        api_model=True,
        max_tokens=10000,
    ),

    "openrouter/anthropic/claude-sonnet-4.5": LLMConfig(
        model_name="anthropic/claude-sonnet-4.5",
        api_model=True,
        max_tokens=10000,
    ),

    "openrouter/anthropic/claude-opus-4.5": LLMConfig(
        model_name="anthropic/claude-opus-4.5",
        api_model=True,
        max_tokens=10000,
    ),


    "openrouter/anthropic/claude-opus-4.5-low": LLMConfig(
        model_name="anthropic/claude-opus-4.5",
        api_model=True,
        max_tokens=10000,
        enable_reasoning="low",
    ),

    "openrouter/anthropic/claude-opus-4.5-high": LLMConfig(
        model_name="anthropic/claude-opus-4.5",
        api_model=True,
        max_tokens=10000,
        enable_reasoning="high",
    ),

   "openrouter/anthropic/claude-haiku-4.5-low": LLMConfig(
        model_name="anthropic/claude-haiku-4.5",
        api_model=True,
        max_tokens=10000,
        enable_reasoning="low"
    ),

    "openrouter/google/gemini-3-flash-preview": LLMConfig(
        model_name="google/gemini-3-flash-preview",
        api_model=True,
        max_tokens=10000,
    ),

    "openrouter/google/gemini-3-pro-preview": LLMConfig(
        model_name="google/gemini-3-pro-preview",
        api_model=True,
        max_tokens=10000,
    ),

    "openrouter/google/gemini-3-pro-preview-low": LLMConfig(
        model_name="google/gemini-3-pro-preview",
        api_model=True,
        max_tokens=10000,
        enable_reasoning="low",
    ),

    "openrouter/google/gemini-3-flash-preview-low": LLMConfig(
        model_name="google/gemini-3-flash-preview",
        api_model=True,
        max_tokens=10000,
        enable_reasoning="low",
    ),


    "openrouter/openai/gpt-5-nano-low": LLMConfig(
        model_name="openai/gpt-5-nano",
        api_model=True,
        max_tokens=10000,
        enable_reasoning="low",
    ),


  "openrouter/openai/gpt-5-mini-low": LLMConfig(
        model_name="openai/gpt-5-mini",
        api_model=True,
        max_tokens=10000,
        enable_reasoning="low",
    ),

    "openrouter/openai/gpt-5.2-none": LLMConfig(
        model_name="openai/gpt-5.2",
        api_model=True,
        max_tokens=10000,
        enable_reasoning="none",
    ),
    ###########################################################################
    # Proposed models for experiments
    ###########################################################################

    "openrouter/openai/gpt-4o-mini": LLMConfig(
        model_name="openai/gpt-4o-mini",
        api_model=True,
        max_tokens=10000,
    ),
    
    "openrouter/google/gemini-2.0-flash-001": LLMConfig(
        model_name="google/gemini-2.0-flash-001",
        api_model=True,
        max_tokens=10000,
    ),

    "openrouter/openai/gpt-5.2-low": LLMConfig(
        model_name="openai/gpt-5.2",
        api_model=True,
        max_tokens=10000,
        enable_reasoning="low",
    ),

    "openrouter/openai/gpt-5.2-medium": LLMConfig(
        model_name="openai/gpt-5.2",
        api_model=True,
        max_tokens=10000,
        enable_reasoning="medium",
    ),

    "openrouter/openai/gpt-5.2-high": LLMConfig(
        model_name="openai/gpt-5.2",
        api_model=True,
        max_tokens=10000,
        enable_reasoning="high",
    ),


  "openrouter/openai/gpt-5-mini-medium": LLMConfig(
        model_name="openai/gpt-5-mini",
        api_model=True,
        max_tokens=10000,
        enable_reasoning="medium",
    ),



    "openrouter/openai/gpt-5-nano-medium": LLMConfig(
        model_name="openai/gpt-5-nano",
        api_model=True,
        max_tokens=10000,
        enable_reasoning="medium",
    ),

 "openrouter/anthropic/claude-haiku-4.5-medium": LLMConfig(
        model_name="anthropic/claude-haiku-4.5",
        api_model=True,
        max_tokens=10000,
        enable_reasoning="medium"
    ),


    "openrouter/anthropic/claude-sonnet-4.5-none": LLMConfig(
        model_name="anthropic/claude-sonnet-4.5",
        api_model=True,
        max_tokens=10000,
        enable_reasoning="none",
    ),

    "openrouter/anthropic/claude-sonnet-4.5-low": LLMConfig(
        model_name="anthropic/claude-sonnet-4.5",
        api_model=True,
        max_tokens=10000,
        enable_reasoning="low",
    ),

    "openrouter/anthropic/claude-sonnet-4.5-medium": LLMConfig(
        model_name="anthropic/claude-sonnet-4.5",
        api_model=True,
        max_tokens=10000,
        enable_reasoning="medium",
    ),

    "openrouter/anthropic/claude-sonnet-4.5-high": LLMConfig(
        model_name="anthropic/claude-sonnet-4.5",
        api_model=True,
        max_tokens=10000,
        enable_reasoning="high",
    ),


    "openrouter/anthropic/claude-opus-4.5-medium": LLMConfig(
        model_name="anthropic/claude-opus-4.5",
        api_model=True,
        max_tokens=10000,
        enable_reasoning="medium",
    ),


    "openrouter/google/gemini-3-pro-preview-medium": LLMConfig(
        model_name="google/gemini-3-pro-preview",
        api_model=True,
        max_tokens=10000,
        enable_reasoning="medium",
    ),

    "openrouter/google/gemini-3-flash-preview-medium": LLMConfig(
        model_name="google/gemini-3-flash-preview",
        api_model=True,
        max_tokens=10000,
        enable_reasoning="medium",
    ),



    ###########################################################################

    "openrouter/google/gemini-3-flash-preview-high": LLMConfig(
        model_name="google/gemini-3-flash-preview",
        api_model=True,
        max_tokens=5000,
        enable_reasoning="high",
    ),

    ###########################################################################
    # gpt-oss via vllm. Note flags are not required here as custom processing.
    ###########################################################################
    "openai/gpt-oss-20b": LLMConfig(
        model_name="openai/gpt-oss-20b",
        api_model=False,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90, # fine for the 20b on 1 H100
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=100000,
        max_tokens=131072,
        seed=666,
        additional_params={
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
        },
        enable_reasoning="medium" # "low"/"medium"/"high"
    ),

    "openai/gpt-oss-120b": LLMConfig(
        model_name="openai/gpt-oss-120b",
        api_model=False,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.90, # fine for 2 H100s
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=10000,
        max_tokens=131072,
        seed=666,
        additional_params={
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
        },
        enable_reasoning="low" # "low"/"medium"/"high"
    )
}
