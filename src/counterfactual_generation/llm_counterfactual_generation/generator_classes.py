"""
This script contains the different generators (currently just BasicGenerator). It is loaded into `generate_counterfactuals.py`

Active generators:
    BasicGenerator: Puts everything into a single prompt and run this. All of the prompts for this are in `/methods`

Future generators:
    AgentGenerator: Launches an LLM server via vLLM and does an agent pipeline based on this.

Note that all generator classes must have a .run_generator() method that accepts a dataset and does the processing.
"""
import asyncio
import torch
from collections import Counter
from itertools import islice
from typing import Dict, Iterable, List, Sequence
from vllm import LLM, SamplingParams
import ast
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys
from google import genai
from openai import OpenAI, AsyncOpenAI
import os
import json
from src.schema import CounterfactualDatabase, OriginalQuestion, CounterfactualInfo, FaithfulnessRecord, ModelInfo
from typing import Any, Dict, Optional, Type, List, Tuple
from src.utils import (
    LLMConfig,
    filter_records_by_reference_answer,
    get_messages,
    parse_message_to_harmony,
    extract_messages_using_harmony,
    split_on_cot_seperator,
)
from src.templates.base import TabularDataset


from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
METHODS_DIR = BASE_DIR / "methods"

class BasicGenerator:
    """
    BasicGenerator Class

    Generates counterfactuals in a single LLM call.
    """
    
    def __init__(
        self,
        config: Optional[LLMConfig] = None, 
        max_batch_size: Optional[int] = 100,
        llm_client: Optional[LLM] = None,
        n_counterfactuals: int =3,
        allowed_method: list = [1,2,3,4],
        dataset_class:Type[TabularDataset]=None,
        ):
        """
        Initialize the BasicGenerator

        Args:
            dataset_filepath: filepath to a parquet
            config: LLM configuration
            llm_client: Optional pre-initialized LLM client. If provided, setup_llm() is skipped.
        """
        self.dataset = None
        self.config = config or LLMConfig()
        self.llm_client = llm_client
        self._harmony_context = None
        self.n_counterfactuals = n_counterfactuals
        self.allowed_method = allowed_method
        self.max_batch_size = max_batch_size
        self.dataset_class = dataset_class
        
        # Build model params dict only with explicitly provided (non-None) values. Contains the engine/init parameters only
        model_params: Dict[str, Any] = {}
        if not config.api_model:
            if getattr(config, 'tensor_parallel_size', None) is not None:
                model_params['tensor_parallel_size'] = config.tensor_parallel_size
            if getattr(config, 'gpu_memory_utilization', None) is not None:
                model_params['gpu_memory_utilization'] = config.gpu_memory_utilization
            if getattr(config, 'max_model_len', None) is not None:
                model_params['max_model_len'] = config.max_model_len
            if getattr(config, 'dtype', None) is not None:
                model_params['dtype'] = config.dtype
            if getattr(config, 'limit_mm_per_prompt', None) is not None:
                model_params['limit_mm_per_prompt'] = config.limit_mm_per_prompt
            model_params['trust_remote_code'] = True
        self.model_params = model_params

        # Extract the sampling parameters from the config. Add in the additional_params
        sampling_params: Dict[str, Any] = {}
        for p in ["max_tokens", "temperature", "seed"]:
            value = getattr(config, p, None)
            if value is not None:
                sampling_params[p] = value
        additional_params = getattr(config, "additional_params", None)
        if additional_params is not None:
            sampling_params.update(additional_params)
        self.sampling_params = sampling_params

        # chat_template_kwargs param. Store enable_reasoning separately (not a vLLM engine param).
        self.enable_reasoning = getattr(config, 'enable_reasoning', None)

        # print params
        print("=" * 60)
        print("LLM PARAMETER SUMMARY")
        print("=" * 60)
        print("MODEL PARAMETERS")
        print("-" * 60)
        if self.model_params:
            width = max(len(str(k)) for k in self.model_params.keys())
            for key, value in self.model_params.items():
                print(f"{str(key):<{width}}\t{value}")
        else:
            print("(none)")
        print()

        print("SAMPLING PARAMETERS")
        print("-" * 60)
        if self.sampling_params:
            width = max(len(str(k)) for k in self.sampling_params.keys())
            for key, value in self.sampling_params.items():
                print(f"{str(key):<{width}}\t\t{value}")
        else:
            print("(none)")
        print()

        print("REASONING SETTINGS")
        print("-" * 60)
        print(f"enable_reasoning\t{self.enable_reasoning}")
        print()

        # Only setup LLM if not provided
        if self.llm_client is None and not self.config.api_model:
            self.setup_llm()

    def get_device_info(self):
        """Get GPU device information"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"CUDA is available!")
            print(f"Using GPU: {gpu_name}")
            print(f"GPU Memory: {gpu_memory:.1f} GB")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            return device
        else:
            print("CUDA is not available. Using CPU.")
            return torch.device("cpu")

    def setup_llm(self):
        """
        Set up the vLLM client
        """
        print(f"Loading model: {self.config.model_name}")
        
        # Get device info
        device = self.get_device_info()
        
        # Configure vLLM 
        try:
            self.llm_client = LLM(model=self.config.model_name, **self.model_params)
            print(f"Model loaded successfully with {torch.cuda.device_count()} GPUs!")
        except Exception as e:
            print(f"Error loading vLLM model: {e}")
            raise    

    def make_prompt(self, question: str, n_counterfactuals: int = 3, allowed_method: list = [1,2,3,4]) -> str:
        """
        For a given instance in the dataframe, make the prompt for BasicGenerator

        Args:
            question (str): Original question to generate the counterfactuals from
            n_counterfactuals (int): Number of counterfactual examples to request from the model.
            allowed_methods (Iterable[int]): Method IDs whose instruction templates should be included.

        Returns:
            str: Prompt ready to send to the counterfactual generator.
        """
        # import the base files and filter to the allowed methods
        with open(METHODS_DIR / "counterfactual_task.txt", 'r') as f:
            counterfactual_task_template = f.read()
        with open(METHODS_DIR / "task_wrapper.txt", 'r') as f:
            task_wrapper_template = f.read()
        methods = {
            m: Path(METHODS_DIR / f"method_{m}.txt").read_text(encoding="utf-8")
            for m in range(1, 5)
        }

        # filter to the allowed methods and fill in templates
        methods = {k:v for k,v in methods.items() if k in allowed_method}
        counterfactual_task = counterfactual_task_template.replace("{n_counterfactuals}", str(n_counterfactuals))
        task_wrapper = task_wrapper_template.replace("{n_counterfactuals}", str(n_counterfactuals))

        # return the full prompt
        return counterfactual_task + "\n".join(methods.values()) + task_wrapper + question

    async def _generate_batch(self, prompts: List[str], max_batch_size: int) -> List[Tuple[Optional[str], str, Optional[int], Optional[int], Optional[int]]]:
        """
        Generate LLM responses for a batch of prompts
        
        Args:
            prompts: List of prompts to process
            max_batch_size: Maximum batch size for each LLM call
            
        Returns:
            List of (cot, response) tuples
        """
        cot_flags = getattr(self.config, "cot_flags", None)
        cot_separator = cot_flags[-1] if cot_flags else None
        all_responses: List[Tuple[Optional[str], str, Optional[int], Optional[int], Optional[int]]] = []
        total_batches = (len(prompts) + max_batch_size - 1) // max_batch_size
        
        for batch_idx in range(0, len(prompts), max_batch_size):
            batch_prompts = prompts[batch_idx:batch_idx + max_batch_size]
            batch_num = (batch_idx // max_batch_size) + 1
            
            print(f"  Processing batch {batch_num}/{total_batches} ({len(batch_prompts)} prompts)...")
            
            # Prepare messages for chat interface
            messages_list = [
                [{"role": "user", "content": prompt}]
                for prompt in batch_prompts
            ]

            if self.config.api_model:
                outputs = await get_messages(prompts=batch_prompts, system_prompt="", config=self.config)
                raw_texts = [output.completion.strip() for output in outputs]
                if cot_separator:
                    batch_responses = [split_on_cot_seperator(text, cot_separator, cot_flags) for text in raw_texts]
                else:
                    batch_responses = [(None, text, None, None, None) for text in raw_texts]


            # add harmony branch. mirrors the reference answer generation pipeline
            elif self.config.model_name in ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]:
                if self._harmony_context is None:
                    from openai_harmony import (
                        HarmonyEncodingName,
                        load_harmony_encoding,
                        Conversation,
                        Message,
                        Role,
                        SystemContent,
                        DeveloperContent,
                        ReasoningEffort,
                    )
                    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
                    self._harmony_context = {"encoding": encoding, "role": Role}
                encoding = self._harmony_context["encoding"]
                Role = self._harmony_context["role"]
                assert self.enable_reasoning in ["low", "medium", "high"]
                print("  Encoding prompts with harmony...")
                harmony_messages = [parse_message_to_harmony(p, self.enable_reasoning, encoding, Role) for p in batch_prompts]
                stop_token_ids = encoding.stop_tokens_for_assistant_actions()
                self.sampling_params["stop_token_ids"] = stop_token_ids
                harmony_sampling = SamplingParams(**self.sampling_params)
                print("  Generating responses...")
                outputs = self.llm_client.generate(
                    harmony_messages,
                    sampling_params=harmony_sampling,
                    use_tqdm=True,
                )
                token_lists = [item.outputs[0].token_ids for item in outputs]
                entries = [encoding.parse_messages_from_completion_tokens(tokens, Role.ASSISTANT) for tokens in token_lists]
                batch_responses = [extract_messages_using_harmony(entry) for entry in entries]

            else:
                sampling_params = SamplingParams(**self.sampling_params)
                chat_params = {"sampling_params": sampling_params}
                
                if self.enable_reasoning is True:
                    chat_params["chat_template_kwargs"] = {"enable_thinking": True}
                elif self.enable_reasoning is False:
                    chat_params["chat_template_kwargs"] = {"enable_thinking": False}
                
                outputs = self.llm_client.chat(messages_list, **chat_params)
                generated_texts = [output.outputs[0].text.strip() for output in outputs]
                if cot_separator:
                    batch_responses = [split_on_cot_seperator(text, cot_separator, cot_flags) for text in generated_texts]
                else:
                    batch_responses = [(None, text, None, None, None) for text in generated_texts]

            all_responses.extend(batch_responses)
            
            print(f"  ✓ Batch {batch_num}/{total_batches} complete")
        
        print(f"✓ All batches complete ({len(all_responses)} responses generated)")
        return all_responses


    def _parse_generator_outputs(self, responses: List[Tuple[Optional[str], str, Optional[int], Optional[int], Optional[int]]]) -> List[Tuple[Optional[str], str, dict]]:
        """
        Given a list of (cot, raw_json_str) tuples, parse each JSON payload.
        Falls back to empty counterfactuals/method on parse errors or missing keys.

        Args:
            responses: list of tuples

        Returns:
            list of tuples
        """
        parsed = []
        for cot, raw, *_ in responses:
            cleaned = raw if isinstance(raw, str) else ""
            cleaned = cleaned.strip()
            # Strip markdown code fences if present
            if cleaned.startswith("```"):
                parts = cleaned.split("```")
                # drop the opening fence segment
                cleaned = "".join(parts[1:]).strip()
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3].strip()
            # Trim to the first JSON object if extra text is present
            if "{" in cleaned and "}" in cleaned:
                start = cleaned.find("{")
                end = cleaned.rfind("}") + 1
                cleaned = cleaned[start:end]
            try:
                payload = json.loads(cleaned)
                if not isinstance(payload, dict):
                    raise ValueError("parsed payload is not a dict")
            except Exception as e:
                print(f"Warning: failed to parse generator output: {e}")
                payload = {}
            # ensure required keys exist
            if "counterfactuals" not in payload or "method" not in payload:
                payload = {
                    "counterfactuals": payload.get("counterfactuals", []) if isinstance(payload, dict) else [],
                    "method": payload.get("method", []) if isinstance(payload, dict) else [],
                }
            parsed.append((cot, raw, payload))
        return parsed

    def _build_counterfactual_db(
        self,
        dataset: pd.DataFrame,
        parsed_responses: List[Tuple[Optional[str], str, dict]],
        generator_model: str,
    ) -> CounterfactualDatabase:
        """
        Flatten parsed responses into a CounterfactualDatabase, one counterfactual per row.

        Args:
            dataset
        """
        db = CounterfactualDatabase()
        thinking_value = self.enable_reasoning
        if thinking_value is not None:
            thinking_value = str(thinking_value)
        model_info = ModelInfo(
            model=self.config.model_name,
            temperature=self.sampling_params.get('temperature'),
            max_tokens=self.sampling_params.get('max_tokens'),
            seed=self.sampling_params.get('seed'),
            additional_params=getattr(self.config, 'additional_params', None),
            thinking=thinking_value,
        )
        for row, parsed in zip(dataset.itertuples(index=False), parsed_responses):
            row_dict = row._asdict()
            cot, raw_text, payload = parsed
            original = OriginalQuestion(
                dataset=row_dict.get("original_dataset"),
                question=row_dict.get("original_question"),
                question_prompt = self.dataset_class.create_reference_prompt(
                    question=row_dict.get("original_question"),
                    answer_last = not row_dict.get("original_answer_first", False)),
                question_idx=row_dict.get("original_question_idx"),
                ground_truth=row_dict.get("original_ground_truth"),
                answer_first=row_dict.get("original_answer_first"),
                description=row_dict.get("original_description"),
                question_options=row_dict.get("original_question_options"),
                reference_response=None,
            )
            counterfactuals = payload.get("counterfactuals", []) if isinstance(payload, dict) else []
            methods = payload.get("method", []) if isinstance(payload, dict) else []
            for idx, cf_text in enumerate(counterfactuals):
                method_val = methods[idx] if isinstance(methods, list) and idx < len(methods) else methods
                cf_info = CounterfactualInfo(
                    generator_model=generator_model,
                    generator_method=method_val,
                    generator_model_cot=cot,
                    generator_model_raw=raw_text,
                    generator_model_info=model_info,
                    question=cf_text,
                    question_prompt = self.dataset_class.create_reference_prompt(
                        question=cf_text,
                        answer_last = not row_dict.get("original_answer_first", False)),
                )
                db.add_record(FaithfulnessRecord(original_question=original, counterfactual=cf_info))
        return db

    async def run_generator(self, dataset):
        """
        Every generator needs to have this method.
        Processes the prompts and feeds them into the LLM.
        
        Args:
            dataset
        """
        print("\n" + "="*80)
        print("COUNTERFACTUAL GENERATION STARTING")
        print("="*80)
        self.dataset=dataset.reset_index(drop=True)
        ix2q = {ix:q for ix, q in zip(self.dataset.index, self.dataset['original_question'])}
        ix2p = {ix:self.make_prompt(question=q, n_counterfactuals=self.n_counterfactuals, allowed_method=self.allowed_method) for ix, q in zip(self.dataset.index, self.dataset['original_question'])}

        # check length
        total_prompts = len(ix2p)
        setup_lines = [
            "",
            "=== Counterfactual Generation Setup ===",
            f"Generator model: {self.config.model_name}",
            f"Total questions loaded: {total_prompts:,}",
            f"Planned counterfactuals: {total_prompts * self.n_counterfactuals:,} ({self.n_counterfactuals} per question)",
            "",
        ]
        print("\n".join(setup_lines))
        print(f"Total prompts to process: {total_prompts}")
        if total_prompts == 0:
            print("WARNING: No prompts found to process!")
            return

        print(f"\n{'='*80}")
        print("Generating counterfactuals...")
        print(f"{'='*80}")

        # generate the responses using _generate_batch() and parse to extract JSON
        responses = await self._generate_batch(list(ix2p.values()), self.max_batch_size) 
        parsed_responses = self._parse_generator_outputs(responses)

        print(f"Responses returned: {len(responses)}")
        print(f"Parsed responses ready: {len(parsed_responses)}")

        # Store responses in counterfactuals
        print(f"\nStoring responses in database...")
        generated_db = self._build_counterfactual_db(self.dataset, parsed_responses, self.config.model_name)
        print(f"Counterfactual rows stored: {len(generated_db.records)}")

        # pass the generated_db back to the main function (note this is different from in Justin's classes)
        return generated_db
