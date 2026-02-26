import time
from typing import Any, Dict, Optional, List, Tuple
import torch
from vllm import LLM, SamplingParams
from src.utils import LLMConfig
from src.schema import CounterfactualDatabase, ModelInfo, Response
from src.utils import parse_response, get_messages, parse_message_to_harmony, extract_messages_using_harmony, split_on_cot_seperator

class ReferenceAnswerGenerator:
    """
    Generates reference predictions and explanations using an LLM (vLLM).
    Processes Parquet files containing original and counterfactual questions.
    """
    
    def __init__(self, config: Optional[LLMConfig] = None, 
                 llm_client: Optional[LLM] = None):
        """
        Initialize the generator
        1. Extract the model params, sampling params, and chat_template_kwargs params.
        2. Print
        3. Init using the model params

        Args:
            config: LLM configuration
            llm_client: Optional pre-initialized LLM client. If provided, setup_llm() is skipped.
        """
        self.config = config or LLMConfig()
        self.llm_client = llm_client
        self._harmony_context = None
        
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
        
        # Configure vLLM with same settings as tests.py
        try:
            self.llm_client = LLM(model=self.config.model_name, **self.model_params)
            print(f"Model loaded successfully with {torch.cuda.device_count()} GPUs!")
        except Exception as e:
            print(f"Error loading vLLM model: {e}")
            raise
    
    async def call_llm(self, prompt: str) -> Tuple[Optional[str], str, Optional[int], Optional[int], Optional[int]]:
        """
        Call the LLM with a single prompt using vLLM
        Legacy code. Just defaults to call_llm_batch now.
        Consider deleting this to clean up.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Tuple of (cot, response)
        """
        return (await self.call_llm_batch([prompt]))[0]
    
    async def call_llm_batch(self, prompts: List[str]) -> List[Tuple[Optional[str], str, Optional[int], Optional[int], Optional[int]]]:
        """
        Call the LLM with a batch of prompts using vLLM
        
        Args:
            prompts: List of input prompts
            
        Returns:
            A list of tuples: (CoT, response). Defaults to (None, response) if no mechanism for separation
        """
        if self.llm_client is None and not self.config.api_model:
            raise ValueError("LLM not initialized. Call setup_llm() first.")
        
        if self.config.api_model:
            start = time.time()
            outputs = await get_messages(prompts, system_prompt="", config=self.config)
            end = time.time()
            #generated_texts = [output.completion.strip() for output in outputs]
            generated_texts = [(r['choices'][0]['message'].get('reasoning', None), 
                                r['choices'][0]['message']['content'],
                                r['usage'].get('prompt_tokens',None),
                                r['usage'].get('completion_tokens_details',{}).get('reasoning_tokens',None),
                                r['usage'].get('completion_tokens',None)) for r in outputs]
            generation_time = end-start
            return generated_texts

        # if gpt-oss use harmony (v annoying from OpenAI). This works but outputs tuple
        # note the harmony imports are here since they are non-trivial to install. There is a small overhead here on the first batch only
        if self.config.model_name in ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]:
            if self._harmony_context is None:
                from openai_harmony import (
                    HarmonyEncodingName,
                    load_harmony_encoding,
                    Conversation,
                    Message,
                    Role,
                    SystemContent,
                    DeveloperContent,
                    ReasoningEffort
                )
                encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
                self._harmony_context = {"encoding": encoding, "role": Role}
            encoding = self._harmony_context["encoding"]
            Role = self._harmony_context['role']
            assert self.enable_reasoning in ["low", "medium", "high"]
            start_time = time.time()
            print("  Encoding prompts with harmony...")
            messages = [parse_message_to_harmony(p, self.enable_reasoning, encoding, Role) for p in prompts]
            stop_token_ids = encoding.stop_tokens_for_assistant_actions()
            self.sampling_params['stop_token_ids']=stop_token_ids
            sampling_parameters = SamplingParams(**self.sampling_params)
            print("  Generating responses...")
            outputs = self.llm_client.generate( # have to use harmony with tqdm
                messages,
                sampling_params=sampling_parameters,
                use_tqdm=True
            )
            end_time = time.time()
            generation_time = end_time - start_time
            print(f"Generation time: {generation_time}")
            output_tokens_list = [x.outputs[0].token_ids for x in outputs]
            entries = [encoding.parse_messages_from_completion_tokens(x, Role.ASSISTANT) for x in output_tokens_list]
            list_of_tuples = [extract_messages_using_harmony(x) for x in entries]
            return list_of_tuples

        # Configure sampling parameters
        sampling_parameters = SamplingParams(**self.sampling_params)
        start_time = time.time()
        chat_messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
        chat_params = {"sampling_params": sampling_parameters}
        if self.enable_reasoning is True:  # Only enable thinking when explicitly True
            chat_params["chat_template_kwargs"] = {"enable_thinking": True}
        elif self.enable_reasoning is False:  # Explicitly disable thinking when False
            chat_params["chat_template_kwargs"] = {"enable_thinking": False}
        # If None, don't set chat_template_kwargs (use model default)
        
        # Extract and generate text
        outputs = self.llm_client.chat(chat_messages, **chat_params)
        end_time = time.time()
        
        generated_texts = [output.outputs[0].text.strip() for output in outputs]
        
        # Calculate metrics
        total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        generation_time = end_time - start_time
        tokens_per_second = total_tokens / generation_time if generation_time > 0 else 0
        print(f"Generated {total_tokens} tokens for {len(prompts)} prompts in {generation_time:.2f}s ({tokens_per_second:.2f} tok/s)")

        # separate into cot, response
        cot_flags = getattr(self.config, "cot_flags", None)
        if cot_flags:
            cot_separator = cot_flags[-1]
            return [split_on_cot_seperator(x, cot_separator, cot_flags) for x in generated_texts]
        else:
            return [(None, x, None, None, None) for x in generated_texts]
    
    async def generate_single_response(self, question: str, max_tokens: Optional[int] = None) -> Tuple[Optional[str], str, Optional[int], Optional[int], Optional[int]]:
        """
        Never used. Running vLLM with batching is far superiour. Legacy code.
        """
        return await self.call_llm(question)
    
    async def process_parquet(self, input_path: str, output_path: str, max_batch_size: int = 256) -> CounterfactualDatabase:
        """
        Process Parquet file: load, generate reference answers, create counterfactual prompts, save.
        
        Args:
            input_path: Path to input Parquet file
            output_path: Path to save output Parquet file
            max_batch_size: Maximum batch size for LLM inference
            
        Returns:
            Enhanced database
        """
        print("="*60)
        print("GENERATING REFERENCE ANSWERS FROM PARQUET")
        print("="*60)
        print(f"Input: {input_path}")
        print(f"Output: {output_path}\n")
        print("Loading Parquet file...")
        db = CounterfactualDatabase.load_parquet(input_path)
        print(f"✓ Loaded {len(db.records)} records\n")
        
        # Identify unique prompts and build mappings
        unique_prompts, prompt_to_records = self._identify_unique_prompts(db)
        cots, responses, tokens_info = await self._generate_unique_answers(unique_prompts, max_batch_size)
        self._map_answers_to_records(db, prompt_to_records, responses, cots, tokens_info)
        # Generate counterfactual prompts (with/without explanation)
        self._generate_counterfactual_prompts(db)
        
        # Save enhanced Parquet
        print("\n" + "="*60)
        print("SAVING ENHANCED PARQUET")
        print("="*60)
        db.save_parquet(output_path)
        print(f"✓ Saved to {output_path}")
        return db
    
    def _identify_unique_prompts(self, db: CounterfactualDatabase) -> Tuple[Dict, Dict]:
        """Identify unique prompts and build mappings for deduplication."""
        print("="*60)
        print("IDENTIFYING UNIQUE PROMPTS")
        print("="*60)
        
        unique_prompts: Dict[Tuple, str] = {}
        prompt_to_records: Dict[Tuple, List[Tuple[int, str]]] = {}
        
        for record_idx, record in enumerate(db.records):
            # Process original question
            orig_key = (record.original_question.dataset, record.original_question.question_idx, record.original_question.answer_first)
            if orig_key not in unique_prompts:
                unique_prompts[orig_key] = record.original_question.question_prompt
                prompt_to_records[orig_key] = []
            prompt_to_records[orig_key].append((record_idx, 'original'))
            
            # Process counterfactual question
            cf_key = (record.original_question.dataset, record.counterfactual.question_idx, record.original_question.answer_first)
            if cf_key not in unique_prompts:
                unique_prompts[cf_key] = record.counterfactual.question_prompt
                prompt_to_records[cf_key] = []
            prompt_to_records[cf_key].append((record_idx, 'counterfactual'))
        
        print(f"Total records: {len(db.records)}")
        print(f"Total prompts (original + counterfactual): {len(db.records) * 2}")
        print(f"Unique prompts: {len(unique_prompts)}")
        print(f"Deduplication ratio: {len(unique_prompts) / (len(db.records) * 2):.2%}\n")
        
        return unique_prompts, prompt_to_records
    
    async def _generate_unique_answers(self, unique_prompts: Dict[Tuple, str], max_batch_size: int) -> Tuple[Dict[Tuple, Optional[str]], Dict[Tuple, str]]:
        """Generate reference answers for all unique prompts."""
        print("="*60)
        print("GENERATING REFERENCE ANSWERS")
        print("="*60)
        
        keys = list(unique_prompts.keys())
        prompts = list(unique_prompts.values())
        
        print(f"Processing {len(prompts)} unique prompts in batches of {max_batch_size}...")
        
        all_responses = []
        all_cots = []
        all_tokens_info = []
        num_batches = (len(prompts) + max_batch_size - 1) // max_batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * max_batch_size
            end_idx = min(start_idx + max_batch_size, len(prompts))
            batch_prompts = prompts[start_idx:end_idx]
            
            print(f"Batch {batch_idx + 1}/{num_batches}: {len(batch_prompts)} prompts...")
            batch_responses = await self.call_llm_batch(batch_prompts)
            all_responses.extend([r[1] for r in batch_responses])
            all_cots.extend([r[0] for r in batch_responses])

            all_tokens_info.extend([(r[2], r[3], r[4]) for r in batch_responses])
        
        print(f"✓ Generated {len(all_responses)} unique responses\n")
        
        return {key: cot for key, cot in zip(keys, all_cots)}, {key: response for key, response in zip(keys, all_responses)}, {key: tokens for key, tokens in zip(keys, all_tokens_info)}   
    
    def _map_answers_to_records(self, db: CounterfactualDatabase, prompt_to_records: Dict, responses: Dict, cots: Dict, tokens_info: Dict) -> None:
        """Map generated answers back to all records."""
        print("="*60)
        print("MAPPING ANSWERS TO RECORDS")
        print("="*60)

        # Pre-load dataset classes for all datasets in this database. Neat.
        unique_datasets = {r.original_question.dataset for r in db.records}
        dataset_classes = {name: db.dataset_class_map[name] for name in unique_datasets}

        # Create model info once. Use actual sampling params with defaults.
        thinking_value = self.enable_reasoning
        if thinking_value is not None:
            thinking_value = str(thinking_value)
        model_info = ModelInfo(
            model=self.config.model_name,
            temperature=self.sampling_params.get('temperature', 1.0),
            max_tokens=self.sampling_params.get('max_tokens', 16),
            thinking=thinking_value,
            seed=self.sampling_params.get('seed'),
            additional_params=getattr(self.config, 'additional_params', None)
        )

        for unique_key, record_locations in prompt_to_records.items():
            raw_response = responses.get(unique_key)
            cot = cots.get(unique_key)
            tokens = tokens_info.get(unique_key)

            if raw_response is None:
                print(f"Warning: No response for key {unique_key}")
                continue

            # Get dataset class from key (key is: dataset, question_idx, answer_first)
            dataset_name = unique_key[0]
            dataset_class = dataset_classes[dataset_name]

            # Parse the response using correct VALID_ANSWERS for this dataset
            parsed_response = parse_response(raw_response, dataset_class.VALID_ANSWERS)
            
            # Extract answer from parsed response
            answer = None
            if parsed_response and isinstance(parsed_response, dict):
                answer = parsed_response.get("answer")
            
            # Create Response object
            response_obj = Response(
                raw_response=raw_response,
                parsed_response=parsed_response,
                answer=answer,
                model_info=model_info,
                cot=cot,
                input_tokens=tokens[0],
                reasoning_tokens=tokens[1],
                output_tokens=tokens[2]
            )
            
            for record_idx, field_type in record_locations:
                record = db.records[record_idx]
                
                if field_type == 'original':
                    record.original_question.reference_response = response_obj
                else:  # counterfactual
                    record.counterfactual.reference_response = response_obj
        
        print(f"✓ Mapped answers to {len(db.records)} records\n")
    
    def _generate_counterfactual_prompts(self, db: CounterfactualDatabase) -> None:
        """Generate counterfactual prompts (with and without explanation)."""
        print("="*60)
        print("GENERATING COUNTERFACTUAL PROMPTS")
        print("="*60)

        # Pre-load dataset classes for all datasets in this database
        unique_datasets = {r.original_question.dataset for r in db.records}
        dataset_classes = {name: db.dataset_class_map[name] for name in unique_datasets}
        print(f"Loaded {len(dataset_classes)} dataset class(es): {list(dataset_classes.keys())}")

        parse_errors = 0

        for record in db.records:
            # Get the response object
            response = record.original_question.reference_response
            if not response or not response.parsed_response:
                parse_errors += 1
                continue

            # Use the parsed response directly
            parsed_answer = response.parsed_response

            if 'error' in parsed_answer:
                parse_errors += 1
                continue

            # Get questions
            original_question = record.original_question.question
            cf_question = record.counterfactual.question

            if not original_question or not cf_question:
                continue

            # Get dataset class for this record
            dataset_class = dataset_classes[record.original_question.dataset]

            # Generate both prompts using the record's dataset template
            try:
                record.counterfactual.prompt_with_explanation = dataset_class.create_counterfactual_prompt(
                    question=original_question,
                    question_explanation=parsed_answer,
                    counterfactual_question=cf_question,
                    answer_last=(not record.original_question.answer_first)
                )

                record.counterfactual.prompt_without_explanation = dataset_class.create_counterfactual_prompt_no_explanation(
                    question=original_question,
                    question_explanation=parsed_answer,
                    counterfactual_question=cf_question,
                    answer_last=(not record.original_question.answer_first)
                )
            except Exception as e:
                print(f"Error generating counterfactual prompts for {record.original_question.dataset}: {e}")
        
        print(f"✓ Generated counterfactual prompts for {len(db.records)} records")
        if parse_errors > 0:
            print(f"{parse_errors} parse errors encountered\n")