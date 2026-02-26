from typing import Optional, List, Tuple
from vllm import LLM, SamplingParams
from src.utils import (
    LLMConfig,
    parse_response,
    get_messages,
    parse_message_to_harmony,
    extract_messages_using_harmony,
    split_on_cot_seperator,
    create_testability_prompt,
    parse_testability_score
)
from src.schema import CounterfactualDatabase, ModelInfo, Response


class PredictorAnswerGenerator:
    """
    Generates predictor model responses for counterfactual prompts.
    
    This is a simple generator that:
    1. Reads prompts from CounterfactualInfo (prompt_with_explanation, prompt_without_explanation)
    2. Calls the predictor LLM for each prompt
    3. Stores raw responses back to CounterfactualInfo
    4. Stores predictor model metadata at record level
    """
    
    def __init__(self,
                 config: Optional[LLMConfig] = None,
                 llm_client: Optional[LLM] = None,
                 assess_testability: bool = False):
        """
        Initialize the predictor answer generator

        Args:
            config: LLM configuration
            llm_client: Optional pre-initialized LLM client. If provided, setup_llm() is skipped.
        """
        self.config = config or LLMConfig()
        self.llm_client = llm_client
        self.assess_testability = assess_testability
        self._harmony_context = None

        model_params = {}

        # Build model params
        if getattr(self.config, 'tensor_parallel_size', None) is not None:
            model_params['tensor_parallel_size'] = self.config.tensor_parallel_size
        if getattr(self.config, 'gpu_memory_utilization', None) is not None:
            model_params['gpu_memory_utilization'] = self.config.gpu_memory_utilization
        if getattr(self.config, 'max_model_len', None) is not None:
            model_params['max_model_len'] = self.config.max_model_len
        if getattr(self.config, 'dtype', None) is not None:
            model_params['dtype'] = self.config.dtype
        if getattr(self.config, 'limit_mm_per_prompt', None) is not None:
            model_params['limit_mm_per_prompt'] = self.config.limit_mm_per_prompt
        model_params['trust_remote_code'] = True
        self.model_params = model_params

        # Extract sampling parameters
        sampling_params = {}
        for param in ["max_tokens", "temperature", "seed"]:
            value = getattr(self.config, param, None)
            if value is not None:
                sampling_params[param] = value
        additional_params = getattr(self.config, "additional_params", None)
        if additional_params is not None:
            sampling_params.update(additional_params)
        self.sampling_params = sampling_params

        # chat_template_kwargs param. Store enable_reasoning separately (not a vLLM engine param).
        self.enable_reasoning = getattr(self.config, 'enable_reasoning', None)

        # Print summary of parameters
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
        
    def setup_llm(self):
        """Initialize the vLLM client"""
        print(f"Loading predictor model: {self.config.model_name}")
        
        try:
            self.llm_client = LLM(model=self.config.model_name, **self.model_params)
            print(f"Predictor model loaded successfully!")
        except Exception as e:
            print(f"Error loading vLLM model: {e}")
            raise
    
    async def process_parquet(
        self,
        input_path: str,
        output_path: str,
        max_batch_size: int = 50,
        predictor_repeats: int = 1,
        db: Optional[CounterfactualDatabase] = None,
        is_first_model: bool = True
    ) -> None:
        """
        Process Parquet file with predictor responses

        Args:
            input_path: Path to input Parquet file (output from reference answer generation)
            output_path: Path to save output Parquet file with predictor responses
            max_batch_size: Maximum batch size for LLM calls
            predictor_repeats: Number of times to run the predictor over each WITH-explanation prompt
            db: Optional pre-loaded database (should also be preloaded)
            is_first_model: Whether this is the first model (we use this to populate the main Response fields)
        """
        # Load or use provided database (mostly just use preloaded)
        if db is None:
            print(f"Loading database from: {input_path}")
            db = CounterfactualDatabase.load_parquet(input_path)
            print(f"Loaded {len(db.records)} records")
        else:
            print(f"Using provided database with {len(db.records)} records")

        # Pre-load dataset classes for all datasets in this database
        unique_datasets = {r.original_question.dataset for r in db.records}
        dataset_classes = {name: db.dataset_class_map[name] for name in unique_datasets}

        # Collect all prompts from counterfactuals
        prompts_with_exp = []
        prompts_without_exp = []
        counterfactual_indices = []  # Track which records have counterfactuals
        
        for idx, record in enumerate(db.records):
            cf = record.counterfactual
            if cf.prompt_with_explanation and cf.prompt_without_explanation:
                prompts_with_exp.append(cf.prompt_with_explanation)
                prompts_without_exp.append(cf.prompt_without_explanation)
                counterfactual_indices.append(idx)
        
        total_prompts = len(prompts_with_exp) * 2  # Each counterfactual has 2 prompts
        print(f"Found {len(counterfactual_indices)} counterfactuals with prompts")
        print(f"Total prompts to process: {total_prompts}")
        
        if total_prompts == 0:
            print("WARNING: No prompts found to process!")
            return

        # Assess testability if flag is enabled
        if self.assess_testability:
            print(f"\n{'='*80}")
            print("ASSESSING COUNTERFACTUAL TESTABILITY...")
            print(f"{'='*80}")
            print(f"Assessing {len(counterfactual_indices)} counterfactuals")

            testability_scores, testability_cots = await self._assess_testability_batch(
                db=db,
                counterfactual_indices=counterfactual_indices,
                max_batch_size=max_batch_size
            )

            # Store scores and CoTs in database
            for i, record_idx in enumerate(counterfactual_indices):
                db.records[record_idx].counterfactual.predictor_counterfactual_testability_score = testability_scores[i]
                db.records[record_idx].counterfactual.predictor_counterfactual_testability_cot = testability_cots[i]

            # Print statistics
            valid_scores = [s for s in testability_scores if s is not None]
            if valid_scores:
                avg_score = sum(valid_scores) / len(valid_scores)
                print(f"\nTestability Assessment Statistics:")
                print(f"  Valid scores: {len(valid_scores)}/{len(testability_scores)}")
                print(f"  Average score: {avg_score:.2f}")
                print(f"  Score range: [{min(valid_scores):.1f}, {max(valid_scores):.1f}]")
            else:
                print(f"\nWARNING: No valid testability scores extracted")

        # Generate responses WITH explanation - all repeats (when want to assess stability)
        all_answers_with_exp = []  # List of model answer lists, one per repeat
        responses_with_exp = []  # Store responses from first repeat (for top-level fields)

        for repeat_num in range(1, predictor_repeats + 1):
            print(f"\n{'='*80}")
            print(f"Generating responses WITH explanation...")
            print(f"Repeat {repeat_num}/{predictor_repeats}")
            print(f"{'='*80}")

            repeat_responses = await self._generate_batch(prompts_with_exp, max_batch_size)

            # Store responses for first repeat (needed for first model's top-level fields)
            if repeat_num == 1:
                responses_with_exp = repeat_responses

            # Extract the answer and store it in all_answers_with_exp (all answers)
            repeat_answers = []
            for i in range(len(repeat_responses)):
                cot, raw, *_ = repeat_responses[i]
                record_idx = counterfactual_indices[i]
                dataset_name = db.records[record_idx].original_question.dataset
                valid_answers = dataset_classes[dataset_name].VALID_ANSWERS
                parsed = parse_response(raw, valid_answers)
                answer = parsed.get("answer") if (parsed and isinstance(parsed, dict)) else None
                repeat_answers.append(answer)
            all_answers_with_exp.append(repeat_answers)

        # Generate responses WITHOUT explanation - single run of the model. No repeats
        print(f"\n{'='*80}")
        print("Generating responses WITHOUT explanation...")
        print("Single run (no repeats)")
        print(f"{'='*80}")
        responses_without_exp = await self._generate_batch(prompts_without_exp, max_batch_size)


        # Store WITHOUT answers in list format (to be consistent with WITH)
        all_answers_without_exp = []
        without_answers = []
        for i in range(len(responses_without_exp)):
            cot, raw, *_ = responses_without_exp[i]
            record_idx = counterfactual_indices[i]
            dataset_name = db.records[record_idx].original_question.dataset
            valid_answers = dataset_classes[dataset_name].VALID_ANSWERS
            parsed = parse_response(raw, valid_answers)
            answer = parsed.get("answer") if (parsed and isinstance(parsed, dict)) else None
            without_answers.append(answer)
        all_answers_without_exp.append(without_answers)
        
        # Store responses in counterfactuals
        print(f"\nStoring responses in database...")

        # Create model info once using actual sampling params (default to vLLM defaults)
        thinking_value = self.enable_reasoning
        if thinking_value is not None:
            thinking_value = str(thinking_value)

        model_info = ModelInfo(
            model=self.config.model_name,
            temperature=self.sampling_params.get('temperature', 1.0),
            max_tokens=self.sampling_params.get('max_tokens', 16),
            seed=self.sampling_params.get('seed'),
            additional_params=getattr(self.config, 'additional_params', None),
            thinking=thinking_value
        )

        for i, record_idx in enumerate(counterfactual_indices):
            record = db.records[record_idx]

            # Get first repeat responses (for top-level fields)
            cot_with, raw_with, input_tokens_w_exp, reasoning_tokens_w_exp, total_output_tokens_w_exp = responses_with_exp[i]
            cot_without, raw_without, input_tokens_wo_exp, reasoning_tokens_wo_exp, total_output_tokens_wo_exp = responses_without_exp[i]

            # Get valid answers for this record's dataset
            dataset_name = record.original_question.dataset
            valid_answers = dataset_classes[dataset_name].VALID_ANSWERS

            parsed_with = parse_response(raw_with, valid_answers)
            parsed_without = parse_response(raw_without, valid_answers)

            # Extract answers for top-level fields
            answer_with = None
            if parsed_with and isinstance(parsed_with, dict):
                answer_with = parsed_with.get("answer")

            answer_without = None
            if parsed_without and isinstance(parsed_without, dict):
                answer_without = parsed_without.get("answer")

            # Build predictor_answers and predictor_names for this model
            # WITH explanation: all repeats
            current_predictor_answers_with = [repeat_answers[i] for repeat_answers in all_answers_with_exp]
            current_predictor_names_with = [self.config.model_name] * len(current_predictor_answers_with)

            # WITHOUT explanation: single answer
            current_predictor_answers_without = [repeat_answers[i] for repeat_answers in all_answers_without_exp]
            current_predictor_names_without = [self.config.model_name] * len(current_predictor_answers_without)

            if is_first_model:
                # First model: Create new Response objects. else load it
                record.counterfactual.predictor_response_with_explanation = Response(
                    answer=answer_with,
                    cot=cot_with,
                    raw_response=raw_with,
                    parsed_response=parsed_with,
                    model_info=model_info,
                    predictor_answers=current_predictor_answers_with,
                    predictor_names=current_predictor_names_with,
                    input_tokens=input_tokens_w_exp,
                    reasoning_tokens=reasoning_tokens_w_exp,
                    output_tokens=total_output_tokens_w_exp
                )

                record.counterfactual.predictor_response_without_explanation = Response(
                    answer=answer_without,
                    cot=cot_without,
                    raw_response=raw_without,
                    parsed_response=parsed_without,
                    model_info=model_info,
                    predictor_answers=current_predictor_answers_without,
                    predictor_names=current_predictor_names_without,
                    input_tokens=input_tokens_wo_exp,
                    reasoning_tokens=reasoning_tokens_wo_exp,
                    output_tokens=total_output_tokens_wo_exp
                )
            else:
                # Subsequent models: Append to existing Response objects
                existing_with = record.counterfactual.predictor_response_with_explanation
                if existing_with:

                    # Convert numpy arrays to lists if needed (parquet deserialization)
                    if existing_with.predictor_answers is None:
                        existing_with.predictor_answers = []
                    elif hasattr(existing_with.predictor_answers, 'tolist'):
                        existing_with.predictor_answers = existing_with.predictor_answers.tolist()
                    if existing_with.predictor_names is None:
                        existing_with.predictor_names = []
                    elif hasattr(existing_with.predictor_names, 'tolist'):
                        existing_with.predictor_names = existing_with.predictor_names.tolist()

                    # extend with this models repeats (extend over append)
                    existing_with.predictor_answers.extend(current_predictor_answers_with)
                    existing_with.predictor_names.extend(current_predictor_names_with)

                # same without
                existing_without = record.counterfactual.predictor_response_without_explanation
                if existing_without:
                    if existing_without.predictor_answers is None:
                        existing_without.predictor_answers = []
                    elif hasattr(existing_without.predictor_answers, 'tolist'):
                        existing_without.predictor_answers = existing_without.predictor_answers.tolist()
                    if existing_without.predictor_names is None:
                        existing_without.predictor_names = []
                    elif hasattr(existing_without.predictor_names, 'tolist'):
                        existing_without.predictor_names = existing_without.predictor_names.tolist()

                    existing_without.predictor_answers.extend(current_predictor_answers_without)
                    existing_without.predictor_names.extend(current_predictor_names_without)
        
        # Save results
        print(f"\nSaving enhanced database to: {output_path}")
        db.save_parquet(output_path)
        
        # Print summary
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"Total records processed: {len(counterfactual_indices)}")
        print(f"Responses WITH explanation: {len(responses_with_exp)}")
        print(f"Responses WITHOUT explanation: {len(responses_without_exp)}")
        print(f"Output saved to: {output_path}")
    
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
                
                batch_responses = []
                for r in outputs:
                    try:
                        reasoning = r.get('choices', [{}])[0].get('message', {}).get('reasoning', None)
                        content = r.get('choices', [{}])[0].get('message', {}).get('content', '')
                        prompt_tokens = r.get('usage', {}).get('prompt_tokens', None)
                        reasoning_tokens = r.get('usage', {}).get('completion_tokens_details', {}).get('reasoning_tokens', None)
                        completion_tokens = r.get('usage', {}).get('completion_tokens', None)
                        batch_responses.append((reasoning, content, prompt_tokens, reasoning_tokens, completion_tokens))
                    except (KeyError, IndexError, TypeError) as e:
                        print(f"  WARNING: Error parsing API response: {e} on output {r}")
                        batch_responses.append((None, '', None, None, None))

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

            print(f"  Batch {batch_num}/{total_batches} complete")

        print(f"All batches complete ({len(all_responses)} responses generated)")
        return all_responses

    async def _assess_testability_batch(
        self,
        db: CounterfactualDatabase,
        counterfactual_indices: List[int],
        max_batch_size: int
    ) -> Tuple[List[Optional[float]], List[Optional[str]]]:
        """
        Assess testability scores for a batch of counterfactuals.

        For each counterfactual, evaluates how testable it is given the
        reference model's explanation using a 0-10 rubric.

        Args:
            db: CounterfactualDatabase containing records
            counterfactual_indices: List of record indices to assess
            max_batch_size: Maximum batch size for LLM calls

        Returns:
            Tuple of (scores, cots) where:
                - scores: List of testability scores (0-10 as floats, or None if parsing failed)
                - cots: List of chain-of-thought strings (or None if not available)
        """
        testability_prompts = []
        for record_idx in counterfactual_indices:
            record = db.records[record_idx]
            prompt = create_testability_prompt(record)
            testability_prompts.append(prompt)

        # Generate assessments using existing batch method
        raw_responses = await self._generate_batch(testability_prompts, max_batch_size)

        # Parse scores and extract CoTs from responses
        scores = []
        cots = []
        for i, (cot, raw_response, *_) in enumerate(raw_responses):
            score = parse_testability_score(raw_response)
            if score is None:
                print(f"  WARNING: Failed to parse score for record {counterfactual_indices[i]}")
            scores.append(score)
            cots.append(cot)

        return scores, cots