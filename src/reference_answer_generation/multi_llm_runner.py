"""
Multi-LLM Experiment Runner

Orchestrates reference answer generation experiments across multiple LLMs.
All results are stored in a single Parquet file for each dataset, with one
column per model containing their responses.
"""

import asyncio
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict

from src.reference_answer_generation.reference_answer_generator import ReferenceAnswerGenerator
from src.schema import CounterfactualDatabase
from src.utils import LLMConfig, cleanup_after_model, get_model_name


@dataclass
class ExperimentConfig:
    """Configuration for multi-LLM experiment"""
    llm_configs: List[LLMConfig]
    input_parquet: str  # Path to input parquet file
    output_folder: str = "experiments/scaling_laws"  # Folder where timestamped run folder will be created
    
    def to_dict(self):
        """Serialize config for saving"""
        return {
            'llm_configs': [asdict(config) for config in self.llm_configs],
            'input_parquet': self.input_parquet,
            'output_folder': self.output_folder,
        }


class MultiLLMExperimentRunner:
    """
    Orchestrates reference answer generation across multiple LLMs.
    
    Workflow:
    1. For each dataset, load the base parquet with reference answers
    2. For each LLM, generate new reference answers and add as columns
    3. Save updated parquet with all models' responses
    4. Analysis is handled separately by MultiLLMAnalyzer
    """
    
    def __init__(self, config: ExperimentConfig, max_batch_size: int = 100):
        self.config = config
        os.makedirs(config.output_folder, exist_ok=True)
        
        # Create timestamped run folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_folder = Path(config.output_folder) / f"run_{timestamp}"
        self.run_folder.mkdir(parents=True, exist_ok=True)
        self.max_batch_size = max_batch_size
        
        # Save experiment config
        config_path = self.run_folder / "experiment_config.json"
        with open(config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        
        print(f"Experiment folder: {self.run_folder}")
    
    async def run(self):
        """
        Main experiment loop: run all LLMs on the input parquet
        """
        print("="*80)
        print("MULTI-LLM REFERENCE ANSWER EXPERIMENT")
        print("="*80)
        print(f"Input: {self.config.input_parquet}")
        print(f"Models: {len(self.config.llm_configs)}")
        print("="*80)
        
        # Load base parquet
        base_parquet = Path(self.config.input_parquet)
        
        if not base_parquet.exists():
            raise FileNotFoundError(f"Input parquet not found: {base_parquet}")
        
        # Load the database
        db = CounterfactualDatabase.load_parquet(base_parquet)
        print(f"Loaded {len(db.records)} records from {base_parquet}")
        
        # Detect unique datasets in the database
        unique_datasets = {r.original_question.dataset for r in db.records}
        if len(unique_datasets) == 1:
            dataset_name = next(iter(unique_datasets))
            dataset_class = db.dataset_class_map[dataset_name]
            print(f"Detected dataset: {dataset_name}")
            print(f"Using dataset class: {dataset_class.__name__}")
        else:
            dataset_name = "combined"
            print(f"Detected dataset: {dataset_name} ({len(unique_datasets)} dataset types)")
            for ds in sorted(unique_datasets):
                print(f"  - {ds}")
        print("="*80)

        output_parquet = self.run_folder / f"{dataset_name}_multi_model_responses.parquet"
        
        # Generate answers for each model and store in memory
        model_databases = {}  # model_name -> CounterfactualDatabase
        
        for llm_idx, llm_config in enumerate(self.config.llm_configs, 1):
            model_name = get_model_name(llm_config) # This is name only. Doesn't load config from this
            
            print(f"\n{'-'*80}")
            print(f"Model {llm_idx}/{len(self.config.llm_configs)}: {model_name}")
            print(f"Full name: {llm_config.model_name}")
            if hasattr(llm_config, 'enable_reasoning') and llm_config.enable_reasoning is not None:
                print(f"Reasoning mode: {llm_config.enable_reasoning}")
            print(f"{'-'*80}")
            
            # Generate answers for this model
            enhanced_db = await self._generate_model_responses(
                llm_config=llm_config,
                db=db
            )
            
            if enhanced_db is not None:
                model_databases[model_name] = enhanced_db
                print(f"✓ Completed {model_name}")
                # Partial save after each successful model
                self._save_multi_model_parquet(model_databases, output_parquet)
            else:
                print(f"✗ Failed {model_name}")
        
        # Combine all model databases into one parquet
        if model_databases:
            self._save_multi_model_parquet(model_databases, output_parquet)
            print(f"\n✓ Saved multi-model responses to: {output_parquet}")
        else:
            print(f"\n✗ No successful model runs")
        
        print("\n" + "="*80)
        print("✓ GENERATION COMPLETE")
        print("="*80)
        print(f"Results saved to: {self.run_folder}")
        print("\nTo analyze results, use MultiLLMAnalyzer class")
    
    async def _generate_model_responses(
        self,
        llm_config: LLMConfig,
        db: CounterfactualDatabase,
    ) -> Optional[CounterfactualDatabase]:
        """
        Generate responses for all records using one model.
        Simply uses ReferenceAnswerGenerator to process the database,
        which handles batching, thinking models, cleanup, etc.

        Args:
            llm_config: LLM configuration
            db: Database with records to process

        Returns:
            New database with reference_response filled in, or None on failure
        """
        generator = None
        try:
            # Create temporary files for this model's processing
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.parquet', delete=False) as tmp_in:
                input_path = tmp_in.name
            with tempfile.NamedTemporaryFile(mode='w', suffix='.parquet', delete=False) as tmp_out:
                output_path = tmp_out.name
            
            # Save current db to temp input
            db.save_parquet(input_path)
            
            # Use ReferenceAnswerGenerator to process. Got rid of dataset_class input as now on the fly
            generator = ReferenceAnswerGenerator(config=llm_config)
            
            # Process and get enhanced database back
            enhanced_db = await generator.process_parquet(input_path, output_path, max_batch_size=self.max_batch_size)
            
            # Cleanup temp files
            import os
            os.unlink(input_path)
            os.unlink(output_path)
            
            return enhanced_db
            
        except Exception as e:
            print(f"❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        finally:
            # CRITICAL: Cleanup GPU memory after each model
            cleanup_after_model(generator)
    
    def _save_multi_model_parquet(
        self,
        model_databases: Dict[str, CounterfactualDatabase],
        output_path: Path
    ):
        """
        Save multi-model responses to a parquet file.
        
        Combines all model databases into one parquet file.
        Each record has the counterfactual's reference_response set to that model's response.
        Models are distinguished by the ModelInfo in the Response object.
        
        Args:
            model_databases: Dict mapping model names to databases with responses
            output_path: Where to save the parquet file
        """
        print("\n" + "="*80)
        print("CREATING MULTI-MODEL PARQUET")
        print("="*80)
        
        # Create a new database combining all model databases
        combined_db = CounterfactualDatabase()
        
        for model_name, db in model_databases.items():
            print(f"  Adding {len(db.records)} records from: {model_name}")
            combined_db.records.extend(db.records)
        
        print(f"✓ Combined {len(combined_db.records)} total records ({len(model_databases)} models)")
        
        # Save to parquet
        combined_db.save_parquet(output_path)
        print(f"✓ Saved to {output_path}")
