"""
LLM Fine-tuning Script
This script provides functionality to fine-tune various LLM models using Hugging Face Transformers.
Supports multiple model types and training configurations.
"""

import os
import json
import torch
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

import pandas as pd
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    get_linear_schedule_with_warmup
)
from transformers.integrations import TensorBoardCallback
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
from accelerate import Accelerator
import wandb


@dataclass
class FineTuningConfig:
    """Configuration class for fine-tuning parameters"""
    
    # Model and data paths
    model_name: str = "microsoft/DialoGPT-medium"
    dataset_path: str = "data/train.json"
    validation_dataset_path: Optional[str] = None
    output_dir: str = "fine_tuned_model"
    
    # Training parameters
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    
    # Tokenization
    max_length: int = 512
    padding_side: str = "left"
    truncation: bool = True
    
    # LoRA configuration
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: ["c_attn", "c_proj"])  # Default for GPT-2/DialoGPT
    
    # Quantization
    use_4bit: bool = True
    use_8bit: bool = False
    
    # Training settings
    save_strategy: str = "steps"
    save_steps: int = 500
    eval_strategy: str = "steps"
    eval_steps: int = 500
    logging_steps: int = 100
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    
    # Monitoring
    use_wandb: bool = False
    wandb_project: str = "llm-fine-tuning"
    wandb_run_name: Optional[str] = None
    
    # Hardware
    fp16: bool = True
    bf16: bool = False
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = False


class LLMFineTuner:
    """Main class for fine-tuning LLM models"""
    
    def __init__(self, config: FineTuningConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.eval_dataset = None
        self.trainer = None
        
        # Initialize accelerator for distributed training
        self.accelerator = Accelerator()
        
        # Set up logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging and monitoring"""
        if self.config.use_wandb and self.accelerator.is_main_process:
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name or f"finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=self.config.__dict__
            )
    
    def load_tokenizer_and_model(self):
        """Load tokenizer and model"""
        print(f"Loading tokenizer and model: {self.config.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            padding_side=self.config.padding_side,
            trust_remote_code=True
        )
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with quantization if specified
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if self.config.fp16 else torch.float32,
        }
        
        if self.config.use_4bit:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif self.config.use_8bit:
            model_kwargs["load_in_8bit"] = True
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs
        )
        
        # Prepare model for training
        if self.config.use_4bit or self.config.use_8bit:
            self.model = prepare_model_for_kbit_training(self.model)
            
        # Apply LoRA if specified
        if self.config.use_lora:
            self._apply_lora()
            
        print("Model and tokenizer loaded successfully!")
    
    def _get_target_modules(self):
        """Automatically detect target modules for LoRA based on model architecture"""
        print("🔍 Detecting target modules for LoRA...")
        
        # Get all module names
        module_names = []
        for name, module in self.model.named_modules():
            module_names.append(name)
        
        # Print model architecture for debugging
        print(f"📋 Model architecture: {self.model.config.architectures}")
        
        # Define target modules for different model types
        target_modules_map = {
            # GPT-2 and DialoGPT models
            "GPT2LMHeadModel": ["c_attn", "c_proj"],
            "GPT2Model": ["c_attn", "c_proj"],
            
            # LLaMA models
            "LlamaForCausalLM": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "LlamaModel": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            
            # Mistral models
            "MistralForCausalLM": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            
            # Falcon models
            "FalconForCausalLM": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
            
            # BLOOM models
            "BloomForCausalLM": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
            
            # OPT models
            "OPTForCausalLM": ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
            
            # Default fallback
            "default": ["q_proj", "v_proj", "k_proj", "o_proj"]
        }
        
        # Get model architecture
        model_type = None
        if hasattr(self.model.config, 'architectures') and self.model.config.architectures:
            model_type = self.model.config.architectures[0]
        else:
            model_type = type(self.model).__name__
        
        # Get target modules for this model type
        target_modules = target_modules_map.get(model_type, target_modules_map["default"])
        
        # Verify that target modules exist in the model
        existing_modules = []
        for target_module in target_modules:
            found = False
            for module_name in module_names:
                if target_module in module_name:
                    found = True
                    break
            if found:
                existing_modules.append(target_module)
        
        # If no modules found with the predefined list, try to find attention modules automatically
        if not existing_modules:
            print("⚠️  No predefined target modules found. Searching for attention modules...")
            
            # Look for common attention module patterns
            attention_patterns = [
                "attn", "attention", "self_attn", "c_attn", "query", "key", "value",
                "q_proj", "k_proj", "v_proj", "qkv", "query_key_value"
            ]
            
            for pattern in attention_patterns:
                for module_name in module_names:
                    if pattern in module_name.lower() and "weight" not in module_name:
                        base_name = module_name.split('.')[-1]
                        if base_name not in existing_modules:
                            existing_modules.append(base_name)
        
        # If still no modules found, use linear layers as fallback
        if not existing_modules:
            print("⚠️  No attention modules found. Using linear layers as fallback...")
            for module_name in module_names:
                if "linear" in module_name.lower() or "dense" in module_name.lower():
                    base_name = module_name.split('.')[-1]
                    if base_name not in existing_modules:
                        existing_modules.append(base_name)
                        if len(existing_modules) >= 4:  # Limit to 4 modules
                            break
        
        print(f"✅ Found target modules: {existing_modules}")
        
        # Print some example module names for debugging
        print("📝 Sample module names from model:")
        for i, name in enumerate(module_names[:10]):
            print(f"   {name}")
        if len(module_names) > 10:
            print(f"   ... and {len(module_names) - 10} more modules")
        
        return existing_modules if existing_modules else ["c_attn"]  # Fallback
    
    def _apply_lora(self):
        """Apply LoRA (Low-Rank Adaptation) to the model"""
        print("Applying LoRA configuration...")
        
        # Auto-detect target modules if the config ones don't exist
        try:
            # First try with configured target modules
            target_modules = self.config.lora_target_modules
        except:
            target_modules = self._get_target_modules()
        
        # If configured modules don't work, auto-detect
        if target_modules == ["q_proj", "v_proj", "k_proj", "o_proj"]:
            detected_modules = self._get_target_modules()
            if detected_modules != target_modules:
                print(f"🔄 Switching from configured modules {target_modules} to detected modules {detected_modules}")
                target_modules = detected_modules
        
        try:
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=target_modules,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            print(f"✅ LoRA applied successfully with target modules: {target_modules}")
            
        except Exception as e:
            print(f"❌ Error applying LoRA with modules {target_modules}: {str(e)}")
            print("🔍 Attempting auto-detection of target modules...")
            
            # Fallback to auto-detection
            detected_modules = self._get_target_modules()
            print(f"🔄 Trying with detected modules: {detected_modules}")
            
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=detected_modules,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            print(f"✅ LoRA applied successfully with auto-detected modules: {detected_modules}")
    
    def load_datasets(self):
        """Load and preprocess datasets"""
        print(f"Loading dataset from: {self.config.dataset_path}")
        
        # Load training dataset
        if self.config.dataset_path.endswith('.json'):
            with open(self.config.dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            train_df = pd.DataFrame(data)
        elif self.config.dataset_path.endswith('.csv'):
            train_df = pd.read_csv(self.config.dataset_path)
        else:
            # Try loading as Hugging Face dataset
            dataset = load_dataset(self.config.dataset_path)
            train_df = dataset['train'].to_pandas()
        
        self.train_dataset = Dataset.from_pandas(train_df)
        
        # Load validation dataset if provided
        if self.config.validation_dataset_path:
            if self.config.validation_dataset_path.endswith('.json'):
                with open(self.config.validation_dataset_path, 'r', encoding='utf-8') as f:
                    eval_data = json.load(f)
                eval_df = pd.DataFrame(eval_data)
            elif self.config.validation_dataset_path.endswith('.csv'):
                eval_df = pd.read_csv(self.config.validation_dataset_path)
            else:
                eval_dataset = load_dataset(self.config.validation_dataset_path)
                eval_df = eval_dataset['validation'].to_pandas()
                
            self.eval_dataset = Dataset.from_pandas(eval_df)
        else:
            # Split training dataset for validation
            train_test_split = self.train_dataset.train_test_split(test_size=0.1)
            self.train_dataset = train_test_split['train']
            self.eval_dataset = train_test_split['test']
        
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.eval_dataset)}")
    
    def preprocess_function(self, examples):
        """Preprocess function for tokenization"""
        # This is a generic preprocessing function
        # You may need to modify this based on your data format
        
        if 'text' in examples:
            inputs = examples['text']
        elif 'input' in examples and 'output' in examples:
            inputs = [f"Input: {inp}\nOutput: {out}" for inp, out in zip(examples['input'], examples['output'])]
        elif 'prompt' in examples and 'completion' in examples:
            inputs = [f"{prompt}{completion}" for prompt, completion in zip(examples['prompt'], examples['completion'])]
        else:
            raise ValueError("Dataset must contain 'text' column or 'input'/'output' columns or 'prompt'/'completion' columns")
        
        # Tokenize inputs
        tokenized = self.tokenizer(
            inputs,
            truncation=self.config.truncation,
            padding="max_length",
            max_length=self.config.max_length,
            return_tensors="pt"
        )
        
        # For causal LM, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized
    
    def prepare_datasets(self):
        """Apply preprocessing to datasets"""
        print("Preprocessing datasets...")
        
        self.train_dataset = self.train_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=self.train_dataset.column_names,
            desc="Tokenizing training dataset"
        )
        
        self.eval_dataset = self.eval_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=self.eval_dataset.column_names,
            desc="Tokenizing validation dataset"
        )
        
        print("Datasets preprocessed successfully!")
    
    def setup_training(self):
        """Setup training arguments and trainer"""
        print("Setting up training configuration...")
        
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            max_grad_norm=self.config.max_grad_norm,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            logging_steps=self.config.logging_steps,
            save_strategy=self.config.save_strategy,
            save_steps=self.config.save_steps,
            eval_strategy=self.config.eval_strategy,
            eval_steps=self.config.eval_steps,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better,
            dataloader_num_workers=self.config.dataloader_num_workers,
            remove_unused_columns=self.config.remove_unused_columns,
            report_to="wandb" if self.config.use_wandb else None,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal LM, not masked LM
        )
        
        # Callbacks
        callbacks = []
        if self.config.early_stopping_patience > 0:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.config.early_stopping_patience,
                    early_stopping_threshold=self.config.early_stopping_threshold
                )
            )
        
        # Setup trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            callbacks=callbacks,
        )
        
        print("Training setup completed!")
    
    def train(self):
        """Start the fine-tuning process"""
        print("Starting fine-tuning...")
        
        # Train the model
        train_result = self.trainer.train()
        
        # Save the final model
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        # Save training metrics
        metrics = train_result.metrics
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        
        # Evaluate the model
        eval_metrics = self.trainer.evaluate()
        self.trainer.log_metrics("eval", eval_metrics)
        self.trainer.save_metrics("eval", eval_metrics)
        
        print("Fine-tuning completed!")
        print(f"Model saved to: {self.config.output_dir}")
        
        return train_result, eval_metrics
    
    def generate_text(self, prompt: str, max_length: int = 100, num_return_sequences: int = 1):
        """Generate text using the fine-tuned model"""
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                temperature=0.8,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_texts = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]
        
        return generated_texts


def create_sample_dataset(output_path: str = "data/sample_data.json"):
    """Create a sample dataset for demonstration"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    sample_data = [
        {
            "input": "What is machine learning?",
            "output": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed."
        },
        {
            "input": "Explain neural networks",
            "output": "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes that process information using mathematical operations."
        },
        {
            "input": "What is deep learning?",
            "output": "Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data."
        },
        # Add more samples as needed
    ]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    print(f"Sample dataset created at: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune an LLM model")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--model_name", type=str, default="microsoft/DialoGPT-medium", 
                       help="Model name or path")
    parser.add_argument("--dataset_path", type=str, default="data/sample_data.json",
                       help="Path to training dataset")
    parser.add_argument("--output_dir", type=str, default="fine_tuned_model",
                       help="Output directory for the fine-tuned model")
    parser.add_argument("--create_sample", action="store_true",
                       help="Create sample dataset")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_dataset(args.dataset_path)
        return
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = FineTuningConfig(**config_dict)
    else:
        config = FineTuningConfig(
            model_name=args.model_name,
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
    
    # Initialize and run fine-tuning
    fine_tuner = LLMFineTuner(config)
    
    try:
        fine_tuner.load_tokenizer_and_model()
        fine_tuner.load_datasets()
        fine_tuner.prepare_datasets()
        fine_tuner.setup_training()
        train_result, eval_metrics = fine_tuner.train()
        
        print("Fine-tuning completed successfully!")
        print(f"Final evaluation loss: {eval_metrics.get('eval_loss', 'N/A')}")
        
    except Exception as e:
        print(f"Error during fine-tuning: {str(e)}")
        raise


if __name__ == "__main__":
    main()
