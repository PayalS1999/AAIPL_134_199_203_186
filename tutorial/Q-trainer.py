import os
import torch
import argparse
import re
import json
import time
from typing import Optional, List, Dict, Any
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments
)
from trl import SFTTrainer
from peft import LoraConfig, PeftModel
import wandb

class QuestionGeneratorTrainer:
    """
    Unified trainer class for fine-tuning a Question Generation Agent using SFT.
    """
    
    def __init__(self, args):
        """Initialize the trainer with configuration arguments."""
        self.args = args
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.dataset = None
        
        # Inference model cache
        self._inference_model = None
        self._inference_tokenizer = None
        
        # **MODIFIED**: System prompt for a Question Agent
        self.system_prompt = f"""
        You are an expert-level examiner with deep expertise in designing highly challenging and conceptually rigorous multiple-choice questions (MCQs).
        Your task is to take a given topic and generate a single, complete, and valid JSON object containing a difficult question on that topic.
        The question must be self-contained, logically sound, and have a clear, verifiable answer with a step-by-step explanation.
        Your entire output must be ONLY the raw JSON object.
        """
        
        # Setup environment
        self._setup_environment()
        
        # Display configuration
        self._display_config()
    
    def _setup_environment(self):
        """Setup environment variables and GPU configuration."""
        gpu_ids = [int(x.strip()) for x in self.args.gpu_ids.split(',') if x.strip().isdigit()]
        if not gpu_ids:
            gpu_ids = [0]
        
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", ','.join(map(str, gpu_ids)))
        print(f"PyTorch detected number of available devices: {torch.cuda.device_count()}")
    
    def _display_config(self):
        """Display training configuration."""
        print("=" * 60)
        print(f"QUESTION GENERATOR SFT TRAINER")
        print("=" * 60)
        print(f"Mode: {self.args.mode}")
        print(f"Model: {self.args.model_name}")
        print(f"Output directory: {self.args.output_dir}")
        print(f"Dataset file: {self.args.dataset_file}")
        print(f"Learning rate: {self.args.learning_rate}")
        print(f"Epochs: {self.args.num_train_epochs}")
        print(f"Batch size: {self.args.per_device_train_batch_size}")
        print(f"LoRA rank: {self.args.lora_r}")
        print(f"LoRA alpha: {self.args.lora_alpha}")
        print(f"Max sequence length: {self.args.max_seq_length}")
        print(f"GPU IDs: {self.args.gpu_ids}")
        print("=" * 60)
    
    def load_dataset(self) -> Dataset:
        """Load and process the question dataset for training."""
        print(f"Loading dataset from: {self.args.dataset_file}")
        raw_items = self._load_raw_dataset()
        self.dataset = self._format_sft_dataset(raw_items)
    
    def _load_raw_dataset(self) -> List[Dict[str, Any]]:
        """Load raw dataset items from the generated JSON file."""
        try:
            with open(self.args.dataset_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                raise ValueError("Dataset must be a JSON array of question objects.")
            
            print(f"Successfully loaded {len(data)} questions from JSON file")
            return data
        except Exception as e:
            print(f"Error: Could not load or parse dataset file: {e}")
            raise

    def _format_sft_dataset(self, raw_items: List[Dict[str, Any]]) -> Dataset:
        """
        **MODIFIED**: Formats raw items for training a Question Agent.
        The model learns to generate a full JSON object when given a topic.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.args.model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        formatted_items = []
        for item in raw_items:
            topic = item.get("topic")
            if not topic:
                continue

            # The user provides the topic
            user_content = f"Generate a difficult question on the topic: {topic}"
            
            # The assistant provides the full JSON object as a string
            assistant_content = json.dumps(item, indent=2)
            
            chat_messages = [
                {'role': 'system', 'content': self.system_prompt},
                {'role': 'user', 'content': user_content},
                {'role': 'assistant', 'content': assistant_content}
            ]
            
            formatted_text = tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=False
            )
            
            formatted_items.append({"text": formatted_text})
        
        print(f"Created {len(formatted_items)} SFT training samples for the Question Agent.")
        return Dataset.from_list(formatted_items)
    
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer."""
        print(f"Loading model: {self.args.model_name}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_name,    
            trust_remote_code=True,    
            use_fast=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.tokenizer.padding_side = "right"
        self.tokenizer.model_max_length = self.args.max_seq_length
        
        print("Model and tokenizer loaded successfully")
    
    def setup_peft_config(self) -> LoraConfig:
        """Setup LoRA configuration."""
        return LoraConfig(
            r=self.args.lora_r,
            lora_alpha=self.args.lora_alpha,
            lora_dropout=self.args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear"
        )
    
    def setup_wandb(self):
        # (WandB setup remains the same)
        pass # Placeholder for brevity

    def train_sft(self):
        """Train using Supervised Fine-Tuning."""
        print("Starting SFT training for Question Agent...")
        
        training_args = TrainingArguments(
            output_dir=self.args.output_dir,
            num_train_epochs=self.args.num_train_epochs,
            per_device_train_batch_size=self.args.per_device_train_batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            save_strategy="steps",
            save_steps=100,
            logging_steps=10,
            learning_rate=self.args.learning_rate,
            weight_decay=0.01,
            fp16=False,
            bf16=True,
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            group_by_length=True,
            lr_scheduler_type="cosine",
            report_to="wandb" if not self.args.disable_wandb else "none",
        )
        
        peft_config = self.setup_peft_config()
        
        self.trainer = SFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            args=training_args,
            train_dataset=self.dataset,
            peft_config=peft_config,
            # dataset_text_field="text",
            # max_seq_length=self.args.max_seq_length,
        )
        
        self.trainer.train()
        
        print(f"Saving LoRA adapters to {self.args.output_dir}")
        self.trainer.save_model(self.args.output_dir)
        self.tokenizer.save_pretrained(self.args.output_dir)
        
        print("SFT training completed successfully")

    def train(self):
        """Main training function."""
        self.load_dataset()
        self.setup_model_and_tokenizer()
        self.setup_wandb()
        self.train_sft()
        
        if not self.args.disable_wandb and wandb.run:
            wandb.finish()
        
        print(f"SFT training for Question Agent completed!")

    # --- MODIFIED INFERENCE METHODS ---
    def setup_inference_model(self):
        """Setup fine-tuned model for inference."""
        if self._inference_model is not None:
            return
        
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Setting up inference model on device: {device}")
            
            base_model = AutoModelForCausalLM.from_pretrained(
                self.args.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            
            self._inference_tokenizer = AutoTokenizer.from_pretrained(self.args.model_name, trust_remote_code=True)
            if self._inference_tokenizer.pad_token is None:
                self._inference_tokenizer.pad_token = self._inference_tokenizer.eos_token

            checkpoint_path = self.args.output_dir
            if not os.path.exists(os.path.join(checkpoint_path, 'adapter_model.safetensors')):
                 print(f"Warning: No trained adapter found in {checkpoint_path}. Using base model for inference.")
                 self._inference_model = base_model
            else:
                 print(f"Loading fine-tuned LoRA adapters from: {checkpoint_path}")
                 self._inference_model = PeftModel.from_pretrained(base_model, checkpoint_path)
            
            self._inference_model = self._inference_model.eval()
            print("Inference model setup completed successfully")
        except Exception as e:
            print(f"Error setting up inference model: {e}")
            raise

    def generate_question(self, topic: str) -> str:
        """Generates a new question JSON based on a topic."""
        if self._inference_model is None:
            self.setup_inference_model()

        try:
            user_content = f"Generate a difficult question on the topic: {topic}"
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content}
            ]
            
            full_prompt_text = self._inference_tokenizer.apply_chat_template(
                messages,    
                tokenize=False,    
                add_generation_prompt=True
            )
            
            inputs = self._inference_tokenizer(full_prompt_text, return_tensors="pt").to(self._inference_model.device)
            input_len = inputs["input_ids"].shape[-1]
            
            with torch.inference_mode():
                generation_output = self._inference_model.generate(
                    **inputs,    
                    max_new_tokens=1024,    
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self._inference_tokenizer.eos_token_id
                )
            
            generated_tokens = generation_output[0][input_len:]
            return self._inference_tokenizer.decode(generated_tokens, skip_special_tokens=True)
        except Exception as e:
            return f"ERROR: {e}"

    def test_single_inference(self):
        """Test inference with a single topic."""
        if not self.args.test_topic:
            print("No test topic provided. Use --test_topic argument.")
            return
        
        print("\n--- RUNNING SINGLE INFERENCE TEST ---")
        print(f"Generating question for topic: {self.args.test_topic}")
        
        response = self.generate_question(self.args.test_topic)
        
        print("\n--- Generated Question JSON ---")
        print(response)
        print("--------------------------")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Question Generator Trainer")
    
    # General arguments
    parser.add_argument("--model_name", type=str, default="/jupyter-tutorial/hf_models/Qwen3-4B", help="Base model name or path")
    parser.add_argument("--output_dir", type=str, default="q_checkpoints/sft/", help="Output directory for checkpoints")
    parser.add_argument("--dataset_file", type=str, default="../generated_questions_dataset_250.json", help="Path to the training dataset file (JSON)")
    parser.add_argument("--test_topic", type=str, help="Single topic for testing inference (e.g., 'Truth-teller and Liar Problems')")
    parser.add_argument("--gpu_ids", type=str, default="0", help="GPU IDs to use")
    
    # Training arguments
    parser.add_argument("--mode", type=str, choices=["train", "inference"], default="train", help="Mode of operation")
    parser.add_argument("--num_train_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=float, default=64, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout rate")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Maximum sequence length")
    
    # WandB arguments
    parser.add_argument("--disable_wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument("--wandb_project", type=str, default="question_generator", help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, help="WandB run name")
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    if args.wandb_run_name is None:
        model_display_name = args.model_name.split('/')[-1]
        args.wandb_run_name = f"{model_display_name}-q_agent-sft-r{args.lora_r}"
    
    trainer = QuestionGeneratorTrainer(args)
    
    if args.mode == 'train':
        trainer.train()
    elif args.mode == 'inference':
        trainer.test_single_inference()

if __name__ == "__main__":
    main()