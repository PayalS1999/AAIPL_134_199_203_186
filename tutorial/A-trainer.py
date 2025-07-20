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

class AnswerAgentTrainer:
    """
    Unified trainer class for fine-tuning an Answer Generation Agent using SFT.
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
        
        # Constants for XML-style response formatting
        self.reasoning_start = "<reasoning>"
        self.reasoning_end = "</reasoning>"
        self.solution_start = "<answer>"
        self.solution_end = "</answer>"
        
        # **NEW**: A dictionary of expert, topic-specific system prompts.
        self.topic_prompts = {
            "Truth-teller and Liar Problems": f"""
You are an expert logician specializing in Knights, Knaves, and Spies puzzles. Your task is to solve the given MCQ.
First, identify the core claims and the conditions (e.g., exactly one Knight).
Second, systematically test hypotheses. Assume a person's type and check for contradictions. Clearly state your assumptions and why they lead to a contradiction or a consistent solution.
Third, once you find a consistent scenario, verify it against all statements.
Place your step-by-step deduction between {self.reasoning_start} and {self.reasoning_end}.
Finally, provide the single correct answer letter between {self.solution_start} and {self.solution_end}.
""",
            "Seating Arrangements (Linear, Circular)": f"""
You are an expert in solving complex seating arrangement puzzles. Your task is to solve the given MCQ.
First, establish the layout (circular or linear) and the number of positions.
Second, identify the most concrete clues (anchors) to place first.
Third, systematically place individuals based on relative positions, creating diagrams or chains of logic as you go. Consider all possible cases and eliminate those that violate conditions.
Place your step-by-step deduction between {self.reasoning_start} and {self.reasoning_end}.
Finally, provide the single correct answer letter between {self.solution_start} and {self.solution_end}.
""",
            "Puzzles involving generations and family tree logic": f"""
You are an expert genealogist specializing in complex family tree and blood relation puzzles. Your task is to solve the given MCQ.
First, deconstruct each statement to establish direct relationships (e.g., 'son of my mother's brother' is 'my cousin').
Second, build a family tree, either mentally or by listing connections, to map out the relationships between all individuals.
Third, use the completed tree to find the relationship asked in the question.
Place your step-by-step deduction between {self.reasoning_start} and {self.reasoning_end}.
Finally, provide the single correct answer letter between {self.solution_start} and {self.solution_end}.
""",
            "default": f"""
You are an expert in logical reasoning and complex problem-solving. Your task is to answer the given MCQ.
Think about the problem and provide your working out between {self.reasoning_start} and {self.reasoning_end}.
Then, provide your answer between {self.solution_start} and {self.solution_end}.
Your answer should be a single letter (A, B, C, or D). Do not include any additional text outside of these tags.
"""
        }
        
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
        print(f"ANSWER AGENT SFT TRAINER")
        print("=" * 60)
        # (Configuration details print here)
    
    def load_dataset(self):
        """Load and process the question dataset."""
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
        **MODIFIED**: Formats raw items for training an Answer Agent.
        It now uses a topic-specific system prompt for each question.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.args.model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        formatted_items = []
        for item in raw_items:
            topic = item.get("topic")
            question_text = item.get("question")
            choices = item.get("choices")
            answer_letter = item.get("answer")
            reasoning = item.get("explanation")

            if not all([topic, question_text, choices, answer_letter, reasoning]):
                continue

            # **NEW**: Select the system prompt based on the topic in the data.
            system_prompt = self.topic_prompts.get(topic, self.topic_prompts["default"])
            
            # The user provides the question and choices
            full_question = f"Question: {question_text}\n" + "\n".join(choices)
            
            # The assistant provides the reasoning and answer in XML format
            model_completion = (
                f"{self.reasoning_start}{reasoning}{self.reasoning_end}"
                f"{self.solution_start}{answer_letter}{self.solution_end}"
            )
            
            chat_messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': full_question},
                {'role': 'assistant', 'content': model_completion}
            ]
            
            formatted_text = tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=False
            )
            
            formatted_items.append({"text": formatted_text})
        
        print(f"Created {len(formatted_items)} SFT training samples for the Answer Agent.")
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

    def train_sft(self):
        """Train using Supervised Fine-Tuning."""
        print("Starting SFT training for Answer Agent...")
        
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
            args=training_args,
            train_dataset=self.dataset,
            peft_config=peft_config,
            dataset_text_field="text",
            max_seq_length=self.args.max_seq_length,
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
        # self.setup_wandb() # WandB setup can be added here
        self.train_sft()
        print(f"SFT training for Answer Agent completed!")

    # --- INFERENCE METHODS ---
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

    def generate_response(self, question_data: dict) -> str:
        """Generates an answer for a given question object."""
        if self._inference_model is None:
            self.setup_inference_model()

        try:
            topic = question_data.get("topic", "default")
            system_prompt = self.topic_prompts.get(topic, self.topic_prompts["default"])
            
            question_text = question_data.get("question", "")
            choices = "\n".join(question_data.get("choices", []))
            user_content = f"Question: {question_text}\n{choices}"

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
            
            full_prompt_text = self._inference_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            inputs = self._inference_tokenizer(full_prompt_text, return_tensors="pt").to(self._inference_model.device)
            input_len = inputs["input_ids"].shape[-1]
            
            with torch.inference_mode():
                generation_output = self._inference_model.generate(
                    **inputs, max_new_tokens=512, do_sample=False,
                    pad_token_id=self._inference_tokenizer.eos_token_id
                )
            
            generated_tokens = generation_output[0][input_len:]
            return self._inference_tokenizer.decode(generated_tokens, skip_special_tokens=True)
        except Exception as e:
            return f"ERROR: {e}"

    def test_single_inference(self):
        """Test inference with a single question from a JSON string."""
        if not self.args.test_question_json:
            print("No test question JSON provided. Use --test_question_json argument.")
            return
        
        print("\n--- RUNNING SINGLE INFERENCE TEST ---")
        try:
            question_data = json.loads(self.args.test_question_json)
            print(f"Answering question: {question_data.get('question', '')[:50]}...")
            
            response = self.generate_response(question_data)
            
            print("\n--- Model Response ---")
            print(response)
            print("--------------------------")
        except json.JSONDecodeError:
            print("Error: Invalid JSON format for --test_question_json.")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Answer Agent Trainer")
    
    # General arguments
    parser.add_argument("--model_name", type=str, default="/jupyter-tutorial/hf_models/Qwen3-4B", help="Base model name or path")
    parser.add_argument("--output_dir", type=str, default="a_checkpoints/a_agent/sft", help="Output directory for checkpoints")
    parser.add_argument("--dataset_file", type=str, default="generated_questions_dataset_full_250.json", help="Path to the training dataset file (JSON)")
    parser.add_argument("--test_question_json", type=str, help="Single question as a JSON string for testing inference.")
    parser.add_argument("--gpu_ids", type=str, default="0", help="GPU IDs to use")
    
    # Mode
    parser.add_argument("--mode", type=str, choices=["train", "inference"], default="train", help="Mode of operation")
    
    # Training arguments
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    
    # WandB arguments (optional)
    parser.add_argument("--disable_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="answer_agent")
    parser.add_argument("--wandb_run_name", type=str)
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    if args.wandb_run_name is None:
        model_display_name = args.model_name.split('/')[-1]
        args.wandb_run_name = f"{model_display_name}-a_agent-sft-r{args.lora_r}"
    
    trainer = AnswerAgentTrainer(args)
    
    if args.mode == 'train':
        trainer.train()
    elif args.mode == 'inference':
        trainer.test_single_inference()

if __name__ == "__main__":
    main()