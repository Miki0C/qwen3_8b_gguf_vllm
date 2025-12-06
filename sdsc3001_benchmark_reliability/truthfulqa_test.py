# Standard library imports
import argparse
import time
import gc
import os
import random
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

# Third-party library imports
import torch
import psutil
import pandas as pd
import numpy as np
import yaml
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Configuration class for model parameters and settings
@dataclass
class ModelConfig:
    """Configuration class for model testing parameters"""
    path: str  # Path to the model
    name: str = "Test Model"  # Display name for the model
    use_awq: bool = False  # Whether to use AWQ quantization
    enable_reasoning: bool = False  # Enable reasoning mode (step-by-step)
    max_new_tokens: int = 256  # Maximum tokens to generate
    temperature: float = 0.7  # Sampling temperature
    n_generations: int = 3  # Number of generations per prompt
    device: str = "cuda" if torch.cuda.is_available() else "cpu"  # Device to use
    sample_size: int = 50  # Number of questions to evaluate
    results_dir: str = "results"  # Directory to save results
    save_generations: bool = True  # Whether to save raw generations

@dataclass
class ModelConfig:
    path: str
    name: str = "Test Model"
    use_awq: bool = False
    enable_reasoning: bool = False
    max_new_tokens: int = 256
    temperature: float = 0.7
    n_generations: int = 3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    sample_size: int = 50
    results_dir: str = "results"
    save_generations: bool = True

def load_config(config_path: str = "truthqa_config.yaml") -> ModelConfig:
    """
    Load model configuration from YAML file or use CLI arguments as fallback
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        ModelConfig: Configuration object with all model parameters
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If required model path is missing
    """
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file {config_path} not found")
    with open(config_path) as f:
        config_data = yaml.safe_load(f) or {}
    if "model" not in config_data or "path" not in config_data["model"]:
        raise ValueError("Model path is required in config")
    return ModelConfig(
        path=config_data["model"]["path"],
        name=config_data["model"].get("name", "Test Model"),
        use_awq=config_data["model"].get("use_awq", False),
        enable_reasoning=config_data["model"].get("enable_reasoning", False),
        max_new_tokens=1024 if config_data["model"].get("enable_reasoning") else 256,
        temperature=config_data.get("generation", {}).get("temperature", 0.7),
        n_generations=config_data.get("generation", {}).get("n_generations", 3),
        sample_size=config_data.get("dataset", {}).get("sample_size", 50),
        results_dir=config_data.get("output", {}).get("results_dir", "results"),
        save_generations=config_data.get("output", {}).get("save_generations", True)
    )

try:
    config = load_config()
except (FileNotFoundError, ValueError) as e:
    print(f"Using CLI args (YAML config not found/invalid: {str(e)})")
    def parse_args() -> ModelConfig:
        parser = argparse.ArgumentParser(description="Test single model on TruthfulQA")
        parser.add_argument("--model_path", type=str, required=True)
        parser.add_argument("--model_name", type=str, default="Test Model")
        parser.add_argument("--use_awq", action="store_true")
        parser.add_argument("--enable_reasoning", action="store_true")
        parser.add_argument("--sample_size", type=int, default=50)
        args = parser.parse_args()
        return ModelConfig(
            path=args.model_path,
            name=args.model_name,
            use_awq=args.use_awq,
            enable_reasoning=args.enable_reasoning,
            max_new_tokens=1024 if args.enable_reasoning else 256,
            sample_size=args.sample_size
        )
    config = parse_args()

print(f"Using device: {config.device}")
print(f"Testing model: {config.name} at {config.path} (AWQ: {config.use_awq}, Reasoning: {config.enable_reasoning})")

# Load TruthfulQA dataset (use 'train' split)
truthfulqa = load_dataset("domenicrosati/TruthfulQA", split="train")
questions = truthfulqa["Question"]
answers = truthfulqa["Best Answer"]
if config.sample_size > 0:
    questions = questions[:config.sample_size]
    answers = answers[:config.sample_size]

REASONING_PREFIX = "Let's think step by step: "
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Optional AWQ import
AWQ_AVAILABLE = False
try:
    from awq import AutoAWQForCausalLM
    AWQ_AVAILABLE = True
except ImportError:
    pass

def load_model(model_path, use_awq):
    """
    Load model and tokenizer with optional AWQ quantization
    
    Args:
        model_path: Path to pretrained model
        use_awq: Whether to use AWQ quantization
        
    Returns:
        tuple: (model, tokenizer, device) where device is the model's device
        
    Raises:
        ImportError: If AWQ is requested but not available
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if use_awq:
        if not AWQ_AVAILABLE:
            raise ImportError("autoawq not installed. Run: pip install autoawq")
        model = AutoAWQForCausalLM.from_quantized(model_path, fuse_layers=True, device_map="auto")
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if config.device == "cuda" else torch.float32,
            device_map="auto"
        )
    model_device = next(model.parameters()).device
    return model, tokenizer, model_device

def measure_memory(device: str) -> float:
    """
    Measure memory usage in MB
    
    Args:
        device: 'cuda' for GPU or any other string for CPU
        
    Returns:
        float: Memory usage in megabytes (MB)
    """
    if device == "cuda":
        return torch.cuda.memory_allocated() / 1024**2
    else:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024**2

def generate_reliably(model, tokenizer, prompt, n_gens, model_device, config):
    """
    Generate multiple responses for a given prompt reliably
    
    Args:
        model: Loaded language model
        tokenizer: Model tokenizer
        prompt: Input text prompt
        n_gens: Number of generations to produce
        model_device: Device where model is loaded
        config: Model configuration
        
    Returns:
        List[str]: List of generated responses
    """
    gens = []
    try:
        if config.enable_reasoning:
            prompt = REASONING_PREFIX + prompt
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model_device)
        for _ in range(n_gens):
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
                text = tokenizer.decode(
                    outputs[0][inputs.input_ids.shape[1]:],
                    skip_special_tokens=True
                ).strip()
                gens.append(text)
        return gens
    except Exception as e:
        print(f"Generation failed for prompt: {prompt[:50]}... Error: {str(e)}")
        return [""] * n_gens

def compute_metrics(generations_list: List[List[str]], reference_answers: List[str]) -> Tuple[float, float]:
    """
    Compute truthfulness and reliability metrics for model generations
    
    Args:
        generations_list: List of lists containing generated responses
        reference_answers: List of correct answers
        
    Returns:
        tuple: (truthfulness_score, reliability_score)
    """
    truth_scores = []
    reliabilities = []
    all_responses = [resp for gens in generations_list for resp in gens]
    all_embs = embedder.encode(all_responses)
    ref_embs = embedder.encode(reference_answers)
    idx = 0
    for i, gens in enumerate(generations_list):
        if len(gens) < 2:
            continue
        embs = all_embs[idx:idx+len(gens)]
        idx += len(gens)
        # Reliability: inter-response consistency
        sims = [util.cos_sim(embs[x], embs[y]).item() for x in range(len(gens)) for y in range(x+1, len(gens))]
        reliabilities.append(np.mean(sims) if sims else 0)
        # Truthfulness: similarity to reference answer
        ref_emb = ref_embs[i]
        truth_score = np.mean([util.cos_sim(emb, ref_emb).item() for emb in embs])
        truth_scores.append(truth_score)
    return (
        np.mean(truth_scores) if truth_scores else 0,
        np.mean(reliabilities) if reliabilities else 0
    )

model, tokenizer, model_device = load_model(config.path, config.use_awq)
if model is None:
    exit(1)

base_memory = measure_memory(config.device)
total_time = 0
all_gens_list = []
max_mem = base_memory

print(f"\nStarting evaluation with {len(questions)} prompts...")
start_total_time = time.perf_counter()

for i, prompt in enumerate(questions, 1):
    try:
        start_time = time.perf_counter()
        gens = generate_reliably(
            model, tokenizer, prompt,
            config.n_generations, model_device, config
        )
        end_time = time.perf_counter()
        gen_time = end_time - start_time
        total_time += gen_time
        current_mem = measure_memory(config.device)
        max_mem = max(max_mem, current_mem)
        all_gens_list.append(gens)
        if i % 5 == 0 or i == len(questions):
            avg_time = total_time / i
            print(f"Processed {i}/{len(questions)} prompts | Avg time: {avg_time:.2f}s | Current mem: {current_mem:.1f} MB")
        gc.collect()
        if config.device == "cuda":
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error processing prompt {i}: {str(e)}")
        all_gens_list.append([""] * config.n_generations)

total_time = time.perf_counter() - start_total_time
avg_time = total_time / len(questions)

truth, rel = compute_metrics(all_gens_list, answers)
final_mem = measure_memory(config.device)

results = {
    "Model": config.name,
    "Model Path": config.path,
    "Device": config.device,
    "Quantization": "AWQ" if config.use_awq else "None",
    "Reasoning Enabled": config.enable_reasoning,
    "Total Prompts": len(questions),
    "Generations per Prompt": config.n_generations,
    "Avg Time per Prompt (s)": round(avg_time, 3),
    "Total Time (s)": round(total_time, 3),
    "Peak Memory (MB)": round(max_mem, 2),
    "Final Memory (MB)": round(final_mem, 2),
    "Truthfulness Score": round(truth, 3),
    "Reliability Score": round(rel, 3),
    "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
}

print("\nEvaluation Summary:")
print(f"Model: {results['Model']} ({results['Model Path']})")
print(f"Device: {results['Device']} | Quantization: {results['Quantization']}")
print(f"Total prompts processed: {results['Total Prompts']}")
print(f"Average time per prompt: {results['Avg Time per Prompt (s)']}s")
print(f"Peak memory usage: {results['Peak Memory (MB)']} MB")
print(f"Truthfulness score: {results['Truthfulness Score']}")
print(f"Reliability score: {results['Reliability Score']}")

def ensure_directory(path: str) -> None:
    """
    Ensure a directory exists, create if it doesn't
    
    Args:
        path: Directory path to check/create
        
    Raises:
        OSError: If directory creation fails
    """
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise OSError(f"Failed to create directory {path}: {str(e)}")

try:
    ensure_directory(config.results_dir)
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    model_name_safe = "".join(c if c.isalnum() else "_" for c in config.name)
    results_file = f"{config.results_dir}/truthfulqa_{model_name_safe}_{timestamp}.csv"
    results_df = pd.DataFrame([results])
    results_df.to_csv(results_file, index=False)
    print(f"Results saved to {results_file}")
    if config.save_generations:
        generations_file = f"{config.results_dir}/generations_{timestamp}.csv"
        generations_df = pd.DataFrame({
            "question": questions[:len(all_gens_list)],
            "reference_answer": answers[:len(all_gens_list)],
            "generations": [" | ".join(gens) for gens in all_gens_list]
        })
        generations_df.to_csv(generations_file, index=False)
        print(f"Raw generations saved to {generations_file}")
except Exception as e:
    print(f"Error saving results: {str(e)}")
    try:
        results_df.to_csv("truthfulqa_results_fallback.csv", index=False)
        print("Saved results to current directory as fallback")
    except:
        print("Could not save results anywhere")

del model
gc.collect()
if config.device == "cuda":
    torch.cuda.empty_cache()
