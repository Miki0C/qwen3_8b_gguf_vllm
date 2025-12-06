# TruthfulQA LLM Benchmark

This project provides a framework for evaluating Large Language Models (LLMs) on the [TruthfulQA](https://huggingface.co/datasets/domenicrosati/TruthfulQA) benchmark using semantic similarity metrics. It supports open-ended model outputs and quantifies both truthfulness and reliability.

## Features
- Loads the TruthfulQA dataset directly from Hugging Face (no CSV required)
- Supports Hugging Face and local models, including AWQ quantization
- Generates multiple responses per prompt
- Computes semantic similarity between model outputs and reference answers
- Measures reliability (consistency across generations)
- Tracks memory usage and generation time
- Saves detailed results and raw generations to CSV
- Configurable via YAML or CLI

## Getting Started

### 1. Install Dependencies
This project uses `pyproject.toml` for dependency management. Install all dependencies with:
```bash
pip install .
```
Or, for development:
```bash
pip install -e .
```
All required packages are specified in `pyproject.toml`.
Optional: `awq` for quantized models.

### 2. Configure Your Model
Edit `truthqa_config.yaml`:
```yaml
model:
  path: "<your_model_path_here>"
  name: "Your Model Name"
  use_awq: false
  enable_reasoning: false

generation:
  n_generations: 3
  temperature: 0.7

dataset:
  sample_size: 50

output:
  results_dir: "results"
  save_generations: true
```

### 3. Run the Evaluation
```bash
python truthfulqa_test.py
```
Or with CLI arguments:
```bash
python truthfulqa_test.py --model_path <your_model_path> --model_name "Your Model Name"
```

## Output
- Results summary CSV: `results/truthfulqa_<model_name>_<timestamp>.csv`
- Raw generations CSV: `results/generations_<timestamp>.csv`

## Metrics
- **Truthfulness Score**: Average semantic similarity between model outputs and reference answers
- **Reliability Score**: Consistency of model outputs across multiple generations

## Customization
- Change the number of prompts (`sample_size`) or generations in the config
- Use different models by updating the config or CLI arguments
- Enable chain-of-thought reasoning with `enable_reasoning: true`

## License
MIT

## Citation
If you use this benchmark, please cite the TruthfulQA dataset and any models you evaluate.
