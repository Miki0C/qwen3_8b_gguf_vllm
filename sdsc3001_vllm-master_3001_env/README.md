# vLLM Benchmarking Tool

Automated command generator for vLLM server deployment and performance benchmarking.

## Overview

This tool generates CLI commands to start a vLLM inference server with specified configuration

## Project Structure

```
.
├── config.yaml       # Main configuration file
├── main.py           # Command generator script
└── pyproject.toml    # Project metadata and dependencies
```

## Requirements

- Python ≥3.12
- Dependencies specified in `pyproject.toml`:
  - numpy ≤2.2
  - transformers ≥4.56.1
  - vllm ≤0.10.2

Install with:
```bash
pip install .  # Installs dependencies from pyproject.toml
```

## Configuration Guide (`config.yaml`)

### Server Configuration
```yaml
server:
  model: "Qwen/Qwen3-8B"  # Model path (HuggingFace or local)
  tensor_parallel_size: 1  # Number of GPUs for model parallelism
  gpu_memory_utilization: 0.7  # Fraction of GPU memory to use (0-1)
  max_model_len: null  # Max sequence length (null for default)
  dtype: "auto"  # Data type (auto, fp16, bf16, etc.)
  trust_remote_code: false  # Allow custom model code
```



## Usage

1. **Configure**: Edit `config.yaml` with your desired parameters

2. **Generate Commands**:
   ```bash
   python main.py
   ```

3. **Execute Commands**:
    ```
    uv run --with vllm -- vllm serve Qwen/Qwen3-8B --tensor-parallel-size 1 --gpu-memory-utilization 0.7 --dtype auto
    ```

## Command Generation Details

### Server Command Construction
- Always includes the model path
- Converts `tensor_parallel_size` → `--tensor-parallel-size`
- Handles boolean flags (e.g., `--trust-remote-code` only if `true`)
- Skips `null` values

## Example Workflow

1. Minimal configuration (`config.yaml`):
   ```yaml
   server:
     model: "facebook/opt-125m"
   ```

2. Generate and run:
   ```bash
   python main.py
   # Output:
   # Server Command: uv run --with vllm -- vllm serve facebook/opt-125m
   ```

## Troubleshooting

- **Command Not Found**: Ensure `uv` is installed or modify commands to use direct Python execution
- **Model Loading Errors**: Verify model path and `trust_remote_code` setting
- **GPU Memory Issues**: Adjust `gpu_memory_utilization` or `tensor_parallel_size`

## Credit

This project uses [vLLM](https://github.com/vllm-project/vllm), an open-source fast LLM inference and serving engine. Please refer to the vLLM repository for more information and citation details.
