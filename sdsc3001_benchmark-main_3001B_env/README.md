# SDSC3001 Benchmarking via LLMPerf - OpenAI Compatible Server Testing Tool

## Quick Start

### 1. Set Up Python Environment
```bash
# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  

# On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
# Install directly from this repository
pip install -e .
```

### 3. Configure Parameters
Use the `config.yaml` file to configure test parameters:
```yaml
model: Qwen/Qwen3-8B
mean_input_tokens: 550
stddev_input_tokens: 150
mean_output_tokens: 150
stddev_output_tokens: 10
max_num_completed_requests: 100
timeout: 600
num_concurrent_requests: 10
results_dir: results
llm_api: openai
additional_sampling_params: '{"temperature": 0.0, "top_p": 1.0, "max_tokens": 256}'
```

### 4. Generate and Run Command
Use `cli_generator.py` to generate the test command:
```bash
python cli_generator.py
```

Example generated command:
```bash
export OPENAI_API_BASE='http://localhost:8000/v1'
export OPENAI_API_KEY='dummy'
python token_benchmark_ray.py --model 'Qwen/Qwen3-8B' --mean-input-tokens 550 --stddev-input-tokens 150 --mean-output-tokens 150 --stddev-output-tokens 10 --max-num-completed-requests 100 --timeout 600 --num-concurrent-requests 10 --results-dir results --llm-api openai --additional-sampling-params '{"temperature": 0.0, "top_p": 1.0, "max_tokens": 256}'
```

## Parameter Descriptions

| Parameter | Description |
|-----------|-------------|
| `model` | Name of the model to test |
| `mean_input_tokens` | Average input tokens |
| `stddev_input_tokens` | Input tokens standard deviation |
| `mean_output_tokens` | Average output tokens |
| `stddev_output_tokens` | Output tokens standard deviation |
| `max_num_completed_requests` | Maximum completed requests |
| `timeout` | Timeout in seconds |
| `num_concurrent_requests` | Concurrent requests count |
| `results_dir` | Results directory |
| `llm_api` | API type (openai) |
| `additional_sampling_params` | Additional sampling parameters |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_BASE` | OpenAI compatible API base URL |
| `OPENAI_API_KEY` | API key (can set to dummy for testing) |

> **Need Help?**  
> For more detailed documentation or if you encounter unresolved issues, please refer to the [original repository](https://github.com/ray-project/llmperf) for complete information.
