import yaml

def generate_llmperf_command(yaml_file):
    """
    Generates an LLMPerf benchmark command from a YAML file.
    
    YAML example:
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
    additional-sampling-params: {}
    """
    with open(yaml_file, 'r') as f:
        params = yaml.safe_load(f)
    
    command = 'python token_benchmark_ray.py'
    
    # Map YAML keys (underscores) to command flags (dashes)
    flag_map = {
        'model': '--model',
        'mean_input_tokens': '--mean-input-tokens',
        'stddev_input_tokens': '--stddev-input-tokens',
        'mean_output_tokens': '--mean-output-tokens',
        'stddev_output_tokens': '--stddev-output-tokens',
        'max_num_completed_requests': '--max-num-completed-requests',
        'timeout': '--timeout',
        'num_concurrent_requests': '--num-concurrent-requests',
        'results_dir': '--results-dir',
        'llm_api': '--llm-api',
        'additional_sampling_params': '--additional-sampling-params'
    }
    
    for key, value in params.items():
        if key in flag_map:
            flag = flag_map[key]
            # Quote values with spaces or special chars like '/'
            if isinstance(value, str) and (' ' in value or '/' in value):
                value = f"'{value}'"
            command += f' {flag} {value}'
    
    return command

if __name__ == '__main__':
    yaml_file = 'config.yaml'  # Hard-coded path; replace with your actual YAML file path
    print("export OPENAI_API_BASE='http://localhost:5000/v1'")
    print("export OPENAI_API_KEY='dummy'")
    print(generate_llmperf_command(yaml_file))