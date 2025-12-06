import yaml
import datetime

def main():
    yaml_file = '/home/mkchua2/Miki/SDSC3001/sdsc3001_vllm-master/config.yaml'
    
    # Load YAML
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)
    
    server_params = config.get('server', {})
    benchmark_params = config.get('benchmark', {})
    
    # Construct server CLI string
    server_cmd = ['uv', 'run', '--with', 'vllm', '--', 'vllm', 'serve', server_params['model']]
    for key, value in server_params.items():
        if key != 'model' and value is not None:
            flag = f'--{key.replace("_", "-")}'
            if isinstance(value, bool):
                if value:
                    server_cmd.append(flag)
            elif isinstance(value, str) and value.startswith('{') and value.endswith('}'):
                server_cmd.extend([flag, f"'{value}'"])
            else:
                server_cmd.extend([flag, str(value)])
    server_cmd_str = ' '.join(server_cmd)
    
    # Print the commands
    print("Server Command (run this in one terminal):")
    print(server_cmd_str)

if __name__ == "__main__":
    main()