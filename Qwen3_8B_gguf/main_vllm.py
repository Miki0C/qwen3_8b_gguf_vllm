from vllm import LLM, SamplingParams

prompts = ["Hello, how are you?"]

sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=256,
)

# 确保模型路径正确（指向你的 gguf 文件）
llm = LLM(
        model="/home/mkchua2/Miki/SDSC3001/Qwen3_8B_gguf/qwen3-8b.gguf",  # 模型路径
        tokenizer="Qwen/Qwen3-8B",
        tensor_parallel_size=1,
        trust_remote_code=True,
        gpu_memory_utilization=0.4,
        dtype="float16",
        max_model_len=5000,
)

outputs = llm.generate(prompts, sampling_params)

for out in outputs:
    print("Prompt :", repr(out.prompt))
    print("Answer :", repr(out.outputs[0].text))
    print("-" * 60)