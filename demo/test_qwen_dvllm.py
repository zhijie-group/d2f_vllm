from d2f_vllm import LLM, SamplingParams
llm = LLM(
    "/data1/ckpts/Qwen/Qwen3-0.6B", 
    enforce_eager=False, 
    tensor_parallel_size=1,
    model_name="qwen3", 
    model_type="causal_lm"
)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = prompts = [
    "Hello, Nano-vLLM.", 
    "What is the meaning of life in your opinion?", 
    "Tell me a joke about AI. Say it in a funny way."
]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]