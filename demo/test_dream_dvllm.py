from d2f_vllm import LLM, SamplingParams
llm = LLM(
    "/data1/ckpts/Dream-org/Dream-v0-Base-7B", 
    lora_path="/data1/xck/ckpt/wx_dream_base/Decoder-ddt_test-20k",
    use_lora=True,
    model_name="dream", 
    model_type="diffusion_lm",
    enforce_eager=True, 
    tensor_parallel_size=1
)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = [
    "Hello, Nano-vLLM.", 
    "What is the meaning of life in your opinion?", 
    "Tell me a joke about AI. Say it in a funny way."
]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]