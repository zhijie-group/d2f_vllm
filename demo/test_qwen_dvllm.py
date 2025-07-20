from d2f_vllm import LLM, SamplingParams
llm = LLM("/data1/ckpts/Qwen/Qwen3-0.6B", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]