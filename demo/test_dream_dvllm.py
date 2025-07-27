from pprint import pprint
from transformers import AutoTokenizer

from d2f_vllm import LLM, SamplingParams

if __name__ == "__main__":
    model = "/data1/ckpts/Dream-org/Dream-v0-Base-7B"
    llm = LLM(
        model, 
        lora_path="/home/jyj/workspace-2/D2F/lora_weight",
        use_lora=True,
        model_name="dream", 
        model_type="diffusion_lm",
        enforce_eager=True, 
        tensor_parallel_size=1,
        accept_threshold=0.95,
        complete_threshold=0.9,
        add_new_block_threshold=0.1,
    )

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True, trust_remote_code=True)

    # 多个测试问题
    chat_prompts = [
        [{"role": "user", "content": "Can you explain what diffusion decoding means in language models?"}],
        [{"role": "user", "content": "What are the advantages of block-sparse attention compared to dense attention?"}],
        [{"role": "user", "content": "How can I adapt Triton kernels dynamically based on different GPU architectures?"}],
        [{"role": "user", "content": "What are the trade-offs between static vs. dynamic kernel configurations in deployment?"}],
        [{"role": "user", "content": "Please summarize the key bottlenecks in scaling LLMs across multiple GPUs."}],
    ]

    # 利用 tokenizer 构造 chat 模板
    prompts = [
        tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        for chat in chat_prompts
    ]

    # 采样参数设置
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)

    # 推理
    outputs = llm.generate(prompts, sampling_params)
    pprint(outputs)