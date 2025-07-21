
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取上一级目录，也就是 project_root/
project_root = os.path.abspath(os.path.join(current_dir, ".."))

# 添加 project_root 到模块搜索路径
sys.path.insert(0, project_root)
# print("\n".join(sys.path))
from d2f_vllm import LLM, SamplingParams
llm = LLM(
    "/data1/ckpts/Dream-org/Dream-v0-Base-7B", 
    model_name="dream", 
    enforce_eager=True, 
    tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
print(outputs[0]["text"])