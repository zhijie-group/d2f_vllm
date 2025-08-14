#!/usr/bin/zsh
CUDA_VISIBLE_DEVICES=1,2 python demo/test_qwen_dvllm.py 2>&1 | tee log/test_dvllm_qwen.log