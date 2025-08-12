#!/usr/bin/zsh
CUDA_VISIBLE_DEVICES=7 python demo/test_dream_dvllm.py 2>&1 | tee log/test_dvllm_dream.log