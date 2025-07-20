#!/usr/bin/zsh

# 基础模型路径
base_model="/data1/ckpts/Dream-org/Dream-v0-Base-7B"
# LoRA模型路径
lora_model="Decoder-ddt_test-20k"
lora_root="/data1/xck/ckpt/wx_dream_base"

# 评测参数（可根据需要修改）
nshot=0
length=256
temperature=0
limit=10000
diffusion_steps=256
block_size=32
block_add_threshold=0.9
decoded_token_threshold=0.9
skip_threshold=0.95
top_p="none"
dtype="bfloat16"
sampling_strategy="default"

# 输出路径
output_path="evals_dream_single/${lora_model}/humaneval-ns${nshot}-len${length}-temp${temperature}-limit${limit}-diffsteps${diffusion_steps}-block${block_size}-thresh${block_add_threshold}-decodethresh${decoded_token_threshold}-skip${skip_threshold}-topp${top_p}-dtype${dtype}-sampling${sampling_strategy}"

lora_path="${lora_root}/${lora_model}"
# 构建model_args
if [[ "$top_p" == "none" ]]; then
    model_args="pretrained=${base_model},lora_path=${lora_path},max_new_tokens=${length},diffusion_steps=${diffusion_steps},temperature=${temperature},add_bos_token=true,escape_until=true,block_size=${block_size},block_add_threshold=${block_add_threshold},skip_threshold=${skip_threshold},decoded_token_threshold=${decoded_token_threshold},dtype=${dtype},sampling_strategy=${sampling_strategy},save_dir=${output_path}"
else
    model_args="pretrained=${base_model},lora_path=${lora_path},max_new_tokens=${length},diffusion_steps=${diffusion_steps},temperature=${temperature},top_p=${top_p},add_bos_token=true,escape_until=true,block_size=${block_size},block_add_threshold=${block_add_threshold},skip_threshold=${skip_threshold},decoded_token_threshold=${decoded_token_threshold},dtype=${dtype},sampling_strategy=${sampling_strategy},save_dir=${output_path}"
fi

export HF_ALLOW_CODE_EVAL=1
CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 29520 --num_processes 1 eval_dream.py --model dream_lora \
    --model_args $model_args \
    --tasks humaneval \
    --num_fewshot $nshot \
    --batch_size 1 \
    --output_path $output_path \
    --log_samples \
    --confirm_run_unsafe_code

echo "Single evaluation completed!"