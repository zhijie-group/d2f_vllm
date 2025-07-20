export HF_ALLOW_CODE_EVAL=1
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file config/accelerate_config.yaml eval_dream.py --model dream_lora \
    --model_args $model_args \
    --tasks humaneval \
    --num_fewshot $nshot \
    --batch_size 1 \
    --output_path $output_path \
    --log_samples \
    --confirm_run_unsafe_code