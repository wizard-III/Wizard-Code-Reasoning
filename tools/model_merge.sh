model_path=../output/grpo/ArcherCodeR/DAPO-Qwen2.5-1.5B/global_step_120/actor

python -m tools.model_merge merge \
    --backend fsdp \
    --local_dir ${model_path} \
    --target_dir ${model_path}/hf_model