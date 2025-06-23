set -x

export PYTHONPATH=`pwd`:$PYTHONPATH

nnodes=2
tp_size=1

# Vllm
n_samples=4
temperature=0.8
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 32))

# Path
base_dir=.
model_path=${base_dir}/model/WizardCodeR-1.5B-DAPO
output_dir=${model_path}/output
data_dir=${base_dir}/data/test
dataset=livecodebench_v5

python -m verl.trainer.main_generation \
    trainer.nnodes=${nnodes} \
    trainer.n_gpus_per_node=8 \
    model.path=${model_path} \
    data.path=${data_dir}/${dataset}.parquet \
    data.output_path=${output_dir}/${dataset}.parquet \
    data.batch_size=2048 \
    data.n_samples=${n_samples} \
    rollout.name=vllm \
    rollout.gpu_memory_utilization=0.9 \
    rollout.enforce_eager=False \
    rollout.free_cache_engine=False \
    rollout.tensor_model_parallel_size=${tp_size} \
    rollout.temperature=$temperature \
    rollout.top_k=-1 \
    rollout.top_p=1.0 \
    rollout.prompt_length=$max_prompt_length \
    rollout.response_length=$max_response_length \
    rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length))
