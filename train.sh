#!/bin/bash

set -e -x

# export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
# export TORCHDYNAMO_VERBOSE=1
export WANDB_BASE_URL="https://api.bandw.top" # [WANDB_BASE_URL can be "https://wandb.wan-ai.com" or "https://wandb.ai"]
export WANDB_MODE="online" # [WANDB_MODE can be "online", "offline", or "disabled"]
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export FINETRAINERS_LOG_LEVEL="DEBUG" # in constant.py, [DEBUG, INFO, WARNING, ERROR, CRITICAL]

# Finetrainers supports multiple backends for distributed training. Select your favourite and benchmark the differences!
# BACKEND="accelerate"
BACKEND="ptd"

# In this setting, I'm using 2 GPUs on a 4-GPU node for training
NUM_GPUS=4
CUDA_VISIBLE_DEVICES="2,3,6,7"

# Check the JSON files for the expected JSON format
TRAINING_DATASET_CONFIG="examples/training/sft/wan/crush_smol_lora/training_disney.json"
VALIDATION_DATASET_FILE="examples/training/sft/wan/crush_smol_lora/validation.json"

# Depending on how many GPUs you have available, choose your degree of parallelism and technique!
  ## 这些参数控制了训练时的并行方式，主要包括：

  # pp_degree：Pipeline Parallelism Degree（流水线并行度）
  # dp_degree：Data Parallelism Degree（数据并行度）
  # dp_shards：Data Parallel Shards（数据并行分片数，通常用于 FSDP）
  # cp_degree：Checkpoint Parallelism Degree（检查点并行度）
  # tp_degree：Tensor Parallelism Degree（张量并行度）
  # ---
  # 1. DDP（Distributed Data Parallel，数据并行）
  # DDP_1：单卡/单进程数据并行
  # DDP_2：2卡数据并行
  # DDP_4：4卡数据并行
  # 特点：每张卡上有一份完整模型，数据被分成多份，各自独立前向/反向，最后梯度同步。适合大多数常规分布式训练。
  # 2. FSDP（Fully Sharded Data Parallel，全参数分片数据并行）
  # FSDP_2：参数在2个分片上分布
  # FSDP_4：参数在4个分片上分布
  # 特点：模型参数被切分到多张卡上，显存占用更低，适合超大模型。
  # 3. HSDP（Hybrid Sharded Data Parallel，混合分片数据并行）
  # HSDP_2_2：2卡数据并行，每卡2个分片
  # 特点：结合 DDP 和 FSDP，适合更复杂的分布式场景。
  # 4. 其他参数
  # pp_degree：流水线并行，适合模型极大时跨多卡分阶段执行（一般用不到，除非模型极大）。
  # tp_degree：张量并行，适合模型内部结构可分割时（如 GPT-3/LLAMA 这类大模型）。
  # cp_degree：检查点并行，主要用于优化 checkpoint 存储和恢复。
  # ---
DDP_1="--parallel_backend $BACKEND --pp_degree 1 --dp_degree 1 --dp_shards 1 --cp_degree 1 --tp_degree 1"
DDP_2="--parallel_backend $BACKEND --pp_degree 1 --dp_degree 2 --dp_shards 1 --cp_degree 1 --tp_degree 1"
DDP_4="--parallel_backend $BACKEND --pp_degree 1 --dp_degree 4 --dp_shards 1 --cp_degree 1 --tp_degree 1"
FSDP_2="--parallel_backend $BACKEND --pp_degree 1 --dp_degree 1 --dp_shards 2 --cp_degree 1 --tp_degree 1"
FSDP_4="--parallel_backend $BACKEND --pp_degree 1 --dp_degree 1 --dp_shards 4 --cp_degree 1 --tp_degree 1"
HSDP_2_2="--parallel_backend $BACKEND --pp_degree 1 --dp_degree 2 --dp_shards 2 --cp_degree 1 --tp_degree 1"

# Parallel arguments
parallel_cmd=(
  $DDP_4
)

# Model arguments
model_cmd=(
  --model_name "wan"
  # --pretrained_model_name_or_path "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
  --pretrained_model_name_or_path "/home/wjh/projects/finetrainers/finetrainers_ckpts/Wan2.1-T2V-1.3B-Diffusers"
)

# Dataset arguments
# Here, we know that the dataset size if about ~50 videos. Since we're using 2 GPUs, we precompute
# embeddings of 25 dataset items per GPU. Also, we're using a very small dataset for finetuning, so
# we are okay with precomputing embeddings once and re-using them without having to worry about disk
# space. Currently, however, every new training run performs precomputation even if it's not required
# (which is something we've to improve [TODO(aryan)])
dataset_cmd=(
  --dataset_config $TRAINING_DATASET_CONFIG
  --dataset_shuffle_buffer_size 10
  --enable_precomputation
  --precomputation_items 15 # 25=50videos/2 
  --precomputation_once
)

# Dataloader arguments
dataloader_cmd=(
  --dataloader_num_workers 0
)

# Diffusion arguments
diffusion_cmd=(
  --flow_weighting_scheme "logit_normal"
)

# Training arguments
# We target just the attention projections layers for LoRA training here.
# You can modify as you please and target any layer (regex is supported)
training_cmd=(
  --training_type "lora"
  --seed 42
  --batch_size 1
  --train_steps 3000
  --rank 32
  --lora_alpha 32
  --target_modules "blocks.*(to_q|to_k|to_v|to_out.0)"
  --gradient_accumulation_steps 1
  --gradient_checkpointing
  --checkpointing_steps 500
  --checkpointing_limit 2
  # --resume_from_checkpoint 3000
  --enable_slicing
  --enable_tiling
)

# Optimizer arguments
optimizer_cmd=(
  --optimizer "adamw"
  --lr 5e-5 # 1e-4 for 14B, 5e-5 for 1.3B
  --lr_scheduler "constant_with_warmup"
  --lr_warmup_steps 1000
  --lr_num_cycles 1
  --beta1 0.9
  --beta2 0.99
  --weight_decay 1e-4
  --epsilon 1e-8
  --max_grad_norm 1.0
)

# Validation arguments
validation_cmd=(
  --validation_dataset_file "$VALIDATION_DATASET_FILE"
  --validation_steps 500
)

# Miscellaneous arguments
miscellaneous_cmd=(
  --tracker_name "finetrainers-wan-wjh"
  --output_dir "/home/wjh/projects/finetrainers/logs" # logs: ln -s /share/wjh/logs/finetrainers/
  --init_timeout 600
  --nccl_timeout 600
  --report_to "wandb"
  --logging_dir 'lg'
  # --logging_dir "/home/wjh/projects/finetrainers/logs/wandb" # logs: ln -s /share/wjh/logs/finetrainers/
)

# Execute the training script
if [ "$BACKEND" == "accelerate" ]; then

  ACCELERATE_CONFIG_FILE=""
  if [ "$NUM_GPUS" == 1 ]; then
    ACCELERATE_CONFIG_FILE="accelerate_configs/uncompiled_1.yaml"
  elif [ "$NUM_GPUS" == 2 ]; then
    ACCELERATE_CONFIG_FILE="accelerate_configs/uncompiled_2.yaml"
  elif [ "$NUM_GPUS" == 4 ]; then
    ACCELERATE_CONFIG_FILE="accelerate_configs/uncompiled_4.yaml"
  elif [ "$NUM_GPUS" == 8 ]; then
    ACCELERATE_CONFIG_FILE="accelerate_configs/uncompiled_8.yaml"
  fi
  
  accelerate launch --config_file "$ACCELERATE_CONFIG_FILE" --gpu_ids $CUDA_VISIBLE_DEVICES train.py \
    "${parallel_cmd[@]}" \
    "${model_cmd[@]}" \
    "${dataset_cmd[@]}" \
    "${dataloader_cmd[@]}" \
    "${diffusion_cmd[@]}" \
    "${training_cmd[@]}" \
    "${optimizer_cmd[@]}" \
    "${validation_cmd[@]}" \
    "${miscellaneous_cmd[@]}"

elif [ "$BACKEND" == "ptd" ]; then

  export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
  
  torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    --rdzv_backend c10d \
    --rdzv_endpoint="localhost:0" \
    train.py \
      "${parallel_cmd[@]}" \
      "${model_cmd[@]}" \
      "${dataset_cmd[@]}" \
      "${dataloader_cmd[@]}" \
      "${diffusion_cmd[@]}" \
      "${training_cmd[@]}" \
      "${optimizer_cmd[@]}" \
      "${validation_cmd[@]}" \
      "${miscellaneous_cmd[@]}"
fi

echo -ne "-------------------- Finished executing script --------------------\n\n"
