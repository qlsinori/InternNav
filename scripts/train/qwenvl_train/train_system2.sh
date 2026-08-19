#!/bin/bash
# 中文训练代码导读（数据、输出与 loss）：
# docs/internvla_n1_training_guide/README.md#6-阶段-a训练-system-2
#SBATCH -J qwenvl
#SBATCH -p gpu_partition
#SBATCH -N 8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH -o ./slurm-%j.out
#SBATCH -e ./slurm-%j.err

# Distributed training configuration
MASTER_ADDR=`scontrol show hostname $SLURM_JOB_NODELIST | head -n1`
MASTER_PORT=$((RANDOM % 101 + 20001))

# DeepSpeed configuration
deepspeed=scripts/train/qwenvl_train/zero2.json

# Model configuration
llm=Qwen/Qwen2.5-VL-7B-Instruct

# Training hyperparameters
lr=2e-5
vision_tower_lr=5e-6
batch_size=2
grad_accum_steps=1
max_pixels=313600
min_pixels=3136

# Stage A 使用 LeRobot 轨迹数据构造三类监督：像素 waypoint 文本、转向箭头和 STOP。
# 这里是数据配置名，不是 raw_data 下供 Habitat 评测使用的 episode JSON。
vln_datasets=r2r_125cm_0_30,r2r_125cm_0_45,r2r_60cm_15_15,r2r_60cm_30_30,rxr_125cm_0_30,rxr_125cm_0_45,rxr_60cm_15_15,rxr_60cm_30_30 #,scalevln_125cm_0_30,scalevln_60cm_30_30

# Output configuration
run_name=InternVLA-N1-System2
output_dir=checkpoints/${run_name}

# pixel_goal_only=False 保留 waypoint、turn、STOP 三类样本；system1=none 使本阶段
# 走 Qwen2.5-VL 原生 next-token CE，而不会构造连续轨迹的 Flow Matching loss。
srun torchrun --nnodes=$SLURM_NNODES --nproc_per_node=8 \
    --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    internnav/trainer/internvla_n1_trainer.py \
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --vln_dataset_use ${vln_datasets} \
    --data_flatten False \
    --tune_mm_vision True \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    \
    --num_history 8 \
    --data_augmentation True \
    --resize_h 384 \
    --resize_w 384 \
    --sample_step 4 \
    --num_future_steps 4 \
    --predict_step_num 32 \
    --pixel_goal_only False \
    --system1 "none" \
    \
    --output_dir ${output_dir} \
    --num_train_epochs 2.0 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels ${max_pixels} \
    --min_pixels ${min_pixels} \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 5 \
    --learning_rate ${lr} \
    --vision_tower_lr ${vision_tower_lr} \
    --weight_decay 0 \
    --warmup_ratio 0.003 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --run_name ${run_name} \
    --report_to wandb
