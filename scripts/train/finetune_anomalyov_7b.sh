# export CUDA_HOME=/data/02/jiacong/cuda-12.1
export OMP_NUM_THREADS=16
#export NCCL_IB_DISABLE=1
#export NCCL_IB_GID_INDEX=3
#export NCCL_SOCKET_IFNAME=eno1
#export NCCL_DEBUG=INFO

#export NCCL_P2P_DISABLE=1
#export NCCL_IB_DISABLE=1
#export NCCL_SOCKET_IFNAME=eth0
#export NCCL_SHM_DISABLE=1

LLM_VERSION="Qwen/Qwen2-7B-Instruct" 
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

############### Pretrain ################

BASE_RUN_NAME="anomalyov_7B_finetune_llm_and_projector_all_data"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

############### Finetune ################

# Stage 2
PROMPT_VERSION="qwen_1_5"
RUN_NAME="anomalyov_7B_finetune_llm_and_projector_all_data" 
PREV_STAGE_CHECKPOINT="lmms-lab/llava-onevision-qwen2-7b-ov"
echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "MID_RUN_NAME: ${RUN_NAME}"

NUM_GPUS=4

# deepspeed.ops.op_builder.builder.CUDAMismatchException
export DS_SKIP_CUDA_CHECK=1
# /usr/bin/ld: cannot find -lcurand: No such file or directory
export LIBRARY_PATH="/usr/local/cuda/lib64:$LIBRARY_PATH"
export WANDB_MODE=offline

# export NCCL_P2P_DISABLE=1
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
# export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128


if [ ! -d "/data/02/jiacong" ]; then
    echo "Directory /data/02/jiacong does not exist. Creating it..."
    sudo mkdir -p /data/02/jiacong
    sudo ln -s $(pwd)/data /data/02/jiacong
else
    echo "Directory /data/02/jiacong already exists."
fi

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node="${NUM_GPUS}" \
    llava/train/train_mem.py \
    --deepspeed scripts/zero2_offload.json \
    --lora_enable True \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version $PROMPT_VERSION \
    --data_path data/datasets.yaml \
    --image_folder data \
    --video_folder None \
    --mm_tunable_parts="mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=1e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints  "(1x1),...,(4x4)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $RUN_NAME \
    --output_dir checkpoints/$RUN_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 3 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --frames_upbound 32
exit 0;

# You can delete the sdpa attn_implementation if you want to use flash attn
