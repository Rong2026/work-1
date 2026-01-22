export DEBUG_MODE="true"

# RUN_NAME="univg_12_20_dapo_second_continue_loss_difficulty_aware_anomaladd_combined_reward_prompt_new"
RUN_NAME="ablation_5000_samples"
#12_17_dapo_second_continue_loss_difficulty_aware_anomalplus_combined_reward_prompt
export LOG_PATH="./debug_log_$RUN_NAME.txt"
# MODEL_NAME=/path/to/your/stage1_cotsft_model
# export MODEL_NAME="path/to"
# export MODEL_NAME="path/to"
export MODEL_NAME="path/to"
# export MODEL_NAME="path/to"
# export MODEL_NAME="path/to"
# export MODEL_NAME="path/to"
# export MODEL_NAME="path/to"
# export MODEL_NAME="path/to"
# export MODEL_NAME="path/to"
# export MODEL_NAME="path/to"
# export MODEL_NAME="path/to"
export CUDA_VISIBLE_DEVICES="0,1,2,3"

# torchrun --nproc_per_node=4 \
#     --nnodes=1 \
#     --node_rank="0" \
#     --master_addr="127.0.0.1" \
#     --master_port="12346" \
#     path/to
#     --deepspeed path/to
#     --output_dir path/to
#     --model_name_or_path $MODEL_NAME \
#     --dataset_name path/to
#     --image_root path/to
#     --freeze_vision_modules true \
#     --max_prompt_length 1024 \
#     --num_generations 8 \
#     --per_device_train_batch_size 8 \
#     --gradient_accumulation_steps 2 \
#     --logging_steps 1 \
#     --bf16 \
#     --torch_dtype bfloat16 \
#     --data_seed 42 \
#     --report_to none \
#     --gradient_checkpointing true \
#     --attn_implementation flash_attention_2 \
#     --num_train_epochs 1 \
#     --run_name $RUN_NAME \
#     --save_steps 1000 \
#     --save_only_model true \
#     --single_img_with_cot False \
#     # --miou_adjust True \
#     # --miou_adjust_math exp

torchrun --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
    path/to
    --deepspeed path/to
    --output_dir path/to
    --model_name_or_path $MODEL_NAME \
    --dataset_name path/to
    --image_root path/to
    --freeze_vision_modules true \
    --max_prompt_length 1024 \
    --num_generations 8 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to none \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $RUN_NAME \
    --save_steps 1500 \
    --save_only_model true \
    --single_img_with_cot False \
    --trainer_type dapo \
    --max_resample_times 1 \
    --dynamic_sampling True \
    --difficulty_adjust True \
    --iou_center True \
    --sample_size 5000 \
    # --miou_adjust True \
    # --miou_adjust_math exp