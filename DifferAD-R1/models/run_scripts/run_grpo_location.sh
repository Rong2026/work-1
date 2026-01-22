export DEBUG_MODE="true"

# RUN_NAME="univg_12_20_dapo_second_continue_loss_difficulty_aware_anomaladd_combined_reward_prompt_new"
RUN_NAME="1_20_full_new"
#12_17_dapo_second_continue_loss_difficulty_aware_anomalplus_combined_reward_prompt
export LOG_PATH="./debug_log_$RUN_NAME.txt"
# MODEL_NAME=/path/to/your/stage1_cotsft_model
# export MODEL_NAME="path/to/Qwen2.5-VL-7B-Instruct"
export MODEL_NAME="path/to/Qwen2-VL-UniVG-R1"
export CUDA_VISIBLE_DEVICES="0,1,2,3"

torchrun --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
    path/to/open_r1/grpo_location_seg.py \
    --deepspeed path/to/local_scripts/zero3.json \
    --output_dir path/to/output_1_20/$RUN_NAME \
    --model_name_or_path $MODEL_NAME \
    --dataset_name path/to/dataset/train_coco_BG_MCH_sys.jsonl \
    --image_root path/to/dataset/dataset/mvisa/data \
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
    # --sample_size 5000 \
    # --miou_adjust True \
    # --miou_adjust_math exp