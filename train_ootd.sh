CUDA_VISIBLE_DEVICES=1 python /home/dwang/miniconda3/envs/ootd/bin/accelerate launch \
    --config_file accelerate_config.json train_ootd.py \
    --pretrained_model_name_or_path="checkpoints/ootd/" \
    --mixed_precision="fp16" \
    --output_dir="output/logs/train_ootd" \
    --dataset_name="SaffalPoosh/VITON-HD-test" \
    --resolution="512" \
    --learning_rate="1e-5" \
    --train_batch_size="1" \
    --dataroot="/opt/disk1/dwang/sci/DVTON/data/updated-VITON-HD" \
    --train_data_list="subtrain_0.1.txt" \
    --test_data_list="subtrain_0.1.txt" \
    --num_train_epochs="10" \
    --checkpointing_steps="500" \
    --use_8bit_adam \
    --gradient_checkpointing \
    --enable_xformers_memory_efficient_attention \
    --validation_steps="10" \
    --inference_steps="50" \
    --log_grads \
    --report_to="wandb" \
    --gradient_accumulation_steps="4" 
    # --tracker_project_name="train_OOTDdiffusion" \
    # --tracker_entity="xuziang" \