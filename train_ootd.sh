accelerate launch train_ootd.py \
    --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
    --mixed_precision="fp16" \
    --output_dir="/home/ubuntu/ouput" \
    --dataset_name="SaffalPoosh/VITON-HD-test" \
    --resolution="512" \
    --learning_rate="1e-5" \
    --train_batch_size="1" \
    --dataroot="/home/ubuntu/OOTDiffusion/controlnet/data/VITON-HD" \
    --train_data_list="subtrain_20.txt" \
    --test_data_list="subtrain_20_bk.txt" \
    --num_train_epochs="100" \
    --checkpointing_steps="500" \
    --gradient_checkpointing \
    --validation_steps="30" \
    --inference_steps="50" \
    --log_grads \
    --report_to="wandb" \
    --seed="42" \
    --gradient_accumulation_steps="4" \
    # --tracker_project_name="train_OOTDdiffusion" \
    # --tracker_entity="xuziang" \
    # --enable_xformers_memory_efficient_attention \
    # --use_8bit_adam \
