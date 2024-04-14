accelerate launch train_ootd.py \
    --pretrained_model_name_or_path="checkpoints/ootd_org" \
    --mixed_precision="fp16" \
    --output_dir="output/logs/train_ootd" \
    --dataset_name="SaffalPoosh/VITON-HD-test" \
    --resolution="512" \
    --learning_rate="1e-5" \
    --train_batch_size="1" \
    --dataroot="../data/VITON-HD" \
    --train_data_list="subtrain_0.1.txt" \
    --validation_data_list="subtrain_0.1.txt" \
    --test_data_list="subtest_20.txt" \
    --num_train_epochs="150" \
    --checkpointing_steps="5000" \
    --gradient_checkpointing \
    --validation_steps="500" \
    --inference_steps="30" \
    --log_grads \
    --report_to="wandb" \
    --seed="42" \
    --gradient_accumulation_steps="4" 
    # --tracker_project_name="train_OOTDdiffusion" \
    # --tracker_entity="xuziang" \
    # --enable_xformers_memory_efficient_attention \
    # --use_8bit_adam \