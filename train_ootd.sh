accelerate launch train_ootd.py \
    --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
    --mixed_precision="fp16" \
    --output_dir="/home/stevexu/VSprojects/OOTDiffusion/output/logs/train_ootd" \
    --dataset_name="SaffalPoosh/VITON-HD-test" \
    --resolution="512" \
    --learning_rate="1e-5" \
    --train_batch_size="1" \
    --dataroot="/home/stevexu/data/VITON-HD" \
    --data_list="subtrain_0.1.txt" \
    --num_train_epochs="10" 
    # --report_to="wandb"