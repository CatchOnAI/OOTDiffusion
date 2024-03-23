export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="output"

CUDA_VISIBLE_DEVICES=1 accelerate launch --config_file accelerate_config.json controlnet/train_controlnet.py \
    --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
    --output_dir="output" --dataset_name=SaffalPoosh/VITON-HD-test  \
    --resolution=512  --learning_rate=1e-5  \
    --validation_image_garm "controlnet/00006_00.jpg"  \
    --validation_image "controlnet/00008_00.jpg" --validation_original_image "controlnet/00008_00.jpg" \
    --validation_prompt "a cloth"   \
    --train_batch_size=4  --dataroot=../data/VITON-HD --report_to wandb \
    --validation_steps 1