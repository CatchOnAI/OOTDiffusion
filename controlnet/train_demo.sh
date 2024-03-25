OUTPUT_DIR="/home/stevexu/VSprojects/OOTDiffusion/output/logs/train_controlnet"
MODEL_DIR="runwayml/stable-diffusion-v1-5"

accelerate launch train_controlnet.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=SaffalPoosh/VITON-HD-test \
 --resolution=512  \
 --learning_rate=1e-5  \
 --validation_image "/home/stevexu/VSprojects/OOTDiffusion/controlnet/conditioning_image_1.png" "/home/stevexu/VSprojects/OOTDiffusion/controlnet/conditioning_image_2.png"  \
 --validation_prompt "red circle with blue background" "cyan circle with brown floral background"  \
 --validation_steps 200 \
 --train_batch_size=4  \
 --dataroot=/home/stevexu/data/VITON-HD  \
 --data_list="subtrain_20.txt"  \
 --report_to="wandb"  