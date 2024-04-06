cd controlnet && \
accelerate launch train_controlnet_orig.py  \
    --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5"\
    --output_dir="output"   --dataset_name=fusing/fill50k  \
    --resolution=512  --learning_rate=1e-5  \
    --validation_image "./00008_00.jpg" \
    --validation_image_garm "./00006_00.jpg"  \
    --validation_prompt "levi's t - shirt with logo print, white-space-surrounding, wlop : :, k-pop, listing image, white borders, cartoonish and simplistic, feminine looking, is essentially arbitrary, levers, right align, ad image, white: 0.5, manuka, whealan, contest, stylised, f 1"\
    --train_batch_size=4  --report_to=wandb   \
    --validation_steps=10 \
    --dataroot="/opt/disk1/dwang/sci/DVTON/data/updated-VITON-HD/"
# accelerate launch train_controlnet_orig.py \
#  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
#  --output_dir="output"  \
#  --dataset_name=fusing/fill50k \
#  --resolution=512 \
#  --dataroot="/opt/disk1/dwang/sci/DVTON/data/updated-VITON-HD/" \
#  --learning_rate=1e-5 \
#  --validation_image "./conditioning_image_1.png" "./conditioning_image_2.png" \
#  --validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
#  --train_batch_size=4 \
#  --report_to=wandb  \
#  --validation_steps=10\
#  --validation_image_garm "./00006_00.jpg"