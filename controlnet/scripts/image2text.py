from PIL import Image
import json
import os
import random
import shutil
from clip_interrogator import Config, Interrogator
import torch

# Set the device to GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Allocate the maximum available GPU memory
torch.cuda.set_per_process_memory_fraction(1.0, device=device)

vtol_folder_path = "/workspace/OOTDiffusion/controlnet/data/VITON-HD"
test_folder_path = f"{vtol_folder_path}/test"
train_folder_path = f"{vtol_folder_path}/train"

is_train = True

if not is_train:
    all_cloth_names = [img_name for img_name in os.listdir(f"{vtol_folder_path}/test/cloth")]
else:
    all_cloth_names = [img_name for img_name in os.listdir(f"{vtol_folder_path}/train/cloth")]

print(all_cloth_names)

images_path = f"{test_folder_path}/cloth" if not is_train else f"{train_folder_path}/cloth"

ci = Interrogator(Config(clip_model_name="ViT-L-14/openai"))

result = {}

for cloth_name in all_cloth_names:
    image = Image.open(f"{images_path}/{cloth_name}").convert('RGB')
    caption = ci.interrogate(image)
    result[cloth_name] = caption

    if not is_train:
        with open(f"{test_folder_path}/cloth_caption/{os.path.splitext(cloth_name)[0]}.txt", 'w') as f:
            f.write(caption)
    else:
        with open(f"{train_folder_path}/cloth_caption/{os.path.splitext(cloth_name)[0]}.txt", 'w') as f:
            f.write(caption)