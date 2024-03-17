from PIL import Image
import json
import os
import random
import shutil
from clip_interrogator import Config, Interrogator
import torch
import tqdm 
import multiprocessing
from functools import partial
import time

# Define a function to interrogate an image
def interrogate_image(image_path, ci, is_train, test_folder_path, train_folder_path):
    image = Image.open(image_path).convert('RGB')
    caption = ci.interrogate(image)
    cloth_name = os.path.basename(image_path)
    result = (cloth_name, caption)
    print(f"Process {image_path}")

    if not is_train:
        with open(f"{test_folder_path}/cloth_caption/{os.path.splitext(cloth_name)[0]}.txt", 'w') as f:
            f.write(caption)
    else:
        with open(f"{train_folder_path}/cloth_caption/{os.path.splitext(cloth_name)[0]}.txt", 'w') as f:
            f.write(caption)

    return result

def main(num_workers=2):
    # Set the device to GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Allocate the maximum available GPU memory
    torch.cuda.set_per_process_memory_fraction(1.0, device=device)

    vtol_folder_path = "/opt/users/dwang/datasets/VITON-HD_ori"
    test_folder_path = f"{vtol_folder_path}/test"
    train_folder_path = f"{vtol_folder_path}/train"

    is_train = True

    if not is_train:
        all_cloth_names = [img_name for img_name in os.listdir(f"{vtol_folder_path}/test/cloth")]
    else:
        all_cloth_names = [img_name for img_name in os.listdir(f"{vtol_folder_path}/train/cloth")]

    all_cloth_names = all_cloth_names[:10]
    print(all_cloth_names)

    images_paths = f"{test_folder_path}/cloth" if not is_train else f"{train_folder_path}/cloth"

    ci = Interrogator(Config(clip_model_name="ViT-L-14/openai", chunk_size=4096, quiet=True))

    start = time.time()

    if num_workers >1:
        # Define a partial function to pass additional arguments to the interrogate_image function
        partial_interrogate = partial(
            interrogate_image, ci=ci, is_train=is_train, test_folder_path=test_folder_path, train_folder_path=train_folder_path
        )

        # Use multiprocessing Pool to parallelize the interrogation process
        with multiprocessing.get_context("spawn").Pool(num_workers) as pool:
            result = dict(pool.map(partial_interrogate, [os.path.join(images_paths, cloth_name) for cloth_name in all_cloth_names]))
    else:
        for cloth_name in all_cloth_names:
            img_path = os.path.join(os.path.join(images_paths, cloth_name))
            interrogate_image(img_path, ci, is_train, test_folder_path,train_folder_path)

    end = time.time()
    print(f"takes {end - start}")

if __name__ == "__main__":
    num_workers = 1
    main(num_workers)
    
   