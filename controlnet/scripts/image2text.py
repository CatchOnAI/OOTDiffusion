from PIL import Image
import json  
import os 
import random 
import shutil
from clip_interrogator import Config, Interrogator


vtol_folder_path = "controlnet/data/VITON-HD/"
test_folder_path = f"{vtol_folder_path}/test"
train_folder_path = f"{vtol_folder_path}/train"


all_cloth_names = [img_name for img_name in os.listdir(f"{vtol_folder_path}/test/cloth")]
sub_cloth_names = set(random.choices(all_cloth_names, k=128))

percent = 20 

def keep_certain_rows(file_path, condition, newfile_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Keep only lines that meet the condition
    # kept_lines = [' '.join([line.split(' ')[1][:-1], line.split(' ')[1][:-1]+'\n']) for line in lines if condition(line)]
    kept_lines = [line for line in lines if condition(line)]

    # Write the kept lines to a new file
    with open(newfile_path, 'w') as file:
        file.writelines(kept_lines)

# Define your condition here
def condition(line):
    # return line.split(' ')[1][:-1] in sub_cloth_names
    return random.random() < percent/100

# Call the function
keep_certain_rows(f'{vtol_folder_path}/train_pairs.txt', condition, f'{vtol_folder_path}/subtrain_{percent}.txt')


images_path = f"{test_folder_path}/cloth"
ci = Interrogator(Config(clip_model_name="ViT-L-14/openai"))

result = {}
for cloth_name in sub_cloth_names:
    image = Image.open(f"{images_path}/{cloth_name}").convert('RGB')
    caption = ci.interrogate(image)
    result[cloth_name] = caption
