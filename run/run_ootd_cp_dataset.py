"""Run ootd inference with cp_dataset"""
from pathlib import Path
import sys
from PIL import Image
from utils_ootd import get_mask_location

PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from data_scripts.cp_dataset import CPDatasetV2 as CPDataset

import torch
from preprocess.openpose.run_openpose import OpenPose
from preprocess.humanparsing.aigc_run_parsing import Parsing
from ootd.inference_ootd_hd import OOTDiffusionHD
from ootd.inference_ootd_dc import OOTDiffusionDC


import argparse
parser = argparse.ArgumentParser(description='run ootd')
parser.add_argument('--gpu_id', '-g', type=int, default=0, required=False)
parser.add_argument('--model_type', type=str, default="hd", required=False)
parser.add_argument('--category', '-c', type=int, default=0, required=False)
parser.add_argument('--scale', type=float, default=2.0, required=False)
parser.add_argument('--step', type=int, default=20, required=False)
parser.add_argument('--sample', type=int, default=1, required=False)
parser.add_argument('--seed', type=int, default=-1, required=False)
parser.add_argument("--resolution", type=int, default=512)
parser.add_argument("--dataroot", type=str, default=None)
parser.add_argument("--train_data_list", type=str, default=None)
args = parser.parse_args()


openpose_model = OpenPose(args.gpu_id)
parsing_model = Parsing(args.gpu_id)


category_dict = ['upperbody', 'lowerbody', 'dress']
category_dict_utils = ['upper_body', 'lower_body', 'dresses']

model_type = "hd" # "hd" or "dc"
category = args.category # 0:upperbody; 1:lowerbody; 2:dress

image_scale = args.scale
n_steps = args.step
n_samples = args.sample
seed = args.seed

if model_type == "hd":
    model = OOTDiffusionHD(args.gpu_id)
elif model_type == "dc":
    model = OOTDiffusionDC(args.gpu_id)
else:
    raise ValueError("model_type must be \'hd\' or \'dc\'!")


if __name__ == '__main__':

    if model_type == 'hd' and category != 0:
        raise ValueError("model_type \'hd\' requires category == 0 (upperbody)!")

    train_dataset = CPDataset(args.dataroot, args.resolution, mode="train", data_list=args.train_data_list)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        # collate_fn=collate_fn_cp,
        batch_size=1,
        num_workers=0,
    )
    for step, batch in enumerate(train_dataloader):
        image_garm = batch["ref_imgs"]
        image_vton = batch["inpaint_image"]
        image_ori = batch["GT"]
        inpaint_mask = batch["inpaint_mask"]
        mask = batch["inpaint_mask"]
        prompt = batch["prompt"]
        file_name = batch["file_name"]
        
        images = model(
            model_type=model_type,
            category=category_dict[category],
            image_garm=image_garm,
            image_vton=image_vton,
            mask=mask,
            image_ori=image_ori,
            num_samples=n_samples,
            num_steps=n_steps,
            image_scale=image_scale,
            seed=seed,
        )

        image_idx = 0
        for image in images:
            image.save(f"./images_output/out_{file_name}_{model_type}_{image_idx}.png")
            image_idx += 1
