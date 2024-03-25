from pathlib import Path
import sys
from PIL import Image
from run.utils_ootd import get_mask_location
import torch
import torchvision

PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from preprocess.openpose.run_openpose import OpenPose
from preprocess.humanparsing.aigc_run_parsing import Parsing
from ootd.inference_ootd_hd import OOTDiffusionHD
from ootd.inference_ootd_dc import OOTDiffusionDC
from data_scripts.cp_dataset import CPDataset


import argparse
parser = argparse.ArgumentParser(description='run ootd')
parser.add_argument('--gpu_id', '-g', type=int, default=0, required=False)
parser.add_argument('--model_path', type=str, default="", required=False)
parser.add_argument('--cloth_path', type=str, default="", required=False)
parser.add_argument('--model_type', type=str, default="hd", required=False)
parser.add_argument('--category', '-c', type=int, default=0, required=False)
parser.add_argument('--scale', type=float, default=2.0, required=False)
parser.add_argument('--step', type=int, default=20, required=False)
parser.add_argument('--sample', type=int, default=4, required=False)
parser.add_argument('--seed', type=int, default=-1, required=False)
args = parser.parse_args()


openpose_model = OpenPose(args.gpu_id)
parsing_model = Parsing(args.gpu_id)


category_dict = ['upperbody', 'lowerbody', 'dress']
category_dict_utils = ['upper_body', 'lower_body', 'dresses']

model_type = args.model_type # "hd" or "dc"
category = args.category # 0:upperbody; 1:lowerbody; 2:dress
cloth_path = args.cloth_path
model_path = args.model_path

image_scale = args.scale
n_steps = args.step
n_samples = args.sample
seed = args.seed

dataroot = "../data/VITON-HD/"


if model_type == "hd":
    model = OOTDiffusionHD(args.gpu_id)
elif model_type == "dc":
    model = OOTDiffusionDC(args.gpu_id)
else:
    raise ValueError("model_type must be \'hd\' or \'dc\'!")


def save_image(data, image_name=""):
    cloth_path = f"test{image_name}.jpg"
    torchvision.utils.save_image(data, cloth_path)


if __name__ == '__main__':

    if model_type == 'hd' and category != 0:
        raise ValueError("model_type \'hd\' requires category == 0 (upperbody)!")

    # cloth_img = Image.open(cloth_path).resize((768, 1024))
    # model_img = Image.open(model_path).resize((768, 1024))
    # keypoints = openpose_model(model_img.resize((384, 512)))
    # model_parse, _ = parsing_model(model_img.resize((384, 512)))

    # mask, mask_gray = get_mask_location(model_type, category_dict_utils[category], model_parse, keypoints)
    # mask = mask.resize((768, 1024), Image.NEAREST)
    # mask_gray = mask_gray.resize((768, 1024), Image.NEAREST)
    
    # masked_vton_img = Image.composite(mask_gray, model_img, mask)
    # masked_vton_img.save('./images_output/mask.jpg')

    train_dataset = CPDataset(dataroot, mode="train")
    test_dataset = CPDataset(dataroot, mode="test")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        # collate_fn=collate_fn_cp,
        batch_size=1,
        num_workers=1,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=True,
        # collate_fn=collate_fn_cp,
        batch_size=1,
        num_workers=1,
    )

    for step, batch in enumerate(train_dataloader):
        with torch.autocast("cuda"):
            image_garm = batch["ref_imgs"]
            image_vton = batch["inpaint_image"]
            image_ori = batch["GT"]
            mask = batch["inpaint_mask"]
            prompt = batch["prompt"]
            import ipdb; ipdb.set_trace()
            # dict_keys(['GT', 'inpaint_image', 'inpaint_pa', 'inpaint_mask', 'ref_imgs', 'warp_feat', 'file_name', 'cloth_array', 'input_ids', 'prompt'])
            save_image( batch["inpaint_image"], "inpaint_image")
            save_image( batch["ref_imgs"], "ref_imgs")
            save_image( batch["GT"], "GT")
            save_image( batch["inpaint_pa"], "inpaint_pa")
            batch["inpaint_mask"]
            
            model_img = batch["GT"]

            import ipdb; ipdb.set_trace()
            ref_imgs = Image.fromarray(
                (batch["ref_imgs"].squeeze().permute(1, 2, 0).numpy() * 255).astype('uint8'))
            
            images = model(
                model_type=model_type,
                category=category_dict[category],
                image_garm=ref_imgs,
                image_vton=batch["inpaint_image"],
                mask=batch["inpaint_mask"],
                image_ori=batch["GT"],
                num_samples=n_samples,
                num_steps=n_steps,
                seed=seed,
            )

        # FIXME: check if parameters in the model are leaf?
        if all(p.is_leaf for p in model.pipe.unet_vton.parameters()):
            # raise ValueError(f"Model parameters are all leaf.")
            print(f"{__name__},loaded unet_vton parameters are all leaf.")

        image_idx = 0
        for image in images:
            image.save('./images_output/out_' + model_type + '_' + str(image_idx) + '.png')
            image_idx += 1
