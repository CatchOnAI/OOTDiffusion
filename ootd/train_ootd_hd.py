import pdb
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).absolute().parents[0].absolute()
sys.path.insert(0, str(PROJECT_ROOT))
import os
import torch
import numpy as np
from PIL import Image
import cv2

import random
import time
import pdb

from ootd.pipelines_ootd.pipeline_ootd_train import OotdPipeline
from pipelines_ootd.unet_garm_2d_condition import UNetGarm2DConditionModel
from pipelines_ootd.unet_vton_2d_condition import UNetVton2DConditionModel
from diffusers import DDPMScheduler
from diffusers import AutoencoderKL
from diffusers.utils import (
    PIL_INTERPOLATION,
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)

import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, CLIPVisionModelWithProjection
from transformers import CLIPTextModel, CLIPTokenizer
from safetensors.torch import load_file

class OOTDiffusionHD:

    def __init__(self, gpu_id, model_path, **kwargs):
        self.gpu_id = 'cuda:' + str(gpu_id)

        MODEL_PATH = model_path
        UNET_PATH = kwargs["unet_path"] if "unet_path" in kwargs else MODEL_PATH
        GARM_UNET_PATH = kwargs["garm_unet_path"] if "garm_unet_path" in kwargs else UNET_PATH
        VTON_UNET_PATH = kwargs["vton_unet_path"] if "vton_unet_path" in kwargs else UNET_PATH
        VIT_PATH = kwargs["vit_path"] if "vit_path" in kwargs else MODEL_PATH
        VAE_PATH = kwargs["vae_path"] if "vae_path" in kwargs else MODEL_PATH

        self.vae = AutoencoderKL.from_pretrained(
            VAE_PATH,
            subfolder="vae",
            torch_dtype=torch.float16,
        )
    
        # unet_sd = load_file(f"{MODEL_PATH}/diffusion_pytorch_model.safetensors")
        self.unet_garm = UNetGarm2DConditionModel.from_pretrained(
            GARM_UNET_PATH,
            subfolder="unet_garm",
            # torch_dtype=torch.float16,
            use_safetensors=True,
            local_files_only=True,
            low_cpu_mem_usage=False,
            ignore_mismatched_sizes=True
        )

        self.unet_vton = UNetVton2DConditionModel.from_pretrained(
            VTON_UNET_PATH,
            subfolder="unet_vton",
            # torch_dtype=torch.float16,
            use_safetensors=True,
            local_files_only=True,
            low_cpu_mem_usage=False,
            ignore_mismatched_sizes=True
        )
        
        self.auto_processor = AutoProcessor.from_pretrained(VIT_PATH)
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(VIT_PATH).to(self.gpu_id)

        self.tokenizer = CLIPTokenizer.from_pretrained(
            MODEL_PATH,
            subfolder="tokenizer",
        )

        self.text_encoder = CLIPTextModel.from_pretrained(
            MODEL_PATH,
            subfolder="text_encoder",
        ).to(self.gpu_id)

        self.scheduler = DDPMScheduler.from_pretrained(
            MODEL_PATH, 
            subfolder="scheduler"
            )
        
        self.pipe = OotdPipeline.from_pretrained(
            MODEL_PATH,
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet_garm=self.unet_garm,
            unet_vton=self.unet_vton,
            scheduler=self.scheduler,
            use_safetensors=True,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(self.gpu_id)
        
    def tokenize_captions(self, captions, max_length):
        inputs = self.tokenizer(
            captions, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids


    def __call__(self,
                model_type='hd',
                category='upperbody',
                image_garm=None,
                image_vton=None,
                image_ori=None,
                mask=None,
                prompt=None,
                prompt_embeds = None,
                negative_prompt = None,
                seed=-1,
                **kwargs
    ):  
        if seed == -1:
            random.seed(time.time())
            seed = random.randint(0, 2147483647)
        print('Initial seed: ' + str(seed))
        generator = torch.manual_seed(seed)

        # TODO: text prompt is not actually used. The prompt embs are based on empty str!
        prompt_image = self.auto_processor(images=image_garm, return_tensors="pt").to(self.gpu_id)
        prompt_image = self.image_encoder(prompt_image.data['pixel_values']).image_embeds
        prompt_image = prompt_image.unsqueeze(1)
        prompt_embeds = self.text_encoder(self.tokenize_captions([" "]*prompt_image.shape[0], 2).to(self.gpu_id))[0]
        # prompt_embeds = self.text_encoder(self.tokenize_captions([prompt]*prompt_image.shape[0], 2).to(self.gpu_id))[0]

        prompt_embeds[:, 1:] = prompt_image[:]

        noise_pred, noise = self.pipe(
            prompt_embeds=prompt_embeds,
            image_garm=image_garm,
            image_vton=image_vton,
            mask=mask,
            image_ori=image_ori, 
            generator=generator,
        )
        return noise_pred, noise
