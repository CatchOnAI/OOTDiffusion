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
from diffusers import UniPCMultistepScheduler
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

    def __init__(self, gpu_id, use_sd=False, **kwargs):
        self.gpu_id = 'cuda:' + str(gpu_id)

        VAE_PATH = kwargs["vae_path"]
        MODEL_PATH = kwargs["model_path"]
        VIT_PATH = kwargs["vit_path"]
        
        if use_sd:
            self.vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5")
        else:
            self.vae = AutoencoderKL.from_pretrained(
                VAE_PATH,
                subfolder="vae",
                # torch_dtype=torch.float16,
            )
    
        # unet_sd = load_file(f"{MODEL_PATH}/diffusion_pytorch_model.safetensors")
        if use_sd:
            self.unet_garm = UNetGarm2DConditionModel.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                # torch_dtype=torch.float16,
                use_safetensors=True,
                low_cpu_mem_usage=False,
            )
        else:
            self.unet_garm = UNetGarm2DConditionModel.from_pretrained(
                MODEL_PATH,
                subfolder="unet_garm",
                # torch_dtype=torch.float16,
                use_safetensors=True,
                local_files_only=True,
                low_cpu_mem_usage=False,
                ignore_mismatched_sizes=True
            )

        if use_sd:
            self.unet_vton = UNetVton2DConditionModel.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                # torch_dtype=torch.float16,
                use_safetensors=True,
                low_cpu_mem_usage=False,
            )
        else:
            self.unet_vton = UNetVton2DConditionModel.from_pretrained(
                MODEL_PATH,
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
            VAE_PATH,
            subfolder="tokenizer",
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            VAE_PATH,
            subfolder="text_encoder",
        ).to(self.gpu_id)

        self.pipe = OotdPipeline.from_pretrained(
            VAE_PATH,
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet_garm=self.unet_garm,
            unet_vton=self.unet_vton,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(self.gpu_id)

        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)


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
                prompt='',
                negative_prompt='',
                **kwargs
    ):  
        noise_pred, noise = self.pipe(
            prompt=prompt,
            image_garm=image_garm,
            image_vton=image_vton,
            mask=mask,
            image_ori=image_ori, 
        )
        return noise_pred, noise
