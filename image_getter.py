import os, sys, glob
import time
import streamlit as st
from einops import rearrange
from pytorch_lightning import seed_everything
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from tqdm import tqdm, trange

@st.cache(ttl=30*60)
def get_image(prompt, seed=1):
    time.sleep(5)
    return "ai_pan.png"

# Copied from https://github.com/CompVis/stable-diffusion/blob/ce05de28194041e030ccfc70c635fe3707cdfc30/scripts/txt2img.py#
def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

#basically follow what they do in the example script, return PIL.Image
#@st.cache(ttl=30*60)
def text_to_image(prompt, steps, image_height, image_width, scale, seed):
    seed_everything(seed)
    config = OmegaConf.load("configs/v1-inference.yaml")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = load_model_from_config(config, "sd-v1-4.ckpt")
    model = model.to(device)    

    sampler = DDIMSampler(model)
    batch_size = 1
    n_rows = 1
    data = [batch_size * [prompt]]
    start_code = None
    precision_scope = torch.autocast

    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                for n in trange(2, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)
                        shape = [4, image_height // 8, image_width // 8]
                        samples_ddim, _ = sampler.sample(S=steps,
                                                         conditioning=c,
                                                         batch_size=1,
                                                         shape=shape,
                                                         verbose=False,
                                                         unconditional_guidance_scale=scale,
                                                         unconditional_conditioning=uc,
                                                         eta=0,
                                                         x_T=start_code)

                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        
                        for x_sample in x_samples_ddim:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            return Image.fromarray(x_sample.astype(np.uint8))
