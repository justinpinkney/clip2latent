import sys

import clip
import gradio as gr
import hydra
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import train
import yaml
from dalle2_pytorch import DiffusionPriorNetwork
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from tqdm.auto import tqdm

sys.path.append("..")
sys.path.append("../stylegan3")
import dnnlib
import legacy
from clip2latent import train_utils
from clip2latent.latent_prior import LatentPrior
from clip2latent.train_utils import (compute_val, denormalise_data, make_grid,
                                     make_image_val_data, make_text_val_data)

checkpoint = "best.ckpt"
use_ema = True
device = "cuda:0"
data = torch.load(checkpoint, map_location="cpu")
# cfg = data["cfg"]
# cfg_file = "/home/jpinkney/code/clip2latent/outputs/2022-05-21/15-40-12/.hydra/config.yaml"
cfg_file = "best.yaml"
cfg = OmegaConf.load(cfg_file)
print(cfg)

G, clip_model, trainer = train.load_models(cfg, device)

trainer.load_state_dict(torch.load(checkpoint, map_location="cpu")["state_dict"], strict=False)
diffusion_prior = trainer.diffusion_prior

n_samples = 16
skips = {
    "Slow": 1,
    "Med": 10,
    "Fast": 100,
}

@torch.no_grad()
def run_model(text_samples, cond_scale, skip_name, select_best):
    diffusion_prior.set_timestep_skip(skips[skip_name])
    diffusion_prior.eval()
    images = []
    text_features = clip_model.embed_text(text_samples)
    out = diffusion_prior.sample(text_features.tile(n_samples, 1), cond_scale=cond_scale)

    pred_w_clip_features = []
    pred_w = out
    for w in tqdm(pred_w):
        out = G.synthesis(w.unsqueeze(0))
        images.append(out)
        image_features = clip_model.embed_image(out)
        pred_w_clip_features.append(image_features)

    pred_w_clip_features = torch.cat(pred_w_clip_features, dim=0)
    sim = torch.cosine_similarity(pred_w_clip_features, text_features)
    sim, idxs = torch.sort(sim, descending=True)
    images = torch.cat(images, dim=0)
    images = images[idxs, ...]
    print(sim)

    if select_best == "Best":
        im_to_show = images[0]
        resize = False
    else:
        im_to_show = images
        resize = True
    grid = train_utils.make_grid(im_to_show, resize=resize)

    return grid
text_input = gr.Textbox("An old man with a beard", lines=1, max_lines=1, placeholder="Enter a text description", label="Description")
cond_input = gr.Slider(-10, 10, 1, step=1, label="Condition scale")
skip_timesteps = gr.Radio(list(skips.keys()), value="Fast", label="Sampling")
select_best = gr.Radio(["Best", "All"], value="Best", label="Show best (at higher resolution) or show all.")

output_image = gr.Image(shape=(1024,1024), label="Generated faces")

demo = gr.Interface(fn=run_model, inputs=[text_input, cond_input, skip_timesteps, select_best], outputs=output_image)

demo.launch(enable_queue=False)

