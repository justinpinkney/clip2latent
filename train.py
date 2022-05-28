from datetime import datetime
import logging
from pathlib import Path
import sys
from tkinter import X
import wandb
import yaml
import hydra
from omegaconf import DictConfig, OmegaConf

import clip
import torch
from tqdm.auto import tqdm

from dalle2_pytorch import DiffusionPriorNetwork
from dalle2_pytorch.train import DiffusionPriorTrainer

sys.path.append("stylegan3")
import dnnlib
import legacy
import torch

from clip2latent import train_utils
from clip2latent.train_utils import compute_val, make_grid, make_image_val_data, make_text_val_data
from clip2latent.latent_prior import WPlusPriorNetwork, LatentPrior

logger = logging.getLogger(__name__)


class Checkpointer():
    def __init__(self, directory, checkpoint_its):
        directory = Path(directory)
        self.directory = directory
        self.checkpoint_its = checkpoint_its
        if not directory.exists():
            directory.mkdir(parents=True)

    def save_checkpoint(self, model, iteration):
        if iteration % self.checkpoint_its:
            return

        k_it = iteration // 1000
        filename = self.directory/f"{k_it:06}.ckpt"
        checkpoint = {"state_dict": model.state_dict()}
        if hasattr(model, "cfg"):
            checkpoint["cfg"] = model.cfg

        print(f"Saving checkpoint to {filename}")
        torch.save(checkpoint, filename)



def validation(current_it, device, diffusion_prior, stats, G, clip_model, val_data):
    image_result, ims = compute_val(diffusion_prior, val_data["val_im"], G, clip_model, device, stats)
    val = image_result.mean()
    wandb.log({'val/image-similiariy': val}, step=current_it)

    single_im = {"clip_features": val_data["val_im"]["clip_features"][0].tile(8,1)}
    image_result, ims = compute_val(diffusion_prior, single_im, G, clip_model, device, stats)
    val = image_result.mean()
    wandb.log({'val/image-vars': val}, step=current_it)
    wandb.log({'val/image/im-variations': wandb.Image(make_grid(ims))}, step=current_it)

    text_result, ims = compute_val(diffusion_prior, val_data["val_text"], G, clip_model, device, stats)
    val = text_result.mean()
    wandb.log({'val/text': val}, step=current_it)
    wandb.log({'val/image/text2im': wandb.Image(make_grid(ims))}, step=current_it)

    text_result, ims = compute_val(diffusion_prior, val_data["val_text"], G, clip_model, device, stats, cond_scale=3.0)
    val = text_result.mean()
    wandb.log({'val/text-super': val}, step=current_it)
    wandb.log({'val/image/text2im-super': wandb.Image(make_grid(ims))}, step=current_it)


def train_step(diffusion_prior, device, batch):
    diffusion_prior.train()
    batch_z, batch_w = batch
    batch_z = batch_z.to(device)
    batch_w = batch_w.to(device)

    loss = diffusion_prior(batch_z, batch_w)
    loss.backward()
    return loss

    
def train(diffusion_prior, trainer, loader, device, stats, G, clip_model, val_data, val_it, print_it, save_checkpoint, max_it):
    
    current_it = 0
    current_epoch = 0

    while current_it < max_it:
    
        wandb.log({'epoch': current_epoch}, step=current_it)
        pbar = tqdm(loader)
        for batch in pbar:

            if current_it % val_it == 0:
                validation(current_it, device, trainer, stats, G, clip_model, val_data)


            diffusion_prior.train()
            trainer.train()
            batch_clip, batch_latent = batch
            
            input_args = {
                "image_embed": batch_latent.to(device),
                "text_embed": batch_clip.to(device)
            }
            loss = trainer(**input_args)

            if (current_it % print_it == 0):
                wandb.log({'loss': loss}, step=current_it)
            
            trainer.update()
            current_it += 1
            pbar.set_postfix({"loss": loss, "epoch": current_epoch, "it": current_it})

            save_checkpoint(trainer, current_it)

        current_epoch += 1


@hydra.main(config_path="config", config_name="config")
def main(cfg):

    wandb.init(
        project="clip2latent",
        config=OmegaConf.to_container(cfg),
        entity="justinpinkney",
    )
    # Load model
    device = cfg["device"]

    prior_network = WPlusPriorNetwork(n_latents=3, **cfg["model"]["network"]).to(device)
    diffusion_prior = LatentPrior(prior_network, **cfg["model"]["diffusion"]).to(device)
    diffusion_prior.cfg = cfg


    # Load eval models
    with dnnlib.util.open_url(cfg["data"]["sg_pkl"]) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
    clip_model, _ = clip.load(cfg["data"]["clip_variant"], device=device)

    val_data = {
        "val_im": make_image_val_data(G, clip_model, cfg["val"]["n_im_val_samples"], device),
        "val_text": make_text_val_data(G, clip_model, cfg["val"]["text_val_samples"], device),
    }


    stats, loader = load_data(cfg.data)

    trainer = DiffusionPriorTrainer(
        diffusion_prior=diffusion_prior,
        lr=cfg["train"]["opt"]["lr"],
        wd = cfg["train"]["opt"]["weight_decay"],
    ).to(device)
    
    checkpoint_dir = f"checkpoints/{datetime.now():%Y%m%d-%H%M%S}"
    checkpointer = Checkpointer(checkpoint_dir, cfg["train"]["loop"]["val_it"])

    train(diffusion_prior, trainer, loader, device, stats, G, clip_model, val_data, **cfg["train"]["loop"], save_checkpoint=checkpointer.save_checkpoint)

import webdataset as wds
from functools import partial

import numpy as np

def identity(x):
    return x

def add_noise(x, scale=0.75):
    orig_norm = x.norm(dim=-1, keepdim=True)
    x = x/orig_norm
    noise = torch.randn_like(x)
    noise /= noise.norm(dim=-1, keepdim=True)
    x += scale*noise
    x /= x.norm(dim=-1, keepdim=True)*orig_norm
    return x

def load_data(cfg):
    if cfg.format != "webdataset":
        raise NotImplementedError()

    n_stats = 10_000
    try:
        data_path = hydra.utils.to_absolute_path(cfg.path)
    except TypeError:
        data_path = [hydra.utils.to_absolute_path(x) for x in cfg.path]
    stats_ds = wds.WebDataset(data_path).decode().to_tuple('img_feat.npy', 'latent.npy').shuffle(5000).batched(n_stats)
    stats_data = next(stats_ds.__iter__())
    
    stats = {
        "clip_features": train_utils.make_data_stats(torch.tensor(stats_data[0])),
        "w": train_utils.make_data_stats(torch.tensor(stats_data[1])),
    }

    ds = (
        wds.WebDataset(data_path)
            .shuffle(5000)
            .decode()
            .to_tuple('img_feat.npy', 'latent.npy')
            .batched(cfg.bs)
            .map_tuple(torch.tensor, torch.tensor)
        )
    if cfg.noise_scale > 0:
        ds = ds.map_tuple(partial(add_noise, scale=cfg.noise_scale), identity)
    ds = ds.map_tuple(identity, partial(train_utils.normalise_data, w_mean=stats["w"][0], w_std=stats["w"][1]))

    # Doesn't seem to work well if we norm z
    loader = wds.WebLoader(ds, num_workers=16, batch_size=None)
    return stats,loader

if __name__ == "__main__":
    main()
