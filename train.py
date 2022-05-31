from functools import partial
import logging
import sys
from datetime import datetime
from pathlib import Path

import clip
import hydra
import torch
from dalle2_pytorch import DiffusionPriorNetwork
from dalle2_pytorch.dalle2_pytorch import exists, default
from dalle2_pytorch.train import DiffusionPriorTrainer
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm
import numpy as np

import wandb

sys.path.append("stylegan3")
import dnnlib
import legacy
import torch
from torchvision import transforms
import torch.nn.functional as F

from clip2latent import train_utils
from clip2latent.latent_prior import LatentPrior, WPlusPriorNetwork
from clip2latent.train_utils import (compute_val, make_grid,
                                     make_image_val_data, make_text_val_data)
from clip2latent.data import load_data

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


class Clipper(torch.nn.Module):
    def __init__(self, clip_variant):
        super().__init__()
        clip_model, _ = clip.load(clip_variant, device="cpu")
        self.clip = clip_model
        self.normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        self.clip_size = (224,224)

    def embed_image(self, image):
        """Expects images in -1 to 1 range"""
        clip_in = F.interpolate(image, self.clip_size, mode="area")
        clip_in = self.normalize(0.5*clip_in + 0.5).clamp(0,1)
        return self.clip.encode_image(self.normalize(clip_in))

    def embed_text(self, text_samples):
        text = clip.tokenize(text_samples).to(self._get_device())
        return self.clip.encode_text(text)

    def _get_device(self):
        for p in self.clip.parameters():
            return p.device

def validation(current_it, device, diffusion_prior, G, clip_model, val_data, samples_per_text):
    single_im = {"clip_features": val_data["val_im"]["clip_features"][0].unsqueeze(0)}
    captions = val_data["val_caption"]

    for input_data, key, cond_scale, repeats in zip(
        [val_data["val_im"], single_im, val_data["val_text"], val_data["val_text"]],
        ["image-similarity", "image-vars", "text2im", "text2im-super2"],
        [1.0, 1.0, 1.0, 2.0],
        [1, 8, samples_per_text, samples_per_text],
    ):
        tiled_data = input_data["clip_features"].repeat_interleave(repeats, dim=0)
        cos_sim, ims = compute_val(diffusion_prior, tiled_data, G, clip_model, device, cond_scale=cond_scale)
        wandb.log({f'val/{key}':cos_sim.mean()}, step=current_it)


        if key.startswith("text"):
            num_chunks = int(np.ceil(ims.shape[0]//repeats))
            for idx, (sim, im_chunk) in enumerate(zip(
                cos_sim.chunk(num_chunks), 
                ims.chunk(num_chunks)
                )):
                
                caption = captions[idx]
                im = wandb.Image(make_grid(im_chunk), caption=f'{sim.mean():.2f} - {caption}')
                wandb.log({f'val/image/{key}/{idx}': im}, step=current_it)
        else:
            for idx, im in enumerate(ims.chunk(int(np.ceil(ims.shape[0]/16)))):
                wandb.log({f'val/image/{key}/{idx}': wandb.Image(make_grid(im))}, step=current_it)

    logger.info("Validation done.")

def train_step(diffusion_prior, device, batch):
    diffusion_prior.train()
    batch_z, batch_w = batch
    batch_z = batch_z.to(device)
    batch_w = batch_w.to(device)

    loss = diffusion_prior(batch_z, batch_w)
    loss.backward()
    return loss

    
def train(trainer, loader, device, val_it, validate, save_checkpoint, max_it, print_it=50):
    
    current_it = 0
    current_epoch = 0

    while current_it < max_it:
    
        wandb.log({'epoch': current_epoch}, step=current_it)
        pbar = tqdm(loader)
        for batch in pbar:
            if current_it % val_it == 0:
                validate(current_it, device, trainer)

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
        project=cfg.wandb_project,
        config=OmegaConf.to_container(cfg),
        entity=cfg.wandb_entity,
        name=cfg.name,
    )
    # Load model
    device = cfg.device

    stats, loader = load_data(cfg.data)

    if cfg.data.n_latents > 1:
        prior_network = WPlusPriorNetwork(n_latents=cfg.data.n_latents, **cfg.model.network).to(device)
    else:
        prior_network = DiffusionPriorNetwork(**cfg.model.network).to(device)

    embed_stats = latent_stats = (None, None)
    if cfg.train.znorm_embed:
        embed_stats = stats["clip_features"]
    if cfg.train.znorm_latent:
        latent_stats = stats["w"]
    
    diffusion_prior = LatentPrior(
        prior_network,
        num_latents=cfg.data.n_latents,
        latent_repeats=cfg.data.latent_repeats,
        latent_stats=latent_stats,
        embed_stats=embed_stats,
        **cfg.model.diffusion).to(device)
    diffusion_prior.cfg = cfg

    # Load eval models
    with dnnlib.util.open_url(cfg.data.sg_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
    clip_model = Clipper(cfg.data.clip_variant).to(device)

    text_embed, text_samples = make_text_val_data(G, clip_model, hydra.utils.to_absolute_path(cfg.data.val_text_samples))
    val_data = {
        "val_im": make_image_val_data(G, clip_model, cfg.data.val_im_samples, device),
        "val_text": text_embed,
        "val_caption": text_samples,
    }

    trainer = DiffusionPriorTrainer(
        diffusion_prior=diffusion_prior,
        lr=cfg.train.lr,
        wd=cfg.train.weight_decay,
        ema_beta=cfg.train.ema_beta,
        ema_update_every=cfg.train.ema_update_every,
    ).to(device)
    
    checkpoint_dir = f"checkpoints/{datetime.now():%Y%m%d-%H%M%S}"
    checkpointer = Checkpointer(checkpoint_dir, cfg.train.val_it)
    validate = partial(validation,
        G=G,
        clip_model=clip_model,
        val_data=val_data,
        samples_per_text=cfg.data.val_samples_per_text,
        )

    train(trainer, loader, device,
        val_it=cfg.train.val_it,
        max_it=cfg.train.max_it,
        validate=validate,
        save_checkpoint=checkpointer.save_checkpoint,
        )

if __name__ == "__main__":
    main()
