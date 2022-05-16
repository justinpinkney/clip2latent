from datetime import datetime
import logging
from pathlib import Path
import sys
import wandb
import yaml

import clip
import torch
from tqdm.auto import tqdm

from dalle2_pytorch import DiffusionPriorNetwork

sys.path.append("stylegan3")
import dnnlib
import legacy
import torch

from clip2latent import train_utils
from clip2latent.train_utils import compute_val, make_grid, make_image_val_data, make_text_val_data
from clip2latent.latent_prior import ZWPrior

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

    text_result, ims = compute_val(diffusion_prior, val_data["val_text"], G, clip_model, device, stats, cond_scale=1.5)
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

    
def train(diffusion_prior, opt, loader, device, stats, G, clip_model, val_data, val_it, print_it, save_checkpoint, max_it):
    
    current_it = 0
    current_epoch = 0

    while current_it < max_it:
    
        wandb.log({'epoch': current_epoch}, step=current_it)
        pbar = tqdm(loader)
        for batch in pbar:

            if current_it % val_it == 0:
                validation(current_it, device, diffusion_prior, stats, G, clip_model, val_data)

            loss = train_step(diffusion_prior, device, batch)

            if (current_it % print_it == 0):
                wandb.log({'loss': loss.item()}, step=current_it)
            
            opt.step()
            opt.zero_grad()
            current_it += 1
            pbar.set_postfix({"epoch": current_epoch, "it": current_it})

            save_checkpoint(diffusion_prior, current_it)

        current_epoch += 1


if __name__ == "__main__":

    with open("config.yaml", "rt") as f:
        cfg = yaml.safe_load(f)
        
    wandb.init(
        project="clip2latent",
        config=cfg,
        entity="justinpinkney",
    )
    # Load model
    device = cfg["device"]

    prior_network = DiffusionPriorNetwork(**cfg["model"]["network"]).to(device)
    diffusion_prior = ZWPrior(prior_network, **cfg["model"]["diffusion"]).to(device)
    diffusion_prior.cfg = cfg

    z = torch.load(cfg["data"]["clip_feature_path"])
    w = torch.load(cfg["data"]["latent_path"])

    stats = {
        "w": train_utils.make_data_stats(w),
        "clip_features": train_utils.make_data_stats(z),
    }

    w_norm = train_utils.normalise_data(w, *stats["w"])
    # Doesn't seem to work well if we norm z
    # z_norm = train_utils.normalise_data(z, *stats["clip_features"])

    # Load eval models
    with dnnlib.util.open_url(cfg["data"]["sg_pkl"]) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
    clip_model, _ = clip.load(cfg["data"]["clip_variant"], device=device)

    val_data = {
        "val_im": make_image_val_data(G, clip_model, cfg["val"]["n_im_val_samples"], device),
        "val_text": make_text_val_data(G, clip_model, cfg["val"]["text_val_samples"], device),
    }

    ds = torch.utils.data.TensorDataset(z, w_norm)
    loader = torch.utils.data.DataLoader(ds, batch_size=cfg["data"]["bs"], shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(prior_network.parameters(), **cfg["train"]["opt"])
    
    checkpoint_dir = f"checkpoints/{datetime.now():%Y%m%d-%H%M%S}"
    checkpointer = Checkpointer(checkpoint_dir, cfg["train"]["loop"]["val_it"])

    train(diffusion_prior, opt, loader, device, stats, G, clip_model, val_data, **cfg["train"]["loop"], save_checkpoint=checkpointer.save_checkpoint)
