# Generate datasets
from multiprocessing import Process
import multiprocessing as mp
import math
from functools import partial
from pathlib import Path
from typing import Any

import clip
import numpy as np
import torch
import typer
from joblib import Parallel, delayed
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import torch.nn.functional as F

from clip2latent.models import load_sg


generators = {
    "sg2-ffhq-1024": partial(load_sg, 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-1024x1024.pkl'),
    "sg3-lhq-256": partial(load_sg, 'data/models/lhq-256-stylegan3-t-25Mimg.pkl'),
}

def mix_styles(w_batch, space):
    """Defines a style mixing procedure"""
    space_spec = {
        "w3": (4, 4, 10),
    }
    latent_mix = space_spec[space]

    bs = w_batch.shape[0]
    spec = torch.tensor(latent_mix).to(w_batch.device)

    index = torch.randint(0,bs, (len(spec),bs)).to(w_batch.device)
    return w_batch[index, 0, :].permute(1,0,2).repeat_interleave(spec, dim=1), spec

@torch.no_grad()
def run_folder_list(
    device_index,
    out_dir,
    generator_name,
    feature_extractor_name,
    out_image_size,
    batch_size,
    n_save_workers,
    samples_per_folder,
    folder_indexes,
    space="w",
    save_im=True,
    ):
    """Generate a directory of generated images and correspdonding embeddings and latents"""
    latent_dim = 512
    device = f"cuda:{device_index}"
    typer.echo(device_index)

    typer.echo("Loading generator")
    G = generators[generator_name]().to(device).eval()

    typer.echo("Loading feature extractor")
    feature_extractor, _ = clip.load(feature_extractor_name, device=device)
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    
    typer.echo("Generating samples")
    typer.echo(f"using space {space}")

    with Parallel(n_jobs=n_save_workers, prefer="threads") as parallel:
        for i_folder in folder_indexes:
            folder_name = out_dir/f"{i_folder:05d}"
            folder_name.mkdir(exist_ok=True)

            z = torch.randn(samples_per_folder, latent_dim, device=device)
            w = G.mapping(z, c=None)
            ds = torch.utils.data.TensorDataset(w)
            loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)
            for batch_idx, batch in enumerate(tqdm(loader, position=device_index)):
                if space == "w":
                    this_w = batch[0].to(device)
                    latents = this_w[:,0,:].cpu().numpy()
                else:
                    this_w, select_idxs = mix_styles(batch[0].to(device), space)
                    latents = this_w[:,select_idxs,:].cpu().numpy()

                out = G.synthesis(this_w)

                out = F.interpolate(out, (out_image_size,out_image_size), mode="area")
                clip_in = 0.5*F.interpolate(out, (224,224), mode="area") + 0.5
                image_features = feature_extractor.encode_image(normalize(clip_in))


                out = out.permute(0,2,3,1).clamp(-1,1)
                out = (255*(out*0.5 + 0.5).cpu().numpy()).astype(np.uint8)
                image_features = image_features.cpu().numpy()
                parallel(
                    delayed(process_and_save)(batch_size, folder_name, batch_idx, idx, latent, im, image_feature, save_im) 
                    for idx, (latent, im, image_feature) in enumerate(zip(latents, out, image_features))
                    ) 

    typer.echo("finished folder")


def process_and_save(batch_size, folder_name, batch_idx, idx, latent, im, image_feature, save_im):
    im = Image.fromarray(im)
    count = batch_idx*batch_size + idx
    basename = folder_name/f"{folder_name.stem}{count:04}"
    np.save(basename.with_suffix(".latent.npy"), latent)
    np.save(basename.with_suffix(".img_feat.npy"), image_feature)
    if save_im:
        im.save(basename.with_suffix(".gen.jpg"), quality=95)


def main(
    out_dir:Path,
    n_samples:int=1_000_000,
    generator_name:str="sg2-ffhq-1024", # Key into `generators` dict`
    feature_extractor_name:str="ViT-B/32",
    n_gpus:int=2,
    out_image_size:int=256,
    batch_size:int=32,
    n_save_workers:int=16,
    space:str="w",
    samples_per_folder:int=10_000,
    save_im:bool=False, # Save the generated images?
    ):
    typer.echo("starting")

    out_dir.mkdir(parents=True)

    n_folders = math.ceil(n_samples/samples_per_folder)
    folder_indexes = range(n_folders)

    sub_indexes = np.array_split(folder_indexes, n_gpus)

    processes = []
    for dev_idx, folder_list in enumerate(sub_indexes):
        p = Process(
            target=run_folder_list, 
            args=(
                dev_idx,
                out_dir,
                generator_name,
                feature_extractor_name,
                out_image_size,
                batch_size,
                n_save_workers,
                samples_per_folder,
                folder_list,
                space,
                save_im,
                ),
            )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    typer.echo("finished all")

if __name__ == "__main__":
    mp.set_start_method('spawn')
    typer.run(main)
