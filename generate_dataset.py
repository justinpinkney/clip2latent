# Generate datasets

from concurrent.futures import process
from functools import partial
from pathlib import Path
from typing import Any
import math

# Simple
# Run a generator model recording generated images and latents
#   Generate latents (either W or W+ for stylegan)
# Run a feature encoding model (e.g. clip) on the generated images, store the features

# Complex
# Encode real data - store original, result, and latent
# Process with feature model
# [Optionally] process labels with text model
import typer
import torch
import clip
from tqdm import tqdm
from joblib import Parallel, delayed

def load_sg(network_pkl):
    import sys
    sys.path.append("stylegan3")
    import dnnlib
    import legacy

    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'] # type: ignore
    return G

generators = {
    "sg2-ffhq-1024": partial(load_sg, 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-1024x1024.pkl'),
}

@torch.no_grad()
def main(
    out_dir:Path,
    n_samples:int=1_000_000,
    generator_name:str="sg2-ffhq-1024", # TODO accept other models
    feature_extractor_name:str="ViT-B/32", # TODO accept other models
    n_gpus:int=1,
    out_image_size:int=256, # TODO what about non-square?
    feat_image_size:int=224,
    feat_im_preprocess:str="resize", # TODO others e.g. nxrandcrop and average
    batch_size:int=32, # 64 struggles?
    n_save_workers:int=16,
    ):
    typer.echo("starting")

    out_dir.mkdir(parents=True)

    samples_per_folder = 10_000
    n_folders = math.ceil(n_samples/samples_per_folder)
    device = "cuda:0"

    typer.echo("Loading generator")
    G = generators[generator_name]().to(device).eval()
    typer.echo("Loading feature extractor")
    feature_extractor, preprocess = clip.load(feature_extractor_name, device=device)

    typer.echo("Generating samples")
    with Parallel(n_jobs=n_save_workers) as parallel:
        for i_folder in range(n_folders):
            folder_name = out_dir/f"{i_folder:05d}"
            folder_name.mkdir(exist_ok=True)

            z = torch.randn(samples_per_folder, 512, device=device)
            w = G.mapping(z, c=None)
            ds = torch.utils.data.TensorDataset(w)
            loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)
            for batch_idx, batch in enumerate(tqdm(loader)):
                out = G.synthesis(batch[0].to(device))
                clip_in = torch.nn.functional.interpolate(out, (224,224), mode="area")
                image_features = feature_extractor.encode_image(clip_in)


                out = torch.nn.functional.interpolate(out, (out_image_size,out_image_size), mode="area").cpu()
                out = out.permute(0,2,3,1).clamp(-1,1)
                out = (255*(out*0.5 + 0.5).numpy()).astype(np.uint8)
                image_features = image_features.cpu().numpy()
                latents = batch[0][:,0,:].cpu().numpy()
                parallel(
                    delayed(process_and_save)(batch_size, folder_name, batch_idx, idx, latent, im, image_feature) 
                    for idx, (latent, im, image_feature) in enumerate(zip(latents, out, image_features))
                    ) 
        typer.echo("finished folder")
    typer.echo("finished all")

def process_and_save(batch_size, folder_name, batch_idx, idx, latent, im, image_feature):
    im = Image.fromarray(im)
    count = batch_idx*batch_size + idx
    save(folder_name, count, latent, im, image_feature)

import numpy as np
def save(folder_name, idx, latent, im, image_feature):
    basename = folder_name/f"{folder_name.stem}{idx:04}"
    np.save(basename.with_suffix(".latent.npy"), latent)
    im.save(basename.with_suffix(".gen.jpg"), quality=95)
    np.save(basename.with_suffix(".img_feat.npy"), image_feature)



from PIL import Image
def tensor2pil(t):
    # Expect CHW
    
    im = Image.fromarray(im.to(torch.uint8).numpy())
    return im


if __name__ == "__main__":
    typer.run(main)