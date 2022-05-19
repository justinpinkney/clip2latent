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
def run_folder_list(device_index, out_dir, generator_name, feature_extractor_name, out_image_size, batch_size, n_save_workers, samples_per_folder, folder_indexes):
    device = f"cuda:{device_index}"
    typer.echo(device_index)
    typer.echo("Loading generator")
    G = generators[generator_name]().to(device).eval()
    typer.echo("Loading feature extractor")
    feature_extractor, preprocess = clip.load(feature_extractor_name, device=device)
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    
    typer.echo("Generating samples")
    with Parallel(n_jobs=n_save_workers, prefer="threads") as parallel:
        for i_folder in folder_indexes:
            folder_name = out_dir/f"{i_folder:05d}"
            folder_name.mkdir(exist_ok=True)

            z = torch.randn(samples_per_folder, 512, device=device)
            w = G.mapping(z, c=None)
            ds = torch.utils.data.TensorDataset(w)
            loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)
            for batch_idx, batch in enumerate(tqdm(loader, position=device_index)):
                out = G.synthesis(batch[0].to(device))

                out = torch.nn.functional.interpolate(out, (out_image_size,out_image_size), mode="area")
                clip_in = 0.5*torch.nn.functional.interpolate(out, (224,224), mode="area") + 0.5
                image_features = feature_extractor.encode_image(normalize(clip_in))


                out = out.permute(0,2,3,1).clamp(-1,1)
                out = (255*(out*0.5 + 0.5).cpu().numpy()).astype(np.uint8)
                image_features = image_features.cpu().numpy()
                latents = batch[0][:,0,:].cpu().numpy()
                parallel(
                    delayed(process_and_save)(batch_size, folder_name, batch_idx, idx, latent, im, image_feature) 
                    for idx, (latent, im, image_feature) in enumerate(zip(latents, out, image_features))
                    ) 
        typer.echo("finished folder")

def process_and_save(batch_size, folder_name, batch_idx, idx, latent, im, image_feature):
    im = Image.fromarray(im)
    count = batch_idx*batch_size + idx
    save(folder_name, count, latent, im, image_feature)


def save(folder_name, idx, latent, im, image_feature):
    basename = folder_name/f"{folder_name.stem}{idx:04}"
    np.save(basename.with_suffix(".latent.npy"), latent)
    im.save(basename.with_suffix(".gen.jpg"), quality=95)
    np.save(basename.with_suffix(".img_feat.npy"), image_feature)



def main(
    out_dir:Path,
    n_samples:int=1_000_000,
    generator_name:str="sg2-ffhq-1024", # TODO accept other models
    feature_extractor_name:str="ViT-B/32", # TODO accept other models
    n_gpus:int=2,
    out_image_size:int=256, # TODO what about non-square?
    feat_im_preprocess:str="resize", # TODO others e.g. nxrandcrop and average
    batch_size:int=32, # 64 struggles?
    n_save_workers:int=16,
    ):
    typer.echo("starting")

    out_dir.mkdir(parents=True)

    samples_per_folder = 10_000
    n_folders = math.ceil(n_samples/samples_per_folder)
    folder_indexes = range(n_folders)

    sub_indexes = np.array_split(folder_indexes, n_gpus)

    ps = []
    for dev_idx, folder_list in enumerate(sub_indexes):
        p = Process(
            target=run_folder_list, 
            args=(dev_idx, out_dir, generator_name, feature_extractor_name, out_image_size, batch_size, n_save_workers, samples_per_folder, folder_list),
            )
        p.start()
        ps.append(p)

    for p in ps:
        p.join()
    typer.echo("finished all")

if __name__ == "__main__":
    mp.set_start_method('spawn')
    typer.run(main)
