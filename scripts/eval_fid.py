from textwrap import wrap
from typing import List
from omegaconf import OmegaConf
import torch
from pathlib import Path
import numpy as np
from PIL import Image
import train
from joblib import Parallel, delayed
from tqdm import tqdm
from datetime import datetime
import typer



class FidWrapper():
    def __init__(self, samples, model, cond_scale=1.0) -> None:
        self.samples = samples
        self.count = 0
        self.model = model
        self.cond_scale = cond_scale

    def forward(self, z):
        bs = z.shape[0]
        inp = self.samples[self.count:(self.count+bs)]
        self.count += bs
        images, _ = self.model(inp, cond_scale=self.cond_scale)
        images = (255*(images.clamp(-1,1)*0.5 + 0.5)).to(torch.uint8)

        # im = Image.fromarray(images[0].detach().cpu().permute(1,2,0).numpy()).save("temp.jpg")
        return images

    def __len__(self):
        return len(self.samples)


def main(
    skips:List[int]=[1,100,250],
    cond_scales:List[float]=[1,1.05,1.1,1.2,1.3,1.5,1.75,2,2.5,3,4,5,10],
    write_results:bool = True,
    checkpoint:str = "best.ckpt",
    cfg_file:str= "best.yaml",
    device:str= "cuda:0",
    n_samples:int= 16,
    truncation:float=1.0,
    ):

    typer.echo(f"Running skips: {skips}")
    typer.echo(f"Running cond scales: {cond_scales}")

    with open("celeba-samples.txt", 'rt') as f:
        text_samples = f.read().splitlines()

    model = Clip2StyleGAN(cfg_file, device, checkpoint, skips=100)
    model.to(device)
    model.eval()


    wrapper = FidWrapper(text_samples, model, cond_scale=1.0)
    print(len(wrapper))

    from cleanfid import fid
    # function that accepts a latent and returns an image in range[0,255]
    gen = lambda z: wrapper.forward(z)
    score = fid.compute_fid(fdir2="/mnt/data_rome/laion/tedigan-data/CelebAMask-HQ/CelebA-HQ-img", gen=gen,
           batch_size=40, num_gen=len(wrapper))

    print(score)


if __name__ == "__main__":
    typer.run(main)