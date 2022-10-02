import math
from clip2latent.models import Clip2StyleGAN
from PIL import Image
import torch
from datetime import datetime
import numpy as np

device = "cuda:0"
skips = 250
model = Clip2StyleGAN("best.yaml", device=device, checkpoint="best.ckpt")
inp = ["a photo",]
warmup = 5
measure = 10

args = {
    "skips": skips,
    "n_samples_per_txt": 16,
    "clip_sort": True,
    "cond_scale": 2.0,
    "show_progress": False,
}

for i in range(warmup):
    out = model(inp, **args)
    torch.cuda.synchronize()

times = []
for i in range(measure):
    start = datetime.now()
    out = model(inp,  **args)
    torch.cuda.synchronize()
    taken = datetime.now() - start
    times.append(taken)
    print(taken)

print("-------------")
print(np.mean(times))