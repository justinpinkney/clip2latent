import gradio as gr
import torch
from clip2latent import models

from clip2latent import train_utils

model_name = "landscape"
device = "cuda:0"
model_choices = {
    "faces": {
        "checkpoint": "best.ckpt",
        "config": "best.yaml",
        "output_size": 1024,
        },
    "landscape": {
        "checkpoint": "checkpoints/lhq-410-best.ckpt",
        "config": "checkpoints/lhq-410-best.yaml",
        "output_size": 256,
    }
}
checkpoint = model_choices[model_name]["checkpoint"]
cfg_file = model_choices[model_name]["config"]
output_size = model_choices[model_name]["output_size"]
model = models.Clip2StyleGAN(cfg_file, device, checkpoint)

n_samples = 9
skips = {
    "Slow": 1,
    "Med": 10,
    "Fast": 100,
    "Super": 250,
}

@torch.no_grad()
def run_model(text_samples, cond_scale, skip_name, select_best):
    n_skips = skips[skip_name]
    images, _ = model(text_samples, n_samples_per_txt=n_samples, cond_scale=cond_scale, skips=n_skips, clip_sort=True)

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

output_image = gr.Image(shape=(output_size,output_size), label="Generated images")

demo = gr.Interface(fn=run_model, inputs=[text_input, cond_input, skip_timesteps, select_best], outputs=output_image)

demo.launch(enable_queue=False)

