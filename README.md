# clip2latent - Official PyTorch Code

[![Open Arxiv](https://img.shields.io/badge/arXiv-tbc.tbc-b31b1b.svg)](TODO)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/justinpinkney/clip2latent/blob/main/demo.ipynb)
[![Open Demo](https://img.shields.io/badge/Open%20Demo-8136E2?logo=data:image/png;base64,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)](https://tinyurl.com/clip2latent)

![](images/headline-large.jpeg)

> ## _clip2latent: Text driven sampling of a pre-trained StyleGAN using denoising diffusion and CLIP_
>
> Justin N. M. Pinkney and Chuan Li
> @ Lambda Inc.
>
> We introduce a new method to efficiently create text-to-image models from a pre-trained CLIP and StyleGAN. It enables text driven sampling with an existing generative model without any external data or fine-tuning. This is achieved by training a diffusion model conditioned on CLIP embeddings to sample latent vectors of a pre-trained StyleGAN, which we call \textit{clip2latent}. We leverage the alignment between CLIP’s image and text embeddings to avoid the need for any text labelled data for training the conditional diffusion model. We demonstrate that clip2latent allows us to generate high-resolution (1024x1024 pixels) images based on text prompts with fast sampling, high image quality, and low training compute and data requirements. We also show that the use of the well studied StyleGAN architecture, without further fine-tuning, allows us to directly apply existing methods to control and modify the generated images adding a further layer of control to our text-to-image pipeline.


## Installation

```bash
git clone https://github.com/justinpinkney/clip2latent.git
cd clip2latent
python -m venv .venv --prompt clip2latent
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

### Inference

To run the models for inference the simplest way is to start the gradio demo:

```bash
python scripts/demo.py
```

This will fetch the required models from huggingface hub and start gradio demo which can be accessed via a web browser.

To run a model via python:

```python
from clip2latent import models

prompt = "a hairy man"
device = "cuda:0"
cfg_file = "https://huggingface.co/lambdalabs/clip2latent/resolve/main/ffhq-sg2-510.yaml"
checkpoint =  "https://huggingface.co/lambdalabs/clip2latent/resolve/main/ffhq-sg2-510.ckpt"

model = models.Clip2StyleGAN(cfg_file, device, checkpoint)
images, clip_score = model(prompt)
# images are tensors of shape: bchw, range: -1..1
```

Or take a look at the example notebook `demo.ipynb`.

### Training

#### Generate data

To train a model of your own first you need to generate some data. We provide a command line interface which will run a StyleGAN model and pass the generated images to CLIP. The W latent vector and the CLIP image embedding will be stored as npy files, packed into tar files ready for use as a webdataset. To generate data used to traing the ffhq model in the paper do:

```bash
python scripts/generate_dataset.py
```

For more details of dataset generation options see the help for `generate_dataset.py`:

```
Usage: generate_dataset.py [OPTIONS] OUT_DIR

Arguments:
  OUT_DIR  Location to save dataset [required]

Options:
  --n-samples INTEGER             Number of samples to generate [default: 1000000]
  --generator-name TEXT           Name of predefined generator loader [default: sg2-ffhq-1024]
  --feature-extractor-name TEXT   CLIP model to use for image embedding [default: ViT-B/32]
  --n-gpus INTEGER                Number of GPUs to use [default: 2]
  --out-image-size INTEGER        If saving generated images, resize to this dimension [default: 256]
  --batch-size INTEGER            Batch size [default: 32]
  --n-save-workers INTEGER        Number of workers to use while saving [default: 16]
  --space TEXT                    Latent space to use [default: w]
  --samples-per-folder INTEGER    Number of samples per tar file [default: 10000]
  --save-im / --no-save-im        Save images? [default: no-save-im]
```

To use a different StyleGAN generator, add the required loading function to the `generators` dict in `generate_dataset.py`, then use that key as the `generator_name`. To use non-StyleGAN generators should be possible but would require additional modifications.

#### Train

To manage configuration for the model and training parameters we use [hydra](https://hydra.cc/), to train with default configuration simply run:

```bash
python scripts/train.py
```

This will run the model with the default configuration as follows:

```yaml
model:
  network:
    dim: 512
    num_timesteps: 1000
    depth: 12
    dim_head: 64
    heads: 12
  diffusion:
    image_embed_dim: 512
    timesteps: 1000
    cond_drop_prob: 0.2
    image_embed_scale: 1.0
    text_embed_scale: 1.0
    beta_schedule: cosine
    predict_x_start: true
data:
  bs: 512
  format: webdataset
  path: data/webdataset/sg2-ffhq-1024-clip/{00000..99}.tar
  embed_noise_scale: 1.0
  sg_pkl: https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-1024x1024.pkl
  clip_variant: ViT-B/32
  n_latents: 1
  latent_dim: 512
  latent_repeats:
  - 18
  val_im_samples: 64
  val_text_samples: text/face-val.txt
  val_samples_per_text: 4
logging: wandb
wandb_project: clip2latent
wandb_entity: null
name: null
device: cuda:0
resume: null
train:
  znorm_embed: false
  znorm_latent: true
  max_it: 1000000
  val_it: 10000
  lr: 0.0001
  weight_decay: 0.01
  ema_update_every: 10
  ema_beta: 0.9999
  ema_power: 0.75
```

To train with a different configuration you can either change individual parameters using the following command line override syntax:

```bash
python scripts/train.py data.bs=128
```

which would set the batch size to 128.

Alternatively you can create your own yaml configuration files and switch between them, e.g. we also provide an example 'small' model configuration at `config/model/small.yaml`, to train using this simply run

```bash
python scripts/train.py model=small
```

For more details please refer to the [hydra documentation](https://hydra.cc/docs/intro/).

Training is set up to run on a single GPU and does not currently support multigpu training. The default settings will take around 18 hours to train on a single A100-80GB, although the best checkpoint is likely to occur within 10 hours of training.

## Citation

TODO