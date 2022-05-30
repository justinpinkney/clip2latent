import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from PIL import Image
import torchvision

def make_data_stats(w):
    w_mean = w.mean(dim=0)
    w_std = w.std(dim=0)
    return w_mean, w_std

def normalise_data(w, w_mean, w_std):
    device = w.device
    w = w - w_mean.to(device)
    w = w / w_std.to(device)
    return w

def denormalise_data(w, w_mean, w_std):
    device = w.device
    w = w * w_std.to(device)
    w = w + w_mean.to(device)
    return w

def make_grid(ims, pil=True, resize=True):
    if resize:
        ims = F.interpolate(ims, size=(256,256))
    grid = torchvision.utils.make_grid(
        ims.clamp(-1,1), 
        normalize=True,
        value_range=(-1,1),
        nrow=4,
        )
    if pil:
        grid = Image.fromarray((255*grid).to(torch.uint8).permute(1,2,0).detach().cpu().numpy())
    return grid
    
@torch.no_grad()
def make_image_val_data(G, clip_model, n_im_val_samples, device, latent_dim=512):
    clip_features = []

    zs = torch.randn((n_im_val_samples, latent_dim), device=device)
    ws = G.mapping(zs, c=None)
    for w in tqdm(ws):
        out = G.synthesis(w.unsqueeze(0))
        image_features = clip_model.embed_image(out)
        clip_features.append(image_features)

    clip_features = torch.cat(clip_features, dim=0)
    val_data = {
        "clip_features": clip_features,
        "z": zs,
        "w": ws,
    }
    return val_data


@torch.no_grad()
def make_text_val_data(G, clip_model, text_samples_file):
    """Load text samples from file"""
    with open(text_samples_file, 'rt') as f:
        text_samples = f.read().splitlines()
    text_features = clip_model.embed_text(text_samples)
    val_data = {"clip_features": text_features,}
    return val_data

@torch.no_grad()
def compute_val(diffusion, input_embed, G, clip_model, device, cond_scale=1.0, bs=8):

    diffusion.eval()
    images = []
    inp = input_embed.to(device)
    out = diffusion.sample(inp, cond_scale=cond_scale)

    pred_w_clip_features = []
    # batch in 1s to not worry about memory
    for w in out.chunk(bs):
        out = G.synthesis(w)
        images.append(out)
        image_features = clip_model.embed_image(out)
        pred_w_clip_features.append(image_features)

    pred_w_clip_features = torch.cat(pred_w_clip_features, dim=0)
    images = torch.cat(images, dim=0)

    y = input_embed/input_embed.norm(dim=1, keepdim=True)
    y_hat = pred_w_clip_features/pred_w_clip_features.norm(dim=1, keepdim=True)
    return torch.cosine_similarity(y, y_hat), images
    