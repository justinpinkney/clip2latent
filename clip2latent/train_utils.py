import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import clip
from PIL import Image
import torchvision
from torchvision import transforms

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

def make_grid(ims, pil=True):
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
def make_image_val_data(G, clip_model, n_im_val_samples, device):
    clip_features = []

    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    zs = torch.randn((n_im_val_samples, 512), device=device)
    ws = G.mapping(zs, c=None)
    for w in tqdm(ws):
        out = G.synthesis(w.unsqueeze(0))
        clip_in = 0.5*F.interpolate(out, (224,224)) + 0.5
        image_features = clip_model.encode_image(normalize(clip_in).clamp(0,1))
        clip_features.append(image_features)

    clip_features = torch.cat(clip_features, dim=0)
    val_data = {
        "clip_features": clip_features,
        "z": zs,
        "w": ws,
    }
    return val_data


@torch.no_grad()
def make_text_val_data(G, clip_model, text_samples, device):

    text = clip.tokenize(text_samples).to(device)
    text_features = clip_model.encode_text(text)
    val_data = {"clip_features": text_features,}
    return val_data

@torch.no_grad()
def compute_val(diffusion, val_im, G, clip_model, device, stats, cond_scale=1):

    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    diffusion.eval()
    images = []
    inp = val_im["clip_features"].to(device)
    out = diffusion.p_sample_loop(inp.shape, {"text_embed": inp}, cond_scale=cond_scale)

    pred_w_clip_features = []
    pred_w = denormalise_data(out, *stats["w"])
    for w in tqdm(pred_w):
        out = G.synthesis(w.tile(1,16,1)) # TODO make configurable
        images.append(out)
        clip_in = 0.5*F.interpolate(out, (224,224), mode="area") + 0.5
        image_features = clip_model.encode_image(normalize(clip_in).clamp(0,1))
        pred_w_clip_features.append(image_features)

    pred_w_clip_features = torch.cat(pred_w_clip_features, dim=0)
    images = torch.cat(images, dim=0)

    y = val_im["clip_features"]/val_im["clip_features"].norm(dim=1, keepdim=True)
    y_hat = pred_w_clip_features/pred_w_clip_features.norm(dim=1, keepdim=True)
    return torch.cosine_similarity(y, y_hat), images
    