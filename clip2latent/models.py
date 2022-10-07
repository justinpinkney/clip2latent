import clip
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn.functional as F
from dalle2_pytorch import DiffusionPriorNetwork
from dalle2_pytorch.train import DiffusionPriorTrainer
from torchvision import transforms
from io import BytesIO
import requests
from pathlib import Path

from clip2latent.latent_prior import LatentPrior, WPlusPriorNetwork


def load_sg(network_pkl):
    import sys
    code_folder = Path(__file__).parent
    sg3_path = str(code_folder/"stylegan3")       
    sys.path.append(sg3_path)
    import dnnlib
    import legacy

    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'] # type: ignore
    return G

def is_url(path):
    if isinstance(path, str) and path.startswith("http"):
        return True
    else:
        return False

def load_remote_cfg(cfg):
    r = requests.get(cfg)
    r.raise_for_status()
    f = BytesIO(r.content)
    return OmegaConf.load(f)

class Clipper(torch.nn.Module):
    def __init__(self, clip_variant):
        super().__init__()
        clip_model, _ = clip.load(clip_variant, device="cpu")
        self.clip = clip_model
        self.normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        self.clip_size = (224,224)

    def embed_image(self, image):
        """Expects images in -1 to 1 range"""
        clip_in = F.interpolate(image, self.clip_size, mode="area")
        clip_in = self.normalize(0.5*clip_in + 0.5).clamp(0,1)
        return self.clip.encode_image(self.normalize(clip_in))

    def embed_text(self, text_samples):
        text = clip.tokenize(text_samples).to(self._get_device())
        return self.clip.encode_text(text)

    def _get_device(self):
        for p in self.clip.parameters():
            return p.device

class Clip2StyleGAN(torch.nn.Module):
    """A wrapper around the compontent models to create an end-to-end text2image model"""
    def __init__(self, cfg, device, checkpoint=None) -> None:
        super().__init__()

        if not isinstance(cfg, DictConfig):
            if is_url(cfg):
                cfg = load_remote_cfg(cfg)
            else:
                cfg = OmegaConf.load(cfg)

        G, clip_model, trainer = load_models(cfg, device)
        if checkpoint is not None:
            if is_url(checkpoint):
                state_dict = torch.hub.load_state_dict_from_url(checkpoint, map_location="cpu")
            else:
                state_dict = torch.load(checkpoint, map_location="cpu")
            trainer.load_state_dict(state_dict["state_dict"], strict=False)
        diffusion_prior = trainer.ema_diffusion_prior.ema_model
        self.G = G
        self.clip_model = clip_model
        self.diffusion_prior = diffusion_prior

    def forward(self, text_samples, n_samples_per_txt=1, cond_scale=1.0, truncation=1.0, skips=1, clip_sort=False, edit=None, show_progress=True):
        self.diffusion_prior.set_timestep_skip(skips)
        text_features = self.clip_model.embed_text(text_samples)
        if n_samples_per_txt > 1:
            text_features = text_features.repeat_interleave(n_samples_per_txt, dim=0)
        pred_w = self.diffusion_prior.sample(text_features, cond_scale=cond_scale, show_progress=show_progress, truncation=truncation)

        if edit is not None:
            pred_w = pred_w + edit.to(pred_w.device)
        images = self.G.synthesis(pred_w)

        pred_w_clip_features = self.clip_model.embed_image(images)
        similarity = torch.cosine_similarity(pred_w_clip_features, text_features)
        if clip_sort:
            similarity, idxs = torch.sort(similarity, descending=True)
            images = images[idxs, ...]

        return images, similarity

def load_models(cfg, device, stats=None):
    """Load the diffusion trainer and eval models based on a config

    If the model requires statistics for embed or latent normalisation
    then these should be passed into this function, unless the state of
    the model is to be loaded from a state_dict (which will contain these)
    statistics, in which case the stats will be filled with dummy values.
    """
    if cfg.data.n_latents > 1:
        prior_network = WPlusPriorNetwork(n_latents=cfg.data.n_latents, **cfg.model.network).to(device)
    else:
        prior_network = DiffusionPriorNetwork(**cfg.model.network).to(device)

    embed_stats = latent_stats = (None, None)
    if stats is None:
        # Make dummy stats assuming they will be loaded from the state dict
        clip_dummy_stat = torch.zeros(cfg.model.network.dim,1)
        w_dummy_stat = torch.zeros(cfg.model.network.dim)
        if cfg.data.n_latents > 1:
            w_dummy_stat = w_dummy_stat.unsqueeze(0).tile(1, cfg.data.n_latents)
        stats = {"clip_features": (clip_dummy_stat, clip_dummy_stat), "w": (w_dummy_stat, w_dummy_stat)}

    if cfg.train.znorm_embed:
        embed_stats = stats["clip_features"]
    if cfg.train.znorm_latent:
        latent_stats = stats["w"]

    diffusion_prior = LatentPrior(
        prior_network,
        num_latents=cfg.data.n_latents,
        latent_repeats=cfg.data.latent_repeats,
        latent_stats=latent_stats,
        embed_stats=embed_stats,
        **cfg.model.diffusion).to(device)
    diffusion_prior.cfg = cfg

    # Load eval models
    G = load_sg(cfg.data.sg_pkl).to(device)
    clip_model = Clipper(cfg.data.clip_variant).to(device)

    trainer = DiffusionPriorTrainer(
        diffusion_prior=diffusion_prior,
        lr=cfg.train.lr,
        wd=cfg.train.weight_decay,
        ema_beta=cfg.train.ema_beta,
        ema_update_every=cfg.train.ema_update_every,
    ).to(device)

    return G, clip_model, trainer
