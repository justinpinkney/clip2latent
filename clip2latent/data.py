from functools import partial
import webdataset as wds
import torch
import hydra

from clip2latent import train_utils

def identity(x):
    return x

def add_noise(x, scale=0.75):
    orig_norm = x.norm(dim=-1, keepdim=True)
    x = x/orig_norm
    noise = torch.randn_like(x)
    noise /= noise.norm(dim=-1, keepdim=True)
    x += scale*noise
    x /= x.norm(dim=-1, keepdim=True)
    x *= orig_norm
    return x

def load_data(cfg, n_stats=10_000, shuffle=5000, n_workers=16):
    """Create train and validation data from a config"""

    if cfg.format != "webdataset":
        raise NotImplementedError()

    try:
        data_path = hydra.utils.to_absolute_path(cfg.path)
    except TypeError:
        # We might specify multiple paths
        data_path = [hydra.utils.to_absolute_path(x) for x in cfg.path]
    stats_ds = wds.WebDataset(data_path).decode().to_tuple('img_feat.npy', 'latent.npy').shuffle(shuffle).batched(n_stats)
    stats_data = next(stats_ds.__iter__())
    
    stats = {
        "clip_features": train_utils.make_data_stats(torch.tensor(stats_data[0])),
        "w": train_utils.make_data_stats(torch.tensor(stats_data[1])),
    }

    ds = (
        wds.WebDataset(data_path)
            .shuffle(shuffle)
            .decode()
            .to_tuple('img_feat.npy', 'latent.npy')
            .batched(cfg.bs)
            .map_tuple(torch.tensor, torch.tensor)
        )
    if cfg.embed_noise_scale > 0:
        ds = ds.map_tuple(partial(add_noise, scale=cfg.embed_noise_scale), identity)
    ds = ds.map_tuple(identity, partial(train_utils.normalise_data, w_mean=stats["w"][0], w_std=stats["w"][1]))

    loader = wds.WebLoader(ds, num_workers=n_workers, batch_size=None)
    return stats,loader
