from clip2latent.latent_prior import LatentPrior, WPlusPriorNetwork
from dalle2_pytorch import DiffusionPriorNetwork
import torch
import pytest

@pytest.mark.parametrize(
    ["n_latents", "repeats"],
    [(1, (18,)), (3, (1, 4, 13)), (18, 18*(1,))],
    )
def test_prior_sample(n_latents, repeats):
    dim = 8
    bs = 2
    t = 4
    out_latents = 18
    if n_latents == 1:
        net = DiffusionPriorNetwork(dim=dim, num_timesteps=t, depth=2, dim_head=8, heads=2)
    else:
        net = WPlusPriorNetwork(n_latents=n_latents, dim=dim, num_timesteps=t, depth=2, dim_head=8, heads=2)
    prior = LatentPrior(net, image_embed_dim=dim, timesteps=t, latent_repeats=repeats, num_latents=n_latents)

    inp = torch.ones(bs, dim)
    out = prior.sample(inp)
    
    assert out.shape == (bs, out_latents, dim)