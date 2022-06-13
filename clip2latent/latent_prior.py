
import torch
import torch.nn.functional as F
from dalle2_pytorch.dalle2_pytorch import (BaseGaussianDiffusion,
                                           DiffusionPrior,
                                           DiffusionPriorNetwork,
                                           eval_decorator, exists, l2norm,
                                           noise_like, prob_mask_like)
from einops import rearrange, repeat
from torch import nn
from tqdm import tqdm

from clip2latent.train_utils import denormalise_data, normalise_data


class LatentPrior(DiffusionPrior):
    """Samples Latent vectors conditions on an encoding

    Supports two types of normalisation for input data:
    - z-score normalisation, i.e. use channels-wise means and std to normalise all channels
    of the latent to be mean=0 var=1. This makes a lot of sense for StyleGAN as we're learning latents
    considering the mean latent as the origin. (we could imagine even more complex setups, e.g. using P space
    where we inverse the final mapping relu and compute the PCA directions. But this seems like too much effort)
    - sqrt(dims) scaling. As suggested in the original dalle2_pytorch repo we might want to divide by the norm and 
    then multiply by sqrt(dims) to scale the norm to the corresponding dim Gaussian.

    We might want to do either/any of the above on the latent as well as the embedding (i.e. condition)

    Note as we're re-using the original DiffusionPrior the nomenclature is a bit confusing:

    Generate -> StyleGAN W latents -> equiv of image_embed in Dalle-2
    Condition on  -> CLIP embedding -> equiv of text_embed in Dalle-2

    By default we operate in w space which is 18 copies of the same 512 latent (for sg2 ffhq 1024). 
    So the diffusion model only needs to generate 1x512 latent and repeat this 18 times.
    Other times we might want to operate in the full w+ 18x512 space (or something in between).
    We could either use a 9,216 dim latent and reshape it, or have 18x512 tokens for the transformer
    as a first try we'll use 18 tokens.

    latent repeates dictates how the latents should repeat for output, e.g working in a coarse, mid, fine
    representation (w3) we migth have repeats of (4, 4, 10)
    """

    def __init__(self, *args,
        text_embed_scale=1.0,
        latent_stats=(None,None),
        embed_stats=(None,None),
        num_latents=1,
        latent_repeats=(18,),
        **kwargs):

        super().__init__(*args, **kwargs)
        self.skip_timesteps = 1
        # Original repo provides image_embed_scale
        self.text_embed_scale = text_embed_scale
        # Z-score stats
        self.register_buffer("latent_mean", latent_stats[0])
        self.register_buffer("latent_std", latent_stats[1])
        self.register_buffer("embed_mean", embed_stats[0])
        self.register_buffer("embed_std", embed_stats[1])
        self.num_latents = num_latents
        assert self.num_latents == len(latent_repeats), \
            f"Number of latents ({num_latents}) and length of repeats ({len(latent_repeats)}) don't match"
        self.latent_repeats = latent_repeats

    def set_timestep_skip(self, skip):
        """Support simple timestep respacing allowing a skip factor"""
        assert self.num_timesteps % skip == 0, f"{self.num_timesteps} timesteps don't divide equally into skip {skip}"
        self.skip_timesteps = skip
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(self.alphas_cumprod):
            if i%skip == 0:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
        new_betas = torch.tensor(new_betas)

        betas = new_betas
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis = 0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)

        self.posterior_variance[::skip] = posterior_variance
        self.posterior_mean_coef1[::skip] = posterior_mean_coef1
        self.posterior_mean_coef2[::skip] = posterior_mean_coef2

    def forward(self, text_embed=None, image_embed=None, *args, **kwargs):
        if exists(self.latent_mean):
            normalise_data(image_embed, self.latent_mean, self.latent_std)
        if exists(self.embed_mean):
            normalise_data(text_embed, self.embed_mean, self.embed_std)

        image_embed *= self.image_embed_scale
        text_embed *= self.text_embed_scale

        text_cond = dict(text_embed = text_embed)

        batch, device = image_embed.shape[0], image_embed.device
        times = torch.randint(0, self.num_timesteps, (batch,), device = device, dtype = torch.long)


        return self.p_losses(image_embed, times, text_cond = text_cond, *args, **kwargs)

    @torch.no_grad()
    def p_sample_loop(self, shape, text_cond, cond_scale = 1., show_progress=True):
        skip = self.skip_timesteps
        device = self.betas.device

        b = shape[0]
        image_embed = torch.randn(shape, device=device)

        if self.init_image_embed_l2norm:
            image_embed = l2norm(image_embed) * self.image_embed_scale

        loop = reversed(range(0, self.num_timesteps, skip))
        if show_progress:
            loop = tqdm(loop, desc='sampling loop time step', total=self.num_timesteps//skip)
        
        for i in loop:
            times = torch.full((b,), i, device = device, dtype = torch.long)
            image_embed = self.p_sample(image_embed, times, text_cond = text_cond, cond_scale = cond_scale)
        return image_embed

    @torch.no_grad()
    def sample(self, embed, cond_scale=1.0, truncation=1.0, **kwargs):
        """Generates latent vectors conditioned on embed.
        Latent vectors are the correct shape (and un-normalised) to pass into the StyleGAN model that generated them
        E.g. for FFHQ stylegan2 this would be (batch_size,18,512)

        Truncation only really makes sense if the data is normalised
        """

        batch = embed.shape[0]
        if self.num_latents == 1:
            shape = (batch, self.image_embed_dim)
        else:
            shape = (batch, self.num_latents, self.image_embed_dim)

        if exists(self.embed_mean):
            normalise_data(embed, self.embed_mean, self.embed_std)
        embed *= self.text_embed_scale

        cond = {"text_embed": embed}
        latents = self.p_sample_loop(shape, cond, cond_scale=cond_scale, **kwargs)

        # Denormalise
        latents = latents/self.image_embed_scale
        if exists(self.latent_mean):
            latents = denormalise_data(latents, self.latent_mean, self.latent_std)

        if truncation < 1.0:
            assert exists(self.latent_mean), "Can't do truncation without latent mean"
            latents = self.latent_mean + truncation*(latents - self.latent_mean)

        # Reformat for StyleGAN
        if self.num_latents == 1:
            latents = latents.unsqueeze(1)
        latent_repeats = torch.tensor(self.latent_repeats).to(torch.long).to(latents.device)
        latents = latents.repeat_interleave(latent_repeats, dim=1)
        
        return latents



class WPlusPriorNetwork(DiffusionPriorNetwork):
    def __init__(
        self,
        dim,
        num_timesteps = None,
        n_latents = 18,
        **kwargs
    ):
        super().__init__(dim, num_timesteps=num_timesteps, **kwargs)
        self.n_latents = n_latents
        self.learned_query = nn.Parameter(torch.randn(self.n_latents, dim))
        
    def forward(
        self,
        image_embed,
        diffusion_timesteps,
        *,
        text_embed,
        text_encodings = None,
        mask = None,
        cond_drop_prob = 0.
    ):
        # For w plus the image embed is now batchxn_layersx512
        batch, n_latents, dim, device, dtype = *image_embed.shape, image_embed.device, image_embed.dtype
        assert self.n_latents == n_latents, "N latents don't match"

        num_time_embeds, num_image_embeds, num_text_embeds = self.num_time_embeds, self.num_image_embeds, self.num_text_embeds

        text_embed = self.to_text_embeds(text_embed)
        # Don't do this
        # image_embed = self.to_image_embeds(image_embed)

        # make text encodings optional
        # although the paper seems to suggest it is present <--

        if not exists(text_encodings):
            text_encodings = torch.empty((batch, 0, dim), device = device, dtype = dtype)

        assert not mask, "why mask!?"
        
        if not exists(mask):
            mask = torch.ones((batch, text_encodings.shape[-2]), device = device, dtype = torch.bool)

        # classifier free guidance

        keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob, device = device)
        keep_mask = rearrange(keep_mask, 'b -> b 1')

        mask &= keep_mask
        
        # whether text embedding is masked or not depends on the classifier free guidance conditional masking

        keep_mask = repeat(keep_mask, 'b 1 -> b n', n = num_text_embeds)
        mask = torch.cat((mask, keep_mask), dim = 1)

        # whether text embedding is used for conditioning depends on whether text encodings are available for attention (for classifier free guidance, even though it seems from the paper it was not used in the prior ddpm, as the objective is different)
        # but let's just do it right

        if exists(mask):
            attend_padding = 1 + num_time_embeds + num_image_embeds + 2*n_latents - 2 # 1 for learned queries + number of image embeds + time embeds
            mask = F.pad(mask, (0, attend_padding), value = True) # extend mask for text embedding, noised image embedding, time step embedding, and learned query

        time_embed = self.to_time_embeds(diffusion_timesteps)

        learned_queries = repeat(self.learned_query, 'l d -> b l d', b = batch)

        tokens = torch.cat((
            text_encodings,
            text_embed,
            time_embed,
            image_embed,
            learned_queries
        ), dim = -2)

        # attend

        tokens = self.causal_transformer(tokens, mask = mask)

        # get learned query, which should predict the image embedding (per DDPM timestep)

        pred_image_embed = tokens[..., -n_latents:, :]

        return pred_image_embed

from dalle2_pytorch.dalle2_pytorch import default
