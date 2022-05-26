
import torch
from einops import rearrange, repeat
from tqdm import tqdm
import torch.nn.functional as F

from dalle2_pytorch.dalle2_pytorch import (BaseGaussianDiffusion, DiffusionPrior, eval_decorator, 
                                           l2norm, noise_like)


class ZWPrior(DiffusionPrior):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_timesteps = 1

    def set_timestep_skip(self, skip):
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
        text_cond = dict(text_embed = text_embed)

        # timestep conditioning from ddpm
        batch, device = image_embed.shape[0], image_embed.device
        times = torch.randint(0, self.num_timesteps, (batch,), device = device, dtype = torch.long)

        # scale image embed (Katherine)
        image_embed *= self.image_embed_scale

        # calculate forward loss

        return self.p_losses(image_embed, times, text_cond = text_cond, *args, **kwargs)

    @torch.no_grad()
    def p_sample_loop(self, shape, text_cond, cond_scale = 1.):
        skip = self.skip_timesteps
        device = self.betas.device

        b = shape[0]
        image_embed = torch.randn(shape, device=device)

        if self.init_image_embed_l2norm:
            image_embed = l2norm(image_embed) * self.image_embed_scale

        for i in tqdm(reversed(range(0, self.num_timesteps, skip)), desc='sampling loop time step', total=self.num_timesteps//skip):
            times = torch.full((b,), i, device = device, dtype = torch.long)
            image_embed = self.p_sample(image_embed, times, text_cond = text_cond, cond_scale = cond_scale)
        return image_embed

from dalle2_pytorch.dalle2_pytorch import exists, prob_mask_like, DiffusionPriorNetwork
from torch import nn

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


class LatentPrior(BaseGaussianDiffusion):
    def __init__(
        self,
        net,
        *,
        latent_dim = 512,
        embed_dim = 512,
        timesteps = 1000,
        cond_drop_prob = 0.,
        loss_type = "l1",
        predict_x_start = True,
        beta_schedule = "cosine",
        sampling_clamp_l2norm = False,
        training_clamp_l2norm = False,
        init_image_embed_l2norm = False,
        image_embed_scale = None,           # this is for scaling the l2-normed image embedding, so it is more suitable for gaussian diffusion, as outlined by Katherine (@crowsonkb) https://github.com/lucidrains/DALLE2-pytorch/issues/60#issue-1226116132
        ):
        super().__init__(
            beta_schedule = beta_schedule,
            timesteps = timesteps,
            loss_type = loss_type
        )

        self.net = net
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim

        self.cond_drop_prob = cond_drop_prob

        # in paper, they do not predict the noise, but predict x0 directly for image embedding, claiming empirically better results. I'll just offer both.
        self.predict_x_start = predict_x_start

        # @crowsonkb 's suggestion - https://github.com/lucidrains/DALLE2-pytorch/issues/60#issue-1226116132
        self.image_embed_scale = default(image_embed_scale, self.embed_dim ** 0.5)

        # whether to force an l2norm, similar to clipping denoised, when sampling
        self.sampling_clamp_l2norm = sampling_clamp_l2norm
        self.training_clamp_l2norm = training_clamp_l2norm
        self.init_image_embed_l2norm = init_image_embed_l2norm

    def p_mean_variance(self, x, t, model_kwargs, clip_denoised: bool):
        pred = self.net(x, t, **model_kwargs)

        if self.predict_x_start:
            x_recon = pred
            # not 100% sure of this above line - for any spectators, let me know in the github issues (or through a pull request) if you know how to correctly do this
            # i'll be rereading https://arxiv.org/abs/2111.14822, where i think a similar approach is taken
        else:
            x_recon = self.predict_start_from_noise(x, t = t, noise = pred)

        if clip_denoised and not self.predict_x_start:
            x_recon.clamp_(-1., 1.)

        if self.predict_x_start and self.sampling_clamp_l2norm:
            x_recon = l2norm(x_recon) * self.image_embed_scale

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.inference_mode()
    def p_sample(self, x, t, model_kwargs = None, clip_denoised = False, repeat_noise = False, super_cond=0):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x = x, t = t, model_kwargs = model_kwargs, clip_denoised = clip_denoised)
        if super_cond:
            model_kwargs = {k: 0*v for k, v in model_kwargs.items()}
            uncond_model_mean, _, uncond_model_log_variance = self.p_mean_variance(x = x, t = t, model_kwargs = model_kwargs, clip_denoised = clip_denoised)
            model_mean = (1+super_cond)*model_mean - super_cond*uncond_model_mean
            model_log_variance = (1+super_cond)*model_log_variance - super_cond*uncond_model_log_variance
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.inference_mode()
    def p_sample_loop(self, shape, model_kwargs, super_cond=0):
        device = self.betas.device

        b = shape[0]
        image_embed = torch.randn(shape, device=device)

        if self.init_image_embed_l2norm:
            image_embed = l2norm(image_embed) * self.image_embed_scale

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            times = torch.full((b,), i, device = device, dtype = torch.long)
            image_embed = self.p_sample(image_embed, times, model_kwargs = model_kwargs, super_cond=super_cond)

        return image_embed

    def p_losses(self, image_embed, times, model_kwargs, noise = None):
        noise = default(noise, lambda: torch.randn_like(image_embed))

        image_embed_noisy = self.q_sample(x_start = image_embed, t = times, noise = noise)

        pred = self.net(
            image_embed_noisy,
            times,
            cond_drop_prob = self.cond_drop_prob,
            **model_kwargs
        )

        if self.predict_x_start and self.training_clamp_l2norm:
            pred = l2norm(pred) * self.image_embed_scale

        target = noise if not self.predict_x_start else image_embed

        loss = self.loss_fn(pred, target)
        return loss

    @torch.inference_mode()
    @eval_decorator
    def sample_batch_size(self, batch_size, model_kwargs):
        device = self.betas.device
        shape = (batch_size, self.embed_dim)

        img = torch.randn(shape, device = device)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            img = self.p_sample(img, torch.full((batch_size,), i, device = device, dtype = torch.long), model_kwargs = model_kwargs)
        return img

    @torch.inference_mode()
    @eval_decorator
    def sample(self, text, num_samples_per_batch = 2):
        # in the paper, what they did was
        # sample 2 image embeddings, choose the top 1 similarity, as judged by CLIP
        text = repeat(text, 'b ... -> (b r) ...', r = num_samples_per_batch)

        batch_size = text.shape[0]
        image_embed_dim = self.embed_dim

        text_embed, text_encodings, text_mask = self.clip.embed_text(text)

        text_cond = dict(text_embed = text_embed)

        if self.condition_on_text_encodings:
            text_cond = {**text_cond, 'text_encodings': text_encodings, 'mask': text_mask}

        image_embeds = self.p_sample_loop((batch_size, image_embed_dim), model_kwargs = text_cond)

        # retrieve original unscaled image embed

        image_embeds /= self.image_embed_scale

        text_embeds = text_cond['text_embed']

        text_embeds = rearrange(text_embeds, '(b r) d -> b r d', r = num_samples_per_batch)
        image_embeds = rearrange(image_embeds, '(b r) d -> b r d', r = num_samples_per_batch)

        text_image_sims = torch.einsum('b r d, b r d -> b r', l2norm(text_embeds), l2norm(image_embeds))
        top_sim_indices = text_image_sims.topk(k = 1).indices

        top_sim_indices = repeat(top_sim_indices, 'b 1 -> b 1 d', d = image_embed_dim)

        top_image_embeds = image_embeds.gather(1, top_sim_indices)
        return rearrange(top_image_embeds, 'b 1 d -> b d')

    def forward(
        self,
        latent = None,
        embed = None,  # as well as CLIP text encodings
        *args,
        **kwargs
    ):
        
        embed_cond = {'embed': embed}
        # timestep conditioning from ddpm
        batch, device = latent.shape[0], latent.device
        times = torch.randint(0, self.num_timesteps, (batch,), device = device, dtype = torch.long)

        # scale image embed (Katherine)
        latent *= self.image_embed_scale

        # calculate forward loss
        return self.p_losses(latent, times, model_kwargs = embed_cond, *args, **kwargs)
