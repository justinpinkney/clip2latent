from omegaconf import OmegaConf
import torch
from pathlib import Path
import numpy as np
from PIL import Image
import train
from joblib import Parallel, delayed
from tqdm import tqdm

@torch.no_grad()
def run_model(diffusion_prior, text_sample, cond_scale, G, clip_model, sort_by_clip=True, n_samples=16):
    diffusion_prior.eval()
    text_features = clip_model.embed_text(text_sample)
    pred_w = diffusion_prior.sample(text_features.tile(n_samples, 1), cond_scale=cond_scale, show_progress=False)
    
    images = G.synthesis(pred_w)
    pred_w_clip_features = clip_model.embed_image(images)
    
    similarity = torch.cosine_similarity(pred_w_clip_features, text_features)
    
    if sort_by_clip:
        similarity, idxs = torch.sort(similarity, descending=True)
        images = images[idxs, ...]
    
    return images, similarity


def run_eval(cond_scale, skips, checkpoint, cfg_file, device, save_best, n_samples):

    output_dir = Path(f"eval_results/{skips}_{cond_scale}")
    output_dir.mkdir(exist_ok=True, parents=True)
    if save_best:
        best_dir = output_dir/"best"
        best_dir.mkdir(exist_ok=True)
        best_file = best_dir/"similarity.csv"
        best_file.unlink(missing_ok=True)

    cfg = OmegaConf.load(cfg_file)

    G, clip_model, trainer = train.load_models(cfg, device)
    trainer.load_state_dict(torch.load(checkpoint, map_location="cpu")["state_dict"], strict=False)
    diffusion_prior = trainer.diffusion_prior
    diffusion_prior.set_timestep_skip(skips)

    def save_im(im, f_name):
        Image.fromarray(im).save(f_name)

    best_scores = []

    with Parallel(n_jobs=n_samples) as parallel:
        for sample in tqdm(text_samples):
            label = sample.lower().replace(' ', '-')
            sample_dir = output_dir/"samples"/label
            sample_dir.mkdir(exist_ok=True, parents=True)
            similarity_file = sample_dir/"similarity.csv"

            images, similarity = run_model(diffusion_prior, sample, cond_scale, G, clip_model, n_samples=n_samples)
            images = 255*(images.clamp(-1,1)*0.5 + 0.5).permute(0, 2, 3, 1).cpu()
            images = images.numpy().astype(np.uint8)

            filenames = [sample_dir/f'{idx:03}.png' for idx in range(images.shape[0])]

            with open(similarity_file, 'wt') as f:
                for idx, (s, filename) in enumerate(zip(similarity, filenames)):
                    f.write(f'{filename.stem}, {s}\n')

            parallel(delayed(save_im)(im, f_name) for im, f_name in zip(images, filenames))

            if save_best:
                best_im = images[0]
                best_score = similarity[0]
                best_filename = best_dir/f'{label}.png'
                Image.fromarray(best_im).save(best_filename)
                with open(best_file, 'at+') as f:
                    f.write(f'"{sample}", {best_score}\n')
                best_scores.append(best_score.unsqueeze(0))

    mean_score = torch.cat(best_scores, dim=0).mean().cpu()

    print('------------')
    print(f'Timestep skips: {skips}, condition_scale: {cond_scale}')
    print(f'Mean CLIP score: {mean_score}')
    print('------------')

    return mean_score

if __name__ == "__main__":
    skips = (100, 10, 1)
    cond_scales = (1, 1.5, 2, 3, 5, 9, 17)
    with open("test.txt", 'rt') as f:
        text_samples = f.read().splitlines()

    with open("all_scores.csv", 'wt') as f:
        f.write(f"skips, cond_scale, score\n")


    checkpoint = "best.ckpt"
    cfg_file = "best.yaml"
    device = "cuda:0"
    save_best = True
    n_samples = 16


    for s in skips:
        for c in cond_scales:
            score = run_eval(c, s, checkpoint, cfg_file, device, save_best, n_samples)
            with open("all_scores.csv", 'at+') as f:
                f.write(f"{s}, {c}, {score}\n")