from typing import List
from omegaconf import OmegaConf
import torch
from pathlib import Path
import numpy as np
from PIL import Image
import train
from joblib import Parallel, delayed
from tqdm import tqdm
from datetime import datetime
import typer

@torch.no_grad()
def run_diffusion(diffusion_prior, clip_model, text_samples, cond_scale, n_samples, truncation=1.0):
    diffusion_prior.eval()
    text_features = clip_model.embed_text(text_samples)
    text_features = text_features.repeat_interleave(n_samples, dim=0)
    pred_w = diffusion_prior.sample(text_features, cond_scale=cond_scale, show_progress=True, truncation=truncation)
    return pred_w    

@torch.no_grad()
def run_synthesis(pred_w, sample, G, clip_model, sort_by_clip=True):
    
    text_features = clip_model.embed_text(sample)
    text_features = text_features.tile(pred_w.shape[0], 1)
    
    images = G.synthesis(pred_w)
    pred_w_clip_features = clip_model.embed_image(images)
    
    similarity = torch.cosine_similarity(pred_w_clip_features, text_features)
    
    if sort_by_clip:
        similarity, idxs = torch.sort(similarity, descending=True)
        images = images[idxs, ...]
    
    return images, similarity


def run_eval(cond_scale, skips, text_samples, checkpoint, cfg_file, device, n_samples, write_results, truncation):

    if write_results:
        output_dir = Path(f"eval_results/{skips}_{cond_scale}")
        output_dir.mkdir(exist_ok=True, parents=True)
        best_dir = output_dir/"best"
        best_dir.mkdir(exist_ok=True)
        best_file = best_dir/"similarity.csv"
        best_file.unlink(missing_ok=True)

    cfg = OmegaConf.load(cfg_file)

    G, clip_model, trainer = train.load_models(cfg, device)
    trainer.load_state_dict(torch.load(checkpoint, map_location="cpu")["state_dict"], strict=False)
    diffusion_prior = trainer.ema_diffusion_prior.ema_model
    diffusion_prior.set_timestep_skip(skips)

    def save_im(im, f_name):
        Image.fromarray(im).save(f_name)

    best_scores = []

    w_pred = run_diffusion(diffusion_prior, clip_model, text_samples, cond_scale, n_samples, truncation=truncation)
    w_preds = w_pred.split(n_samples)

    with Parallel(n_jobs=n_samples) as parallel:
        for sample, w_pred in tqdm(zip(text_samples, w_preds), total=len(text_samples)):
            if write_results:
                label = sample.lower().replace(' ', '-')
                sample_dir = output_dir/"samples"/label
                sample_dir.mkdir(exist_ok=True, parents=True)
                similarity_file = sample_dir/"similarity.csv"

            images, similarity = run_synthesis(w_pred, sample, G, clip_model)
            images = 255*(images.clamp(-1,1)*0.5 + 0.5).permute(0, 2, 3, 1).cpu()
            images = images.numpy().astype(np.uint8)

            if write_results:
                filenames = [sample_dir/f'{idx:03}.png' for idx in range(images.shape[0])]

                with open(similarity_file, 'wt') as f:
                    for idx, (s, filename) in enumerate(zip(similarity, filenames)):
                        f.write(f'{filename.stem}, {s}\n')

                parallel(delayed(save_im)(im, f_name) for im, f_name in zip(images, filenames))

            best_im = images[0]
            best_score = similarity[0]
            mean_score = similarity.mean()
            # Is std worth recording?
            if write_results:
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

def main(
    skips:List[int]=[1,100,250],
    cond_scales:List[float]=[1,1.05,1.1,1.2,1.3,1.5,1.75,2,2.5,3,4,5,10],
    write_results:bool = True,
    checkpoint:str = "best.ckpt",
    cfg_file:str= "best.yaml",
    device:str= "cuda:0",
    n_samples:int= 16,
    truncation:float=1.0,
    ):

    typer.echo(f"Running skips: {skips}")
    typer.echo(f"Running cond scales: {cond_scales}")

    with open("test.txt", 'rt') as f:
        text_samples = f.read().splitlines()
        text_samples = ["a photograph of " + x for x in text_samples]

    score_filename = f"all_scores_{datetime.now()}.csv"
    if write_results:
        with open(score_filename, 'wt') as f:
            f.write(f"skips, cond_scale, score\n")

    for s in skips:
        for c in cond_scales:
            score = run_eval(c, s, text_samples, checkpoint, cfg_file, device, n_samples, write_results, truncation=truncation)
            if write_results:
                with open(score_filename, 'at+') as f:
                    f.write(f"{s}, {c}, {score}\n")


if __name__ == "__main__":
    typer.run(main)